"""실제 ROS 메시지/DDS를 사용하며 YOLO만 결정론적 대역으로 교체한다."""

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image

from asac_interfaces.msg import DetectedObjectArray
from asac_perception.detector_node import DetectorNode


class FakeBoxes:
    xyxy = np.array([[8, 8, 24, 24]], dtype=float)
    conf = np.array([0.9])
    cls = np.array([0])

    def cpu(self):
        return self

    def numpy(self):
        return self


class FakeModel:
    task = 'detect'
    names = {0: 'apple', 1: 'other'}

    def predict(self, **kwargs):
        assert kwargs['source'].shape == (32, 32, 3)
        assert kwargs['classes'] == [0]
        return [SimpleNamespace(boxes=FakeBoxes(), names={0: 'apple'})]


@pytest.fixture
def ros_nodes(tmp_path):
    rclpy.init()
    weights = tmp_path / 'fake.pt'
    weights.touch()
    detector = DetectorNode(
        model_factory=lambda _: FakeModel(), namespace='/asac',
        parameter_overrides=[
            Parameter('model_path', value=str(weights)),
            Parameter('max_frame_age_sec', value=5.0),
            Parameter('target_classes', value='apple'),
        ],
    )
    peer = Node('test_peer')
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(detector)
    executor.add_node(peer)
    try:
        yield detector, peer, executor
    finally:
        executor.shutdown()
        detector.destroy_node()
        peer.destroy_node()
        rclpy.shutdown()


def frames(detector, value=1000):
    color = detector.bridge.cv2_to_imgmsg(np.zeros((32, 32, 3), dtype=np.uint8), 'bgr8')
    depth = detector.bridge.cv2_to_imgmsg(np.full((32, 32), value, dtype=np.uint16), '16UC1')
    color.header.frame_id = 'camera_color_optical_frame'
    color.header.stamp = detector.get_clock().now().to_msg()
    depth.header = color.header
    info = CameraInfo(header=color.header, height=32, width=32)
    info.k = [100., 0., 16., 0., 100., 16., 0., 0., 1.]
    info.distortion_model = 'plumb_bob'
    return color, depth, info


def spin_until(executor, predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not predicate():
        executor.spin_once(timeout_sec=0.02)
    assert predicate(), 'Timed out waiting for ROS data'


@pytest.mark.parametrize('depth_value', [1000, 0])
def test_synchronized_input_to_output(ros_nodes, depth_value):
    detector, peer, executor = ros_nodes
    received, images = [], []
    peer.create_subscription(DetectedObjectArray, '/asac/detections', received.append, 1)
    peer.create_subscription(Image, '/asac/annotated_image', images.append, 1)
    publishers = [peer.create_publisher(kind, detector.config[topic], 1) for kind, topic in (
        (Image, 'color_topic'), (Image, 'depth_topic'), (CameraInfo, 'camera_info_topic'),
    )]
    spin_until(executor, lambda: all(p.get_subscription_count() for p in publishers)
               and detector.result_pub.get_subscription_count()
               and detector.image_pub.get_subscription_count())
    color, depth, info = frames(detector, depth_value)
    for pub, message in zip(publishers, (color, depth, info)):
        pub.publish(message)
    spin_until(executor, lambda: bool(received) and bool(images))
    result = received[0]
    assert result.header == color.header == images[0].header
    assert len(result.objects) == 1
    obj = result.objects[0]
    assert obj.class_name == 'apple'
    assert obj.confidence == pytest.approx(0.9)
    assert obj.depth_valid == (depth_value != 0)
    if obj.depth_valid:
        assert obj.depth_m == pytest.approx(1.0)
        assert obj.position.x == pytest.approx(0.0)
        assert obj.position.y == pytest.approx(0.0)
        assert obj.range_m == pytest.approx(1.0)
    else:
        assert math.isnan(obj.position.z)
        assert obj.valid_depth_samples == 0


def test_wrong_frame_and_resolution_rejected(ros_nodes):
    detector, _, _ = ros_nodes
    color, depth, info = frames(detector)
    # 각 영상의 Header 객체를 공유하지 않도록 별도 깊이 헤더를 설정한다.
    depth.header = type(color.header)(
        stamp=color.header.stamp, frame_id='camera_depth_optical_frame',
    )
    with pytest.raises(ValueError, match='optical frame'):
        detector._process(color, depth, info)
    depth.header = color.header
    depth.width = 16
    with pytest.raises(ValueError, match='resolutions'):
        detector._process(color, depth, info)


def test_old_frame_rejected(ros_nodes):
    detector, _, _ = ros_nodes
    color, _, _ = frames(detector)
    color.header.stamp.sec -= 10
    assert not detector._fresh(color)


def test_no_detections_publishes_empty_array(ros_nodes):
    detector, peer, executor = ros_nodes
    received = []
    peer.create_subscription(DetectedObjectArray, '/asac/detections', received.append, 1)
    spin_until(executor, lambda: bool(detector.result_pub.get_subscription_count()))
    empty = SimpleNamespace(
        xyxy=np.empty((0, 4)), conf=np.empty(0), cls=np.empty(0),
    )
    empty.cpu = lambda: empty
    empty.numpy = lambda: empty
    detector.model.predict = lambda **_: [SimpleNamespace(boxes=empty, names={0: 'apple'})]
    detector._process(*frames(detector))
    spin_until(executor, lambda: bool(received))
    assert not received[0].objects


@pytest.mark.parametrize('deadline_sec', [0.03, 5.0])
def test_slow_inference_respects_result_deadline(ros_nodes, deadline_sec):
    detector, peer, executor = ros_nodes
    received = []
    peer.create_subscription(DetectedObjectArray, '/asac/detections', received.append, 1)
    spin_until(executor, lambda: bool(detector.result_pub.get_subscription_count()))
    detector.config['max_frame_age_sec'] = deadline_sec
    original_predict = detector.model.predict

    def slow_predict(**kwargs):
        time.sleep(0.08)
        return original_predict(**kwargs)

    detector.model.predict = slow_predict
    rgb, depth, info = frames(detector)
    detector._process(rgb, depth, info)
    if deadline_sec > 0.08:
        spin_until(executor, lambda: bool(received))
        assert received[0].header == rgb.header
    else:
        # 구독자가 메시지를 처리할 시간을 준 뒤 폐기 여부를 확인한다.
        end = time.monotonic() + 0.2
        while time.monotonic() < end:
            executor.spin_once(timeout_sec=0.02)
        assert not received
