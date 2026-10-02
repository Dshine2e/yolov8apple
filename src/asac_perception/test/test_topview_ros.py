"""ROS contracts with deterministic segmentation; actual model E2E is a separate script."""

import json
import time
from types import SimpleNamespace

from builtin_interfaces.msg import Time as Stamp
from geometry_msgs.msg import TransformStamped
import numpy as np
import pytest
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_ros import Buffer, ExtrapolationException

from asac_interfaces.msg import AppleObservationArray
from asac_perception.segmentation import Instance
from asac_perception.topview_node import TopviewNode


class DeterministicSegmenter:
    model = SimpleNamespace(names={0: "apple"})

    def __init__(self, *args):
        pass

    def predict(self, bgr, *args):
        v, u = np.indices(bgr.shape[:2])
        mask = (u - 40) ** 2 + (v - 30) ** 2 < 12**2
        return [Instance(np.array([28.0, 18.0, 53.0, 43.0]), 0.9, 0, mask)]


@pytest.fixture
def nodes(request):
    rclpy.init()
    node = TopviewNode(
        segmenter_factory=DeterministicSegmenter,
        namespace="/asac",
        parameter_overrides=[
            Parameter("model_path", value="unused.pt"),
            Parameter("expected_optical_frame", value="optical"),
            Parameter("base_frame", value="robot_root"),
            Parameter("calibration_verified", value=True),
            Parameter("rgb_only", value=bool(getattr(request, "param", False))),
        ],
    )
    peer = Node("topview_test_peer")
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    executor.add_node(peer)
    try:
        yield node, peer, executor
    finally:
        executor.shutdown()
        node.destroy_node()
        peer.destroy_node()
        rclpy.shutdown()


def frames(node):
    v, u = np.indices((60, 80))
    rgb = np.stack((u, v, u + v), axis=-1).astype(np.uint8)
    color = node.bridge.cv2_to_imgmsg(rgb, "bgr8")
    color.header.frame_id = "optical"
    color.header.stamp = node.get_clock().now().to_msg()
    # Ray-sphere intersections. This is an explicit synthetic geometry input, not E2E.
    dx, dy = (u - 40) / 240, (v - 30) / 240
    a = dx**2 + dy**2 + 1
    discriminant = 0.8**2 - a * (0.8**2 - 0.04**2)
    z = np.full((60, 80), np.nan, np.float32)
    valid = discriminant >= 0
    z[valid] = (0.8 - np.sqrt(discriminant[valid])) / a[valid]
    depth = node.bridge.cv2_to_imgmsg(z, "32FC1")
    depth.header = color.header
    info = CameraInfo(header=color.header, width=80, height=60)
    info.k = [240.0, 0.0, 40.0, 0.0, 240.0, 30.0, 0.0, 0.0, 1.0]
    info.distortion_model = "plumb_bob"
    return color, info, depth


def install_tf(node):
    transform = TransformStamped()
    transform.header.frame_id, transform.child_frame_id = "robot_root", "optical"
    transform.transform.translation.x, transform.transform.translation.z = 0.4, 1.0
    transform.transform.rotation.x = 1.0
    transform.transform.rotation.w = 0.0
    node.tf_buffer.set_transform_static(transform, "test")


@pytest.mark.parametrize(
    "fault", ["none", "uncalibrated", "missing_tf", "no_depth", "units", "frame"]
)
def test_validity_crops_capture_tf_and_json(nodes, fault):
    node, peer, executor = nodes
    arrays, payloads = [], []
    peer.create_subscription(AppleObservationArray, "/asac/observations", arrays.append, 1)
    peer.create_subscription(String, "/asac/observations_json", payloads.append, 1)
    end = time.monotonic() + 2
    while not node.observations_pub.get_subscription_count() and time.monotonic() < end:
        executor.spin_once(timeout_sec=0.01)
    color, info, depth = frames(node)
    if fault != "missing_tf":
        install_tf(node)
    if fault == "uncalibrated":
        node.config["calibration_verified"] = False
    elif fault == "no_depth":
        node.config["rgb_only"] = True
    elif fault == "units":
        values = node.bridge.imgmsg_to_cv2(depth).copy() * 1000
        depth = node.bridge.cv2_to_imgmsg(values, "32FC1")
        depth.header = color.header
    elif fault == "frame":
        color.header.frame_id = "changed_optical"
    node._receive(color, info, depth)
    node._infer()
    end = time.monotonic() + 2
    while (not arrays or not payloads) and time.monotonic() < end:
        executor.spin_once(timeout_sec=0.01)
    assert arrays and payloads
    msg = arrays[0].apples[0]
    payload = json.loads(payloads[0].data)["observations"][0]
    assert arrays[0].header == color.header == msg.rgb_crop.header == msg.mask_crop.header
    assert msg.rgb_crop.width == msg.mask_crop.width == msg.roi.width
    assert msg.rgb_crop.height == msg.mask_crop.height == msg.roi.height
    rgb = node.bridge.imgmsg_to_cv2(msg.rgb_crop)
    original = node.bridge.imgmsg_to_cv2(color)
    np.testing.assert_equal(
        rgb,
        original[
            msg.roi.y_offset: msg.roi.y_offset + msg.roi.height,
            msg.roi.x_offset: msg.roi.x_offset + msg.roi.width,
        ],
    )
    if fault == "none":
        assert msg.center_camera_valid and msg.center_base_valid and msg.direction_valid
        np.testing.assert_allclose(
            [msg.center_base.point.x, msg.center_base.point.y, msg.center_base.point.z],
            [0.4, 0, 0.2],
            atol=0.002,
        )
        assert msg.radius_m == pytest.approx(0.04, abs=0.002)
        assert msg.direction_base.vector.z > 0.99
    elif fault in ("uncalibrated", "missing_tf"):
        assert msg.center_camera_valid and msg.radius_valid
        assert not msg.center_base_valid and not msg.direction_valid
        assert payload["center_base"] is None and payload["direction_base"] is None
    else:
        assert not msg.center_camera_valid and not msg.radius_valid
        assert payload["radius_m"] is None and payload["center_camera"] is None
        assert payload["reasons"]
    assert "NaN" not in payloads[0].data


def test_tf_lookup_uses_capture_time_and_no_latest_fallback():
    rclpy.init()
    try:
        buffer = Buffer()
        for sec, x in [(10, 1.0), (20, 3.0)]:
            tf = TransformStamped()
            tf.header.frame_id, tf.child_frame_id = "root", "optical"
            tf.header.stamp = Stamp(sec=sec)
            tf.transform.translation.x = x
            tf.transform.rotation.w = 1.0
            buffer.set_transform(tf, "test")
        assert buffer.lookup_transform(
            "root", "optical", Time(seconds=15)
        ).transform.translation.x == pytest.approx(2.0)
        assert buffer.lookup_transform(
            "optical", "root", Time(seconds=15)
        ).transform.translation.x == pytest.approx(-2.0)
        with pytest.raises(ExtrapolationException):
            buffer.lookup_transform("root", "optical", Time(seconds=21))
    finally:
        rclpy.shutdown()


def test_stale_result_never_updates_tracker_or_publishes(nodes):
    node, _, _ = nodes
    color, info, depth = frames(node)
    color.header.stamp.sec -= 10
    node._receive(color, info, depth)
    node._infer()
    assert node.tracker.counter == 0


@pytest.mark.parametrize("nodes", [True], indirect=True)
def test_rgb_only_dds_without_camera_info(nodes):
    node, peer, executor = nodes
    from sensor_msgs.msg import Image

    received = []
    publisher = peer.create_publisher(Image, node.config["color_topic"], 1)
    peer.create_subscription(AppleObservationArray, "/asac/observations", received.append, 1)
    end = time.monotonic() + 2
    while (
        not publisher.get_subscription_count()
        or not node.observations_pub.get_subscription_count()
    ) and time.monotonic() < end:
        executor.spin_once(timeout_sec=0.01)
    end = time.monotonic() + 5
    while not received and time.monotonic() < end:
        color, _, _ = frames(node)
        publisher.publish(color)
        executor.spin_once(timeout_sec=0.01)
    assert received and received[0].apples
    apple = received[0].apples[0]
    assert apple.circularity_valid and apple.radius_px > 0
    assert not apple.radius_valid and not apple.center_camera_valid and not apple.direction_valid
    assert "rgb_only_no_depth" in apple.reasons


def test_delayed_inference_discarded_before_tracking(nodes):
    node, _, _ = nodes
    node.config["max_frame_age_sec"] = 0.03
    predict = node.segmenter.predict

    def delayed(*args):
        time.sleep(0.05)
        return predict(*args)

    node.segmenter.predict = delayed
    node._receive(*frames(node))
    node._infer()
    assert node.tracker.counter == 0
