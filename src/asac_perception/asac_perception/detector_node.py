"""촬영 시각이 일치하는 RGB-D 영상으로 YOLOv8 탐지 및 깊이 추정."""

from pathlib import Path
import threading
import time
from typing import Callable, Optional

import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from asac_interfaces.msg import DetectedObject, DetectedObjectArray
from asac_perception.depth import depth_in_meters, estimate_position, normalized_ray


def load_model(path: str, *, device: str, image_size: int, cpu_threads: int):
    import torch
    from ultralytics import YOLO
    model = YOLO(path, task='detect')
    warmup = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    # 첫 predict에서 backend 초기화/fuse 및 CPU thread 수 재설정이 일어난다.
    model.predict(source=warmup, imgsz=image_size, device=device, verbose=False)
    if device.lower() == 'cpu':
        torch.set_num_threads(cpu_threads)
    # 구독 시작 전에 설정된 thread 수로도 한 번 실행한다.
    model.predict(source=warmup, imgsz=image_size, device=device, verbose=False)
    return model


class DetectorNode(Node):
    def __init__(self, model_factory: Optional[Callable] = None, **kwargs) -> None:
        super().__init__('detector', **kwargs)
        defaults = {
            'model_path': '',
            'color_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'confidence': 0.5, 'iou': 0.45, 'image_size': 416,
            'device': 'cpu', 'cpu_threads': 2,
            'max_detections': 100, 'inference_rate_hz': 5.0,
            'target_classes': '',
            'sync_slop_sec': 0.05, 'sync_queue_size': 5, 'max_frame_age_sec': 2.0,
            'depth_roi_fraction': 0.3, 'min_depth_m': 0.2, 'max_depth_m': 6.0,
            'min_depth_samples': 5, 'publish_annotated_image': True,
        }
        self.config = {
            name: self.declare_parameter(
                name, value, ParameterDescriptor(read_only=True),
            ).value for name, value in defaults.items()
        }
        self._validate_config()
        model_path = Path(self.config['model_path']).expanduser().resolve()
        if not model_path.is_file() or model_path.suffix != '.pt':
            raise ValueError(f'model_path must point to an existing YOLOv8 .pt: {model_path}')
        self.model = model_factory(str(model_path)) if model_factory is not None else load_model(
            str(model_path), device=self.config['device'], image_size=self.config['image_size'],
            cpu_threads=self.config['cpu_threads'],
        )
        if getattr(self.model, 'task', None) != 'detect':
            raise ValueError('Only YOLO object detection models are supported')
        requested = {name.strip() for name in self.config['target_classes'].split(',')
                     if name.strip()}
        names = self.model.names
        names = names if isinstance(names, dict) else dict(enumerate(names))
        missing = requested - set(names.values())
        if missing:
            raise ValueError(f'Target classes not in model: {sorted(missing)}; names={names}')
        self.class_ids = sorted(key for key, name in names.items() if name in requested)
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = None
        self._last_input = time.monotonic()
        self.result_pub = self.create_publisher(DetectedObjectArray, 'detections', 1)
        self.image_pub = self.create_publisher(Image, 'annotated_image', 1)
        # Best-effort 구독은 RealSense의 reliable / sensor-data 발행 양쪽과 호환된다.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.subscribers = [
            message_filters.Subscriber(self, msg_type, self.config[topic], qos_profile=qos)
            for msg_type, topic in [
                (Image, 'color_topic'), (Image, 'depth_topic'),
                (CameraInfo, 'camera_info_topic'),
            ]
        ]
        self.sync = message_filters.ApproximateTimeSynchronizer(
            self.subscribers, self.config['sync_queue_size'], self.config['sync_slop_sec'],
            allow_headerless=False,
        )
        self.sync.registerCallback(self._receive)
        # 구독 callback과 추론 callback은 서로 다른 그룹에서 실행한다.
        self._inference_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(
            1.0 / self.config['inference_rate_hz'], self._infer_latest,
            callback_group=self._inference_group,
        )
        self.create_timer(5.0, self._watchdog)
        self.get_logger().info(
            f'Model ready: {model_path}; device={self.config["device"]}, '
            f'image_size={self.config["image_size"]}, cpu_threads={self.config["cpu_threads"]}, '
            f'max_frame_age_sec={self.config["max_frame_age_sec"]}'
        )

    def _validate_config(self) -> None:
        c = self.config
        for key in ('confidence', 'iou', 'depth_roi_fraction'):
            if not np.isfinite(c[key]) or not 0 < c[key] <= 1:
                raise ValueError(f'{key} must be in (0, 1]')
        for key in ('inference_rate_hz', 'sync_slop_sec', 'max_frame_age_sec'):
            if not np.isfinite(c[key]) or c[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        for key in (
            'image_size', 'sync_queue_size', 'max_detections', 'min_depth_samples', 'cpu_threads',
        ):
            if c[key] < 1:
                raise ValueError(f'{key} must be positive')
        if not (np.isfinite(c['min_depth_m']) and np.isfinite(c['max_depth_m'])
                and 0 < c['min_depth_m'] < c['max_depth_m']):
            raise ValueError('Require 0 < min_depth_m < max_depth_m')
        if any(not c[key] for key in ('color_topic', 'depth_topic', 'camera_info_topic')):
            raise ValueError('Input topic names must not be empty')

    def _receive(self, color: Image, depth: Image, info: CameraInfo) -> None:
        with self._lock:
            self._latest = (color, depth, info)
            self._last_input = time.monotonic()

    def _watchdog(self) -> None:
        with self._lock:
            age = time.monotonic() - self._last_input
        if age > 5.0:
            self.get_logger().warning(
                'No synchronized RGB/depth/CameraInfo. Check topics, timestamps, '
                'QoS and align_depth.enable.', throttle_duration_sec=5.0,
            )

    def _frame_age_sec(self, color: Image) -> float:
        return (self.get_clock().now() - Time.from_msg(color.header.stamp)).nanoseconds / 1e9

    def _fresh(self, color: Image) -> bool:
        age = self._frame_age_sec(color)
        return -self.config['sync_slop_sec'] <= age <= self.config['max_frame_age_sec']

    def _infer_latest(self) -> None:
        if not self.context.ok():
            return
        with self._lock:
            frame, self._latest = self._latest, None
        if frame is None:
            return
        color, depth, info = frame
        if not self._fresh(color):
            self.get_logger().warning(
                f'Dropping stale RGB-D frame: age={self._frame_age_sec(color):.3f}s, '
                f'limit={self.config["max_frame_age_sec"]:.3f}s', throttle_duration_sec=5.0,
            )
            return
        try:
            self._process(color, depth, info)
        except (ValueError, RuntimeError, CvBridgeError, cv2.error) as error:
            # 변환/추론 실패는 가짜 탐지 결과로 대체하지 않고 원인을 기록한다.
            self.get_logger().error(f'RGB-D processing failed: {error}', throttle_duration_sec=5.0)

    def _process(self, color: Image, depth: Image, info: CameraInfo) -> None:
        process_started = time.perf_counter()
        input_age = self._frame_age_sec(color)
        if not color.header.frame_id or not (
            color.header.frame_id == depth.header.frame_id == info.header.frame_id
        ):
            raise ValueError('RGB, aligned depth and CameraInfo must share the color optical frame')
        if (color.width, color.height) != (depth.width, depth.height) or (
            color.width, color.height
        ) != (info.width, info.height):
            raise ValueError('RGB, aligned depth and CameraInfo resolutions must match')
        if info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset:
            raise ValueError('Binned/cropped CameraInfo requires adjusted intrinsics')
        normalized_ray(0.0, 0.0, info.k, info.d, info.distortion_model)
        bgr = self.bridge.imgmsg_to_cv2(color, desired_encoding='bgr8')
        depth_m = depth_in_meters(self.bridge.imgmsg_to_cv2(depth), depth.encoding)
        c = self.config
        inference_started = time.perf_counter()
        result = self.model.predict(
            source=bgr, conf=c['confidence'], iou=c['iou'], imgsz=c['image_size'],
            device=c['device'], max_det=c['max_detections'], verbose=False,
            classes=self.class_ids or None,
        )[0]
        inference_sec = time.perf_counter() - inference_started
        output = DetectedObjectArray(header=color.header)
        canvas = bgr.copy() if c['publish_annotated_image'] else None
        # 한 번에 CPU로 이동하여 detection마다 GPU 동기화하지 않는다.
        boxes = result.boxes.cpu().numpy()
        for bbox, score, class_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
            bbox = np.asarray(bbox, dtype=float).copy()
            if not np.all(np.isfinite(bbox)) or not np.isfinite(score):
                raise ValueError('Model returned nonfinite detection')
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, color.width)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, color.height)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            estimate = estimate_position(
                depth_m, bbox, info.k, info.d, info.distortion_model,
                c['depth_roi_fraction'], c['min_depth_m'], c['max_depth_m'],
                c['min_depth_samples'],
            )
            detection = DetectedObject(
                class_id=int(class_id), class_name=str(result.names[int(class_id)]),
                confidence=float(score), bbox_xyxy=bbox.tolist(),
                depth_valid=estimate.valid, valid_depth_samples=estimate.samples,
                depth_m=estimate.z, range_m=estimate.range_m,
            )
            detection.position.x, detection.position.y, detection.position.z = (
                estimate.x, estimate.y, estimate.z,
            )
            output.objects.append(detection)
            if canvas is not None:
                x1, y1, x2, y2 = np.rint(bbox).astype(int)
                label = f'{detection.class_name} {score:.2f}'
                label += f' Z={estimate.z:.2f}m' if estimate.valid else ' depth=N/A'
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(canvas, label, (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # 긴 추론 후에도 촬영 시각 기준 최대 age를 만족하는 결과만 발행한다.
        if not self.context.ok():
            return
        process_sec = time.perf_counter() - process_started
        output_age = self._frame_age_sec(color)
        if not self._fresh(color):
            self.get_logger().warning(
                'Inference exceeded max_frame_age_sec; result discarded: '
                f'input_age={input_age:.3f}s, inference={inference_sec:.3f}s, '
                f'processing={process_sec:.3f}s, output_age={output_age:.3f}s, '
                f'limit={c["max_frame_age_sec"]:.3f}s', throttle_duration_sec=5.0,
            )
            return
        self.result_pub.publish(output)
        if canvas is not None:
            annotated = self.bridge.cv2_to_imgmsg(canvas, encoding='bgr8')
            annotated.header = color.header
            self.image_pub.publish(annotated)
        self.get_logger().info(
            f'RGB-D timing: input_age={input_age:.3f}s, inference={inference_sec:.3f}s, '
            f'processing={process_sec:.3f}s, output_age={output_age:.3f}s, '
            f'objects={len(output.objects)}', throttle_duration_sec=5.0,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        node = DetectorNode()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown()
            if node is not None:
                node.destroy_node()
        except KeyboardInterrupt:
            # 프로세스 그룹 SIGINT와 launch의 전달 SIGINT가 연이어 도착할 수 있다.
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()
