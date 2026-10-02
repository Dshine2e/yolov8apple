#!/usr/bin/env python3
"""Replay an actual test RGB image through the selected detector with invalid depth."""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from asac_interfaces.msg import DetectedObjectArray

ASAC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ASAC_ROOT / 'src/asac_perception'))
from asac_perception.model_selection import detector_selection  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--min-detections', type=int, default=1)
    options = parser.parse_args()
    model, _ = detector_selection(ASAC_ROOT)
    image = options.image or sorted(
        (ASAC_ROOT / 'datasets/apple-v2/test/images').glob('*.jpg'))[0]
    bgr = cv2.imread(str(image))
    scale = 960 / max(bgr.shape[:2])
    bgr = cv2.resize(bgr, (round(bgr.shape[1] * scale), round(bgr.shape[0] * scale)))
    bridge = CvBridge()
    height, width = bgr.shape[:2]
    process = subprocess.Popen([
        'bash', str(ASAC_ROOT / 'scripts/run_detector.sh'),
        'start_camera:=false', 'image_size:=640', 'cpu_threads:=2',
        'max_frame_age_sec:=5.0',
    ], start_new_session=True)
    rclpy.init()
    peer = Node('imported_detector_replay')
    results, overlays = [], []
    peer.create_subscription(DetectedObjectArray, '/asac/detections', results.append, 1)
    peer.create_subscription(Image, '/asac/annotated_image', overlays.append, 1)
    publishers = [peer.create_publisher(kind, topic, 1) for kind, topic in [
        (Image, '/camera/camera/color/image_raw'),
        (Image, '/camera/camera/aligned_depth_to_color/image_raw'),
        (CameraInfo, '/camera/camera/color/camera_info'),
    ]]
    try:
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline and (not results or not overlays):
            if process.poll() is not None:
                raise RuntimeError(f'Detector launch exited: {process.returncode}')
            if all(publisher.get_subscription_count() for publisher in publishers):
                color = bridge.cv2_to_imgmsg(bgr, 'bgr8')
                depth = bridge.cv2_to_imgmsg(np.zeros((height, width), np.uint16), '16UC1')
                color.header.frame_id = 'test_rgb_optical_frame'
                color.header.stamp = peer.get_clock().now().to_msg()
                depth.header = color.header
                info = CameraInfo(header=color.header, height=height, width=width)
                info.k = [640., 0., width / 2, 0., 640., height / 2, 0., 0., 1.]
                info.distortion_model = 'plumb_bob'
                for publisher, message in zip(publishers, [color, depth, info]):
                    publisher.publish(message)
            rclpy.spin_once(peer, timeout_sec=0.1)
        if not results or not overlays or len(results[0].objects) < options.min_detections:
            raise RuntimeError('No real YOLO detections and overlay reached ROS subscribers')
        result = results[0]
        assert all(not obj.depth_valid and all(math.isnan(value) for value in (
            obj.position.x, obj.position.y, obj.position.z, obj.depth_m, obj.range_m,
        )) for obj in result.objects)
        assert result.header == overlays[0].header
        report = {
            'status': 'passed', 'model': str(model), 'actual_rgb_source': str(image),
            'image_size': [width, height], 'ros_domain_id': os.environ.get('ROS_DOMAIN_ID'),
            'topics': ['/asac/detections', '/asac/annotated_image'],
            'expected_min_detections': options.min_detections,
            'frame': result.header.frame_id,
            'stamp': {'sec': result.header.stamp.sec, 'nanosec': result.header.stamp.nanosec},
            'detections': [{'class_name': obj.class_name, 'confidence': float(obj.confidence),
                            'bbox_xyxy': [float(value) for value in obj.bbox_xyxy],
                            'depth_valid': obj.depth_valid,
                            'camera_position_m': None} for obj in result.objects],
            'depth_input': 'all-zero synthetic input; no metric geometry tested',
            'intrinsics': 'synthetic test metadata; physical camera calibration not tested',
            'physical_camera_started': False,
        }
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        cv2.imwrite(str(options.output.with_suffix('.jpg')),
                    bridge.imgmsg_to_cv2(overlays[0], 'bgr8'))
        print(json.dumps(report, indent=2))
    finally:
        peer.destroy_node()
        rclpy.shutdown()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)


if __name__ == '__main__':
    main()
