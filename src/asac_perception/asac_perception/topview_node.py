"""Fixed-camera recognition only. No quality decisions, planner or hardware commands."""

from dataclasses import asdict
import json
import math
from pathlib import Path
import threading
import time

import cv2
from cv_bridge import CvBridge
import message_filters
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from asac_interfaces.msg import AppleObservation, AppleObservationArray
from asac_perception.geometry import GeometryConfig, depth_meters
from asac_perception.observations import measure_instances, metadata
from asac_perception.segmentation import Segmenter
from asac_perception.snapshots import SnapshotWriter
from asac_perception.tracking import Tracker


DEFAULTS = {
    "model_path": "",
    "target_classes": "apple",
    "device": "cpu",
    "image_size": 640,
    "cpu_threads": 2,
    "confidence": 0.35,
    "iou": 0.45,
    "max_detections": 30,
    "input_mode": "real",
    "rgb_only": False,
    "camera_id": "top_camera",
    "color_topic": "/camera/camera/color/image_raw",
    "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
    "camera_info_topic": "/camera/camera/color/camera_info",
    "expected_optical_frame": "",
    "base_frame": "",
    "calibration_verified": False,
    "depth_aligned_to_rgb": True,
    "depth_is_optical_z": True,
    "depth_scale_16uc1": 0.001,
    "image_state": "raw",
    "qos_reliability": "best_effort",
    "qos_depth": 1,
    "sync_slop_sec": 0.03,
    "sync_queue_size": 5,
    "tf_timeout_sec": 0.08,
    "max_frame_age_sec": 2.0,
    "inference_rate_hz": 5.0,
    "association_max_distance_m": 0.06,
    "association_min_iou": 0.1,
    "track_ttl_sec": 2.0,
    "duplicate_iou": 0.85,
    "save_directory": "",
    "save_interval_sec": 10.0,
    "save_max_batches": 50,
}


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encode(value):
    return json.dumps(json_safe(value), allow_nan=False, separators=(",", ":"))


class TopviewNode(Node):
    def __init__(self, segmenter_factory=Segmenter, **kwargs):
        super().__init__("topview", **kwargs)
        defaults = {**DEFAULTS, **asdict(GeometryConfig())}
        self.config = {
            k: self.declare_parameter(k, v, ParameterDescriptor(read_only=True)).value
            for k, v in defaults.items()
        }
        self._validate()
        c = self.config
        self.geometry = GeometryConfig(**{k: c[k] for k in asdict(GeometryConfig())})
        self.segmenter = segmenter_factory(
            str(Path(c["model_path"]).expanduser()),
            c["target_classes"],
            c["device"],
            c["image_size"],
            c["cpu_threads"],
        )
        self.bridge = CvBridge()
        self.tracker = Tracker(
            c["association_max_distance_m"],
            c["association_min_iou"],
            c["track_ttl_sec"],
            c["duplicate_iou"],
        )
        self.writer = SnapshotWriter(
            c["save_directory"], c["save_interval_sec"], c["save_max_batches"]
        )
        self.observations_pub = self.create_publisher(AppleObservationArray, "observations", 1)
        self.json_pub = self.create_publisher(String, "observations_json", 1)
        self.planning_pub = self.create_publisher(String, "planning_input", 1)
        self.image_pub = self.create_publisher(Image, "overlay", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "markers", 1)
        self.status_pub = self.create_publisher(String, "status", 1)
        self.create_service(Trigger, "snapshot", self._snapshot)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.lock, self.latest = threading.Lock(), None
        self.last_receive = time.monotonic()
        qos = QoSProfile(
            depth=c["qos_depth"],
            reliability=(
                ReliabilityPolicy.BEST_EFFORT
                if c["qos_reliability"] == "best_effort"
                else ReliabilityPolicy.RELIABLE
            ),
        )
        self.rgb_info = None
        if c["rgb_only"]:
            # 2D measurements require neither depth nor calibration; CameraInfo is optional.
            self.subscribers = [
                self.create_subscription(Image, c["color_topic"], self._receive_rgb, qos),
                self.create_subscription(
                    CameraInfo, c["camera_info_topic"], self._receive_info, qos
                ),
            ]
        else:
            inputs = [
                (Image, "color_topic"),
                (CameraInfo, "camera_info_topic"),
                (Image, "depth_topic"),
            ]
            self.subscribers = [
                message_filters.Subscriber(self, cls, c[key], qos_profile=qos)
                for cls, key in inputs
            ]
            self.sync = message_filters.ApproximateTimeSynchronizer(
                self.subscribers, c["sync_queue_size"], c["sync_slop_sec"], allow_headerless=False
            )
            self.sync.registerCallback(self._receive)
        self.inference_group = MutuallyExclusiveCallbackGroup()
        self.create_timer(
            1.0 / c["inference_rate_hz"], self._infer, callback_group=self.inference_group
        )
        self.create_timer(5.0, self._watchdog, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f"Topview segmenter ready; names={self.segmenter.model.names}; "
            f"mode={c['input_mode']}, base={c['base_frame']}, "
            f"calibration_verified={c['calibration_verified']}"
        )

    def _validate(self):
        c = self.config
        if c["input_mode"] not in ("sim", "real") or c["image_state"] not in ("raw", "rectified"):
            raise ValueError("input_mode must be sim/real and image_state raw/rectified")
        if c["qos_reliability"] not in ("best_effort", "reliable"):
            raise ValueError("Unsupported QoS reliability")
        for k in (
            "confidence",
            "iou",
            "association_min_iou",
            "duplicate_iou",
            "min_depth_ratio",
            "sphere_min_inlier_ratio",
            "sphere_min_cap_fraction",
            "min_solidity",
        ):
            if not np.isfinite(c[k]) or not 0 < c[k] <= 1:
                raise ValueError(f"{k} must be in (0,1]")
        for k in (
            "inference_rate_hz",
            "sync_slop_sec",
            "tf_timeout_sec",
            "max_frame_age_sec",
            "association_max_distance_m",
            "track_ttl_sec",
            "depth_scale_16uc1",
            "save_interval_sec",
            "sphere_inlier_threshold_m",
            "sphere_max_rmse_m",
            "sphere_max_condition",
            "depth_mad_factor",
            "depth_mad_floor_m",
        ):
            if not np.isfinite(c[k]) or c[k] <= 0:
                raise ValueError(f"{k} must be finite and positive")
        for k in (
            "image_size",
            "cpu_threads",
            "qos_depth",
            "sync_queue_size",
            "max_detections",
            "min_depth_points",
            "max_fit_points",
            "sphere_trials",
            "min_mask_pixels",
            "save_max_batches",
        ):
            if c[k] < 1:
                raise ValueError(f"{k} must be positive")
        if c["min_depth_points"] < 4 or c["max_fit_points"] < c["min_depth_points"]:
            raise ValueError("Sphere fit needs >=4 points and sufficient max_fit_points")
        if c["erosion_pixels"] < 0 or c["hole_max_pixels"] < 0:
            raise ValueError("Negative morphology setting")
        for lo, hi in [
            ("min_depth_m", "max_depth_m"),
            ("sphere_min_radius_m", "sphere_max_radius_m"),
        ]:
            if not 0 < c[lo] < c[hi] or not np.isfinite(c[hi]):
                raise ValueError(f"Invalid {lo}/{hi}")
        if not c["camera_id"] or not c["color_topic"] or not c["camera_info_topic"]:
            raise ValueError("camera_id and input topics are required")
        if c["calibration_verified"] and (not c["base_frame"] or not c["expected_optical_frame"]):
            raise ValueError("Verified calibration needs explicit base/optical frames")

    def _snapshot(self, request, response):
        response.success = (
            self.writer.directory is not None and self.writer.count < self.writer.limit
        )
        response.message = (
            "Next fresh capture requested"
            if response.success
            else "Saving disabled or quota reached"
        )
        if response.success:
            self.writer.requested = True
        return response

    def _receive(self, color, info, depth=None):
        with self.lock:
            self.latest = (color, info, depth)
            self.last_receive = time.monotonic()

    def _receive_info(self, info):
        with self.lock:
            self.rgb_info = info

    def _receive_rgb(self, color):
        with self.lock:
            info = self.rgb_info
        if info is None:
            info = CameraInfo(header=color.header, width=color.width, height=color.height)
        self._receive(color, info)

    def _status(self, reason, color=None):
        if not self.context.ok():
            return
        payload = {"state": reason, "camera_id": self.config["camera_id"]}
        if color:
            payload.update(
                stamp={"sec": color.header.stamp.sec, "nanosec": color.header.stamp.nanosec},
                camera_frame=color.header.frame_id,
            )
        self.status_pub.publish(String(data=encode(payload)))
        if reason != "ok":
            self.get_logger().warning(reason, throttle_duration_sec=5.0)

    def _watchdog(self):
        if time.monotonic() - self.last_receive > 5:
            self._status("no_synchronized_input_check_topics_qos_stamps")

    def _fresh(self, color):
        age = (self.get_clock().now() - Time.from_msg(color.header.stamp)).nanoseconds / 1e9
        return -self.config["sync_slop_sec"] <= age <= self.config["max_frame_age_sec"]

    def _infer(self):
        if not self.context.ok():
            return
        with self.lock:
            frame, self.latest = self.latest, None
        if frame is None:
            return
        color, info, depth = frame
        if not self._fresh(color):
            self._status("stale_or_future_input", color)
            return
        started = time.perf_counter()
        try:
            bgr = self.bridge.imgmsg_to_cv2(color, desired_encoding="bgr8")
            if bgr.shape[:2] != (color.height, color.width):
                raise ValueError("rgb_shape_mismatch")
            instances = self.segmenter.predict(
                bgr, self.config["confidence"], self.config["iou"], self.config["max_detections"]
            )
            depth_array, intrinsics, depth_reason = self._input_geometry(color, info, depth)
            transform, tf_reason = self._lookup(color)
            observations = measure_instances(
                bgr,
                instances,
                depth_array,
                intrinsics,
                self.geometry,
                transform,
                depth_reason,
                tf_reason,
            )
            # Inference/TF delay is checked before association; stale frames never change tracks.
            if not self.context.ok():
                return
            if not self._fresh(color):
                self._status("stale_result_discarded", color)
                return
            stamp = color.header.stamp.sec + color.header.stamp.nanosec / 1e9
            observations = self.tracker.update(observations, stamp, color.header.frame_id)
            header = self._metadata_header(color, info, depth)
            header["processing_sec"] = time.perf_counter() - started
            for o in observations:
                o.update(
                    camera_id=self.config["camera_id"],
                    camera_frame=color.header.frame_id,
                    base_frame=self.config["base_frame"],
                    stamp=header["stamp"],
                    camera_metadata=header["camera_metadata"],
                    measurement_config=header["measurement_config"],
                )
            if not self._publish(color, bgr, observations, header):
                return
            self.writer.write(observations, header, stamp)
            self._status("ok", color)
            self.get_logger().info(
                f"apples={len(observations)} processing={header['processing_sec']:.3f}s "
                f"base_valid={sum(o['center_base_valid'] for o in observations)}",
                throttle_duration_sec=5.0,
            )
        except Exception as error:
            # Preserve error visibility; no empty success result or simulated masks on failure.
            self._status(f"processing_error:{type(error).__name__}:{error}", color)

    def _input_geometry(self, color, info, depth):
        c = self.config
        intrinsics = (list(info.k), list(info.d), info.distortion_model)
        if c["rgb_only"]:
            return None, intrinsics, "rgb_only_no_depth"
        reason = ""
        if not c["expected_optical_frame"] or color.header.frame_id != c["expected_optical_frame"]:
            reason = "unexpected_or_unspecified_optical_frame"
        elif not (color.header.frame_id == info.header.frame_id == depth.header.frame_id):
            reason = "unaligned_input_frames"
        elif (color.width, color.height) != (depth.width, depth.height) or (
            color.width,
            color.height,
        ) != (info.width, info.height):
            reason = "unaligned_input_resolution"
        elif (
            info.binning_x > 1
            or info.binning_y > 1
            or info.roi.x_offset
            or info.roi.y_offset
            or info.roi.width not in (0, info.width)
            or info.roi.height not in (0, info.height)
        ):
            reason = "unadjusted_binning_or_roi"
        elif not c["depth_aligned_to_rgb"] or not c["depth_is_optical_z"]:
            reason = "unsupported_depth_contract_require_aligned_optical_z"
        elif not np.isfinite(info.k).all() or info.k[0] <= 0 or info.k[4] <= 0:
            reason = "uncalibrated_intrinsics"
        if c["image_state"] == "rectified":
            p = np.asarray(info.p).reshape(3, 4)
            if not np.isfinite(p).all() or p[0, 0] <= 0 or p[1, 1] <= 0 or np.any(p[:, 3]):
                reason = "invalid_rectified_projection_matrix"
            intrinsics = (p[:, :3].ravel().tolist(), [], "")
        if reason:
            return None, intrinsics, reason
        try:
            image = self.bridge.imgmsg_to_cv2(depth, desired_encoding="passthrough")
            return depth_meters(image, depth.encoding, c["depth_scale_16uc1"]), intrinsics, ""
        except ValueError as error:
            return None, intrinsics, str(error)

    def _lookup(self, color):
        c = self.config
        if not c["calibration_verified"]:
            return None, "eye_to_hand_uncalibrated"
        if color.header.frame_id != c["expected_optical_frame"]:
            return None, "unexpected_optical_frame"
        if color.header.stamp.sec == 0 and color.header.stamp.nanosec == 0:
            return None, "zero_stamp_cannot_lookup_capture_tf"
        try:
            tf = self.tf_buffer.lookup_transform(
                c["base_frame"],
                color.header.frame_id,
                Time.from_msg(color.header.stamp),
                timeout=Duration(seconds=c["tf_timeout_sec"]),
            )
            t, q = tf.transform.translation, tf.transform.rotation
            return ([t.x, t.y, t.z], [q.x, q.y, q.z, q.w]), ""
        except TransformException as error:
            return None, f"tf_capture_lookup_failed:{type(error).__name__}:{error}"

    def _metadata_header(self, color, info, depth):
        c = self.config
        return {
            "stamp": {"sec": color.header.stamp.sec, "nanosec": color.header.stamp.nanosec},
            "camera_frame": color.header.frame_id,
            "base_frame": c["base_frame"],
            "camera_id": c["camera_id"],
            "session_id": self.tracker.session,
            "input_mode": c["input_mode"],
            "view_type": "top",
            "measurement_config": asdict(self.geometry),
            "camera_metadata": json_safe(
                {
                    "k": list(info.k),
                    "p": list(info.p),
                    "d": list(info.d),
                    "distortion_model": info.distortion_model,
                    "width": info.width,
                    "height": info.height,
                    "image_state": c["image_state"],
                    "projection_source": None
                    if c["rgb_only"]
                    else "P"
                    if c["image_state"] == "rectified"
                    else "K_with_inverse_distortion",
                    "depth_encoding": depth.encoding if depth else None,
                    "depth_scale_16uc1": c["depth_scale_16uc1"],
                    "depth_quantity": "optical_axis_z_m" if depth else None,
                    "calibration_verified": c["calibration_verified"],
                    "depth_stamp": {
                        "sec": depth.header.stamp.sec,
                        "nanosec": depth.header.stamp.nanosec,
                    }
                    if depth
                    else None,
                    "info_stamp": {
                        "sec": info.header.stamp.sec,
                        "nanosec": info.header.stamp.nanosec,
                    },
                }
            ),
        }

    def _point(self, destination, values, header, frame):
        destination.header.stamp, destination.header.frame_id = header.stamp, frame
        destination.point.x, destination.point.y, destination.point.z = values or [math.nan] * 3

    def _publish(self, color, bgr, observations, header):
        c = self.config
        array = AppleObservationArray(
            header=color.header,
            camera_id=c["camera_id"],
            base_frame=c["base_frame"],
            session_id=self.tracker.session,
            input_mode=c["input_mode"],
        )
        canvas = bgr.copy()
        legend = []
        markers = MarkerArray()
        markers.markers.append(Marker(action=Marker.DELETEALL))
        for i, o in enumerate(observations):
            msg = AppleObservation(
                header=color.header,
                apple_id=o["apple_id"],
                camera_id=c["camera_id"],
                base_frame=c["base_frame"],
                view_type="top",
                confidence=o["confidence"],
                class_id=o["class_id"],
                bbox_xyxy=o["bbox"],
                image_width=color.width,
                image_height=color.height,
                metadata_json=encode(metadata(o)),
            )
            msg.roi.x_offset, msg.roi.y_offset, msg.roi.width, msg.roi.height = o["roi"]
            msg.rgb_crop = self.bridge.cv2_to_imgmsg(o["rgb_crop"], encoding="bgr8")
            msg.mask_crop = self.bridge.cv2_to_imgmsg(o["mask_crop"], encoding="mono8")
            msg.rgb_crop.header = msg.mask_crop.header = color.header
            for name in ("center_camera", "center_base", "surface_camera", "surface_base"):
                self._point(
                    getattr(msg, name),
                    o[name],
                    color.header,
                    c["base_frame"] if name.endswith("base") else color.header.frame_id,
                )
            for name in (
                "center_camera_valid",
                "center_base_valid",
                "surface_valid",
                "surface_base_valid",
                "radius_valid",
                "circularity_valid",
                "direction_valid",
            ):
                setattr(msg, name, o[name])
            for name in (
                "radius_m",
                "diameter_m",
                "radius_px",
                "circularity_top",
                "valid_depth_ratio",
                "fit_rmse_m",
                "fit_condition",
                "inlier_ratio",
                "cap_fraction",
            ):
                setattr(msg, name, float(o[name]) if o[name] is not None else math.nan)
            msg.valid_depth_points = o["valid_depth_points"]
            msg.measurement_method, msg.tracking_method = o["method"], o["tracking_method"]
            msg.occlusion_state, msg.reasons = o["occlusion_state"], o["reasons"]
            msg.direction_base.header.stamp, msg.direction_base.header.frame_id = (
                color.header.stamp,
                c["base_frame"],
            )
            (
                msg.direction_base.vector.x,
                msg.direction_base.vector.y,
                msg.direction_base.vector.z,
            ) = o["direction_base"] or [math.nan] * 3
            msg.camera_pose_base.header = msg.direction_base.header
            msg.camera_pose_valid = o["camera_pose_base"] is not None
            pose = o["camera_pose_base"]
            (
                msg.camera_pose_base.pose.position.x,
                msg.camera_pose_base.pose.position.y,
                msg.camera_pose_base.pose.position.z,
            ) = pose["position"] if pose else [math.nan] * 3
            q = msg.camera_pose_base.pose.orientation
            q.x, q.y, q.z, q.w = pose["quaternion_xyzw"] if pose else [math.nan] * 4
            array.apples.append(msg)
            tint = np.array([20, 190, 40], np.uint8)
            canvas[o["full_mask"]] = (0.65 * canvas[o["full_mask"]] + 0.35 * tint).astype(np.uint8)
            x, y = o["roi"][:2]
            radius = f"{o['radius_m']:.3f}m" if o["radius_valid"] else "N/A"
            circ = f"{o['circularity_top']:.3f}" if o["circularity_valid"] else "uncertain"
            counter = o["apple_id"].split("-")[-1]
            label = f"{counter} {o['confidence']:.2f}"
            for thickness, ink in [(3, (0, 0, 0)), (1, (0, 255, 255))]:
                cv2.putText(
                    canvas,
                    label,
                    (x, max(16, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    ink,
                    thickness,
                )
            state = "base OK" if o["center_base_valid"] else "|".join(o["reasons"])[:45]
            legend.append(f"ID={counter} conf={o['confidence']:.2f} r={radius} Ctop={circ} {state}")
            if o["center_base_valid"]:
                markers.markers.extend(self._markers(i, o, color.header))
        rows = min(len(legend), max(0, color.height // 60))
        if rows:
            top = color.height - rows * 20 - 6
            cv2.rectangle(canvas, (0, top), (color.width, color.height), (30, 30, 30), -1)
            for row, label in enumerate(legend[:rows]):
                cv2.putText(
                    canvas,
                    label,
                    (6, top + 17 + row * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (230, 230, 230),
                    1,
                )
        if not self.context.ok():
            return False
        if not self._fresh(color):
            self._status("stale_result_discarded", color)
            return False
        header["output_age_sec"] = (
            self.get_clock().now() - Time.from_msg(color.header.stamp)
        ).nanoseconds / 1e9
        self.observations_pub.publish(array)
        payload = dict(header, observations=[metadata(o) for o in observations])
        self.json_pub.publish(String(data=encode(payload)))
        planning = dict(
            header,
            apples=[
                {
                    k: o[k]
                    for k in (
                        "apple_id",
                        "center_base",
                        "center_base_valid",
                        "radius_m",
                        "diameter_m",
                        "radius_valid",
                        "occlusion_state",
                        "reasons",
                    )
                }
                for o in observations
            ],
        )
        self.planning_pub.publish(String(data=encode(planning)))
        overlay = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
        overlay.header = color.header
        self.image_pub.publish(overlay)
        self.marker_pub.publish(markers)
        return True

    def _markers(self, index, o, header):
        markers = []
        for j, kind in enumerate((Marker.SPHERE, Marker.TEXT_VIEW_FACING, Marker.ARROW)):
            m = Marker(type=kind, action=Marker.ADD, ns="apples", id=index * 3 + j)
            m.header.stamp, m.header.frame_id = header.stamp, self.config["base_frame"]
            m.pose.orientation.w = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 1.0, 0.2, 0.65
            m.lifetime = Duration(seconds=self.config["max_frame_age_sec"]).to_msg()
            m.pose.position.x, m.pose.position.y, m.pose.position.z = o["center_base"]
            if kind == Marker.SPHERE:
                m.scale.x = m.scale.y = m.scale.z = o["diameter_m"]
            elif kind == Marker.TEXT_VIEW_FACING:
                m.scale.z = 0.02
                m.pose.position.z += o["radius_m"] + 0.025
                m.text = o["apple_id"]
            else:
                from geometry_msgs.msg import Point

                start = np.asarray(o["center_base"])
                end = start + np.asarray(o["direction_base"]) * 0.12
                m.pose.position.x = m.pose.position.y = m.pose.position.z = 0.0
                m.points = [
                    Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in (start, end)
                ]
                m.scale.x, m.scale.y, m.scale.z = 0.005, 0.012, 0.02
            markers.append(m)
        return markers


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=3)
    try:
        node = TopviewNode()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown()
            if node:
                node.destroy_node()
        except KeyboardInterrupt:
            # Terminal SIGINT and launch's forwarded SIGINT can arrive consecutively.
            pass
        finally:
            if rclpy.ok():
                rclpy.shutdown()
