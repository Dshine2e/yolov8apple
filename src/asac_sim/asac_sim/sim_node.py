import json
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock as RclClock, ClockType
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from asac_sim.scene import Scene


class SimNode(Node):
    def __init__(self):
        super().__init__("scene")
        share = Path(get_package_share_directory("asac_sim"))
        path = self.declare_parameter("scene_file", str(share / "config/scene.yaml")).value
        self.cfg = yaml.safe_load(Path(path).read_text())
        self.scene = Scene(self.cfg, share / "assets/piper")
        self.bridge, self.rng = CvBridge(), np.random.default_rng(10)
        self.color_pub = self.create_publisher(
            Image, "/asac_sim/rgb/image", qos_profile_sensor_data
        )
        self.depth_pub = self.create_publisher(
            Image, "/asac_sim/depth/image", qos_profile_sensor_data
        )
        self.info_pub = self.create_publisher(
            CameraInfo, "/asac_sim/rgb/camera_info", qos_profile_sensor_data
        )
        self.clock_pub = self.create_publisher(Clock, "/clock", 1)
        self.joint_pub = self.create_publisher(JointState, "/asac_sim/joint_states", 1)
        self.truth_pub = self.create_publisher(String, "/asac_sim/evaluation_truth", 1)
        self.marker_pub = self.create_publisher(MarkerArray, "/asac_sim/scene_markers", 1)
        self.broadcaster = StaticTransformBroadcaster(self)
        self.start = time.monotonic()
        self.frame_index = 0
        self._static_transforms()
        # Wall timer drives /clock; all payloads use the same advancing sim clock.
        self.create_timer(
            1 / self.cfg["rate_hz"], self._tick, clock=RclClock(clock_type=ClockType.STEADY_TIME)
        )
        self.get_logger().info(f"TinyRenderer RGB-D ready; PiPER root={self.scene.base_frame}")

    def _static_transforms(self):
        camera = self.cfg["camera"]
        transforms = []

        def add(parent, child, xyz, quat):
            t = TransformStamped()
            t.header.frame_id, t.child_frame_id = parent, child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(
                float, xyz
            )
            q = t.transform.rotation
            q.x, q.y, q.z, q.w = map(float, quat)
            transforms.append(t)

        world = self.cfg["world_frame"]
        add(world, self.scene.base_frame, self.cfg["robot_position"], [0, 0, 0, 1])
        if not self.cfg.get("faults", {}).get("disable_camera_tf", False):
            # REP103: body +X forward/+Y left/+Z up, optical +Z forward/+X right/+Y down.
            body_from_optical = Rotation.from_euler("xyz", [-np.pi / 2, 0, -np.pi / 2])
            world_from_body = self.scene.rotation * body_from_optical.inv()
            add(world, camera["link_frame"], camera["position"], world_from_body.as_quat())
            add(
                camera["link_frame"],
                camera["optical_frame"],
                [0, 0, 0],
                body_from_optical.as_quat(),
            )
        self.broadcaster.sendTransform(transforms)

    def _tick(self):
        from builtin_interfaces.msg import Time

        elapsed = 1.0 + time.monotonic() - self.start
        ns = int(elapsed * 1e9)
        stamp = Time(sec=ns // 10**9, nanosec=ns % 10**9)
        self.clock_pub.publish(Clock(clock=stamp))
        self.scene.apply_events(elapsed)
        rgb, depth = self.scene.render()
        self.frame_index += 1
        faults = self.cfg.get("faults", {})
        depth[self.rng.random(depth.shape) < faults.get("depth_hole_fraction", 0)] = np.nan
        depth *= faults.get("depth_unit_multiplier", 1.0)
        frame = self.cfg["camera"]["optical_frame"]
        switch = faults.get("change_frame_after_sec", -1)
        if switch >= 0 and elapsed >= switch:
            frame += "_changed"
        delay_ns = int(faults.get("stamp_delay_sec", 0) * 1e9)
        capture_ns = max(1, ns - delay_ns)
        capture_stamp = Time(sec=capture_ns // 10**9, nanosec=capture_ns % 10**9)
        color = self.bridge.cv2_to_imgmsg(rgb, encoding="rgb8")
        color.header.stamp, color.header.frame_id = capture_stamp, frame
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header = color.header
        info = CameraInfo(
            header=color.header,
            width=self.scene.width,
            height=self.scene.height,
            distortion_model="plumb_bob",
            d=[0.0] * 5,
            k=self.scene.k,
            r=np.eye(3).ravel().tolist(),
        )
        k = np.asarray(self.scene.k).reshape(3, 3)
        info.p = np.column_stack((k, np.zeros(3))).ravel().tolist()
        if faults.get("uncalibrated_info", False):
            info.k, info.p = [0.0] * 9, [0.0] * 12
        drop = faults.get("drop_rgb_every", 0)
        if not drop or self.frame_index % drop:
            self.color_pub.publish(color)
        self.depth_pub.publish(depth_msg)
        self.info_pub.publish(info)
        joint = JointState()
        joint.header.stamp = stamp
        joint.name = [name for name, _ in self.scene.joints]
        joint.position = [position for _, position in self.scene.joints]
        self.joint_pub.publish(joint)
        truth = {
            "stamp": {"sec": capture_stamp.sec, "nanosec": capture_stamp.nanosec},
            "base_frame": self.scene.base_frame,
            "apples": self.scene.truth(),
        }
        self.truth_pub.publish(String(data=json.dumps(truth, allow_nan=False)))
        self._scene_markers(stamp)

    def _scene_markers(self, stamp):
        array = MarkerArray()
        for i, obj in enumerate(self.scene.objects):
            m = Marker(type=Marker.CUBE, action=Marker.ADD, ns="scene", id=i)
            m.header.stamp, m.header.frame_id = stamp, self.cfg["world_frame"]
            m.pose.orientation.w = 1.0
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, obj["position"])
            m.scale.x, m.scale.y, m.scale.z = map(float, obj["size"])
            m.color.r, m.color.g, m.color.b, m.color.a = map(float, obj["color"])
            array.markers.append(m)
        # Camera visual, distinct from measured apples; no truth apple markers.
        m = Marker(type=Marker.CUBE, action=Marker.ADD, ns="camera", id=0)
        m.header.stamp, m.header.frame_id = stamp, self.cfg["camera"]["link_frame"]
        m.pose.orientation.w = 1.0
        m.scale.x, m.scale.y, m.scale.z = 0.04, 0.09, 0.03
        m.color.b, m.color.a = 1.0, 1.0
        array.markers.append(m)
        self.marker_pub.publish(array)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SimNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.scene.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
