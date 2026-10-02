from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml

from asac_sim.scene import robot_description


def start(context):
    def get(name):
        return LaunchConfiguration(name).perform(context)

    share = Path(get_package_share_directory("asac_sim"))
    cfg = yaml.safe_load(Path(get("scene_file")).read_text())
    description, root = robot_description(share / "assets/piper", for_rviz=True)
    camera = cfg["camera"]
    actions = [
        Node(
            package="asac_sim",
            executable="scene",
            namespace="asac_sim",
            output="screen",
            parameters=[{"scene_file": get("scene_file"), "use_sim_time": True}],
        ),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace="asac_sim",
            output="screen",
            parameters=[{"robot_description": description, "use_sim_time": True}],
            remappings=[("joint_states", "/asac_sim/joint_states")],
        ),
        Node(
            package="asac_perception",
            executable="topview",
            name="topview",
            namespace="asac",
            output="screen",
            parameters=[
                get("params_file"),
                {
                    "model_path": get("model_path"),
                    "input_mode": "sim",
                    "rgb_only": False,
                    "use_sim_time": True,
                    "camera_id": camera["camera_id"],
                    "color_topic": "/asac_sim/rgb/image",
                    "depth_topic": "/asac_sim/depth/image",
                    "camera_info_topic": "/asac_sim/rgb/camera_info",
                    "expected_optical_frame": camera["optical_frame"],
                    "base_frame": root,
                    "calibration_verified": True,
                    "image_state": "rectified",
                    "depth_aligned_to_rgb": True,
                    "depth_is_optical_z": True,
                },
            ],
        ),
    ]
    if get("rviz").lower() == "true":
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(share / "config/topview.rviz")],
                parameters=[{"use_sim_time": True}],
                output="screen",
            )
        )
    if get("evaluate").lower() == "true":
        actions.append(
            Node(
                package="asac_sim",
                executable="evaluate",
                namespace="asac_sim",
                parameters=[
                    {
                        "use_sim_time": True,
                        "duration_sec": float(get("evaluation_duration")),
                        "output_file": get("evaluation_output"),
                    }
                ],
                output="screen",
            )
        )
    return actions


def generate_launch_description():
    share = Path(get_package_share_directory("asac_sim"))
    perception = Path(get_package_share_directory("asac_perception"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path"),
            DeclareLaunchArgument("scene_file", default_value=str(share / "config/scene.yaml")),
            DeclareLaunchArgument(
                "params_file", default_value=str(perception / "config/topview_real.yaml")
            ),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("evaluate", default_value="false"),
            DeclareLaunchArgument("evaluation_duration", default_value="30.0"),
            DeclareLaunchArgument("evaluation_output", default_value="/tmp/asac-evaluation.json"),
            OpaqueFunction(function=start),
        ]
    )
