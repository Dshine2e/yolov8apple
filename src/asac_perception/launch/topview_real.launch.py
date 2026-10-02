from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    def get(name):
        return LaunchConfiguration(name).perform(context)

    params = {"model_path": get("model_path"), "input_mode": "real", "use_sim_time": False}
    if get("rgb_only"):
        params["rgb_only"] = get("rgb_only").lower() == "true"
    actions = []
    # Lazy resolution preserves real RGB-only/external-driver mode without RealSense installed.
    if get("start_camera").lower() == "true":
        share = get_package_share_directory("realsense2_camera")
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(Path(share) / "launch/rs_launch.py")),
                launch_arguments={
                    "enable_color": "true",
                    "enable_depth": "true",
                    "align_depth.enable": "true",
                    "enable_sync": "true",
                }.items(),
            )
        )
    actions.append(
        Node(
            package="asac_perception",
            executable="topview",
            name="topview",
            namespace="asac",
            output="screen",
            parameters=[get("params_file"), params],
        )
    )
    return actions


def generate_launch_description():
    share = Path(get_package_share_directory("asac_perception"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("model_path"),
            DeclareLaunchArgument(
                "params_file", default_value=str(share / "config/topview_real.yaml")
            ),
            DeclareLaunchArgument("start_camera", default_value="false"),
            DeclareLaunchArgument("rgb_only", default_value=""),
            OpaqueFunction(function=start),
        ]
    )
