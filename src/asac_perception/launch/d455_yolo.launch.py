from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def detector_action(context, prefix):
    overrides = {
        'model_path': ParameterValue(LaunchConfiguration('model_path'), value_type=str),
        'device': ParameterValue(LaunchConfiguration('device'), value_type=str),
        'target_classes': ParameterValue(LaunchConfiguration('target_classes'), value_type=str),
        'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
        'color_topic': prefix + ['/color/image_raw'],
        'depth_topic': prefix + ['/aligned_depth_to_color/image_raw'],
        'camera_info_topic': prefix + ['/color/camera_info'],
    }
    # 지정하지 않은 launch 인자는 YAML 값을 유지한다.
    for name, value_type in (
        ('image_size', int), ('cpu_threads', int),
        ('inference_rate_hz', float), ('max_frame_age_sec', float),
    ):
        value = LaunchConfiguration(name).perform(context)
        if value:
            overrides[name] = value_type(value)
    return [Node(
        package='asac_perception', executable='detector', name='detector',
        namespace='asac', output='screen',
        parameters=[LaunchConfiguration('params_file'), overrides],
    )]


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory('asac_perception'))
    camera_namespace = LaunchConfiguration('camera_namespace')
    camera_name = LaunchConfiguration('camera_name')
    # 카메라 namespace/name 변경 시 세 입력 토픽도 함께 변경한다.
    prefix = ['/', camera_namespace, '/', camera_name]
    return LaunchDescription([
        DeclareLaunchArgument('model_path', description='Trained YOLOv8 .pt path'),
        DeclareLaunchArgument('start_camera', default_value='true'),
        DeclareLaunchArgument('camera_namespace', default_value='camera'),
        DeclareLaunchArgument('camera_name', default_value='camera'),
        DeclareLaunchArgument('serial_no', default_value="''"),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('target_classes', default_value=''),
        DeclareLaunchArgument('image_size', default_value=''),
        DeclareLaunchArgument('cpu_threads', default_value=''),
        DeclareLaunchArgument('inference_rate_hz', default_value=''),
        DeclareLaunchArgument('max_frame_age_sec', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('color_profile', default_value='640x480x30'),
        DeclareLaunchArgument('depth_profile', default_value='640x480x30'),
        DeclareLaunchArgument('params_file', default_value=str(share / 'config/detector.yaml')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(PathJoinSubstitution([
                FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py',
            ])),
            condition=IfCondition(LaunchConfiguration('start_camera')),
            launch_arguments={
                'camera_namespace': camera_namespace,
                'camera_name': camera_name,
                'serial_no': LaunchConfiguration('serial_no'),
                'enable_color': 'true',
                'enable_depth': 'true',
                'align_depth.enable': 'true',
                'enable_sync': 'true',
                'rgb_camera.color_profile': LaunchConfiguration('color_profile'),
                'depth_module.depth_profile': LaunchConfiguration('depth_profile'),
            }.items(),
        ),
        OpaqueFunction(function=detector_action, kwargs={'prefix': prefix}),
    ])
