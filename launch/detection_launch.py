import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Path to SICK lidar launch file (precompiled binary)
    sick_launch_file = os.path.join(
        get_package_share_directory('sick_scan_xd'),
        'launch',
        'sick_multiscan.launch.py'
    )

    return LaunchDescription([

        # RealSense Camera Node
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense_camera',
            output='screen',
            parameters=[{
                'align_depth': True,
                'enable_pointcloud': True
            }]
        ),

        # Ultralytics YOLO Detector Node
        Node(
            package='ultralyticsYOLO_ros',
            executable='detector',
            name='human_detector_node',
            output='screen'
        ),

        # Include SICK LiDAR launch file with CLI-style arguments
        # Node(
            # package='sick_scan_xd',
            # executable='sick_generic_caller',
            # output='screen',
            # arguments=['/opt/ros/humble/share/sick_scan_xd/launch/sick_multiscan.launch', 'hostname:=192.168.0.1', 'udp_receiver_ip:=192.168.0.5']
        # )
    ])

