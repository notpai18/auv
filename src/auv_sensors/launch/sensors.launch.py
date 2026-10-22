"""Launch file for AUV Sensors Layer.

Starts:
  1. Gazebo <-> ROS 2 sensor bridge (parameter_bridge driven by config/bridge.yaml)
  2. Simulated Bar02 Depth Sensor node (depth_sensor)
  3. Doppler Velocity Log & Acoustic Altimeter (dvl_sensor)
  4. 6-DOF Sensor Fusion & State Estimator (sensor_fusion_node)
  5. Real-Time AUV Safety Supervisor (safety_supervisor_node)
  6. Simulated Leak Detector node (leak_mock)
  7. Simulated Battery Monitor node (battery_mock)
  8. (Optional) Standalone IMU Mock node (imu_mock) when running without Gazebo
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('auv_sensors')
    bridge_config = os.path.join(pkg_share, 'config', 'bridge.yaml')

    odom_topic = LaunchConfiguration('odom_topic')
    launch_bridge = LaunchConfiguration('launch_bridge')
    launch_battery = LaunchConfiguration('launch_battery')
    launch_imu_mock = LaunchConfiguration('launch_imu_mock')
    launch_dvl = LaunchConfiguration('launch_dvl')
    launch_fusion = LaunchConfiguration('launch_fusion')
    launch_safety = LaunchConfiguration('launch_safety')

    return LaunchDescription([
        DeclareLaunchArgument(
            'odom_topic', default_value='/auv/odom',
            description='Odometry topic for simulated sensor derivation'
        ),
        DeclareLaunchArgument(
            'launch_bridge', default_value='true',
            description='Whether to start the Gazebo <-> ROS 2 parameter bridge'
        ),
        DeclareLaunchArgument(
            'launch_battery', default_value='true',
            description='Whether to start the simulated battery monitor node'
        ),
        DeclareLaunchArgument(
            'launch_imu_mock', default_value='false',
            description='Whether to start the offline IMU mock node (set true when Gazebo is not running)'
        ),
        DeclareLaunchArgument(
            'launch_dvl', default_value='true',
            description='Whether to start the Doppler Velocity Log (DVL) node'
        ),
        DeclareLaunchArgument(
            'launch_fusion', default_value='true',
            description='Whether to start the 6-DOF Sensor Fusion & State Estimator node'
        ),
        DeclareLaunchArgument(
            'launch_safety', default_value='true',
            description='Whether to start the AUV Safety Supervisor node'
        ),

        # Gazebo <-> ROS 2 Bridge
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='sensor_bridge',
            parameters=[{'config_file': bridge_config}],
            output='screen',
            condition=IfCondition(launch_bridge)
        ),

        # Simulated Bar02 Depth Sensor
        Node(
            package='auv_sensors',
            executable='depth_sensor',
            name='depth_sensor',
            parameters=[{'odom_topic': odom_topic}],
            output='screen',
        ),

        # Doppler Velocity Log (DVL) & Acoustic Altimeter
        Node(
            package='auv_sensors',
            executable='dvl_sensor',
            name='dvl_sensor',
            parameters=[{'odom_topic': odom_topic}],
            output='screen',
            condition=IfCondition(launch_dvl)
        ),

        # 6-DOF Sensor Fusion & State Estimator
        Node(
            package='auv_sensors',
            executable='sensor_fusion_node',
            name='sensor_fusion_node',
            output='screen',
            condition=IfCondition(launch_fusion)
        ),

        # AUV Safety Supervisor & Failsafe Manager
        Node(
            package='auv_sensors',
            executable='safety_supervisor_node',
            name='safety_supervisor_node',
            output='screen',
            condition=IfCondition(launch_safety)
        ),

        # Simulated Leak Sensor (SOS bilge detector)
        Node(
            package='auv_sensors',
            executable='leak_mock',
            name='leak_mock',
            output='screen',
        ),

        # Simulated Battery Monitor
        Node(
            package='auv_sensors',
            executable='battery_mock',
            name='battery_mock',
            output='screen',
            condition=IfCondition(launch_battery)
        ),

        # Offline IMU Mock (disabled by default when Gazebo is active)
        Node(
            package='auv_sensors',
            executable='imu_mock',
            name='imu_mock',
            parameters=[{'odom_topic': odom_topic}],
            output='screen',
            condition=IfCondition(launch_imu_mock)
        ),
    ])
