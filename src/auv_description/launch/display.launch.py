import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_name = 'auv_description'
    pkg_dir = get_package_share_directory(pkg_name)
    
    # 1. Process the Xacro file using the xacro python library
    xacro_file = os.path.join(pkg_dir, 'urdf', 'auv.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_description = {'robot_description': robot_description_config.toxml()}
    
    # 2. Path to the RViz configuration file
    rviz_config_file = os.path.join(pkg_dir, 'rviz', 'auv.rviz')
    
    # 3. Define the nodes
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description]
    )
    
    node_joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        output='screen'
    )
    
    node_rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file]
    )
    
    # 4. Return the LaunchDescription
    return LaunchDescription([
        node_robot_state_publisher,
        node_joint_state_publisher,
        node_rviz
    ])