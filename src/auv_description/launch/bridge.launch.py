from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Image bridge: raw left/right images
    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=[
            '/model/auv_box/stereo_front/left/image_raw',
            '/model/auv_box/stereo_front/right/image_raw'
        ],
        output='screen'
    )

    # Parameter bridge: cmd_vel, odometry, and camera_info (calibration) for both cameras
    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        remappings=[('/model/auv_box/odometry', '/auv/odom')],
        arguments=[
            '/model/auv_box/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            '/model/auv_box/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/model/auv_box/stereo_front/left/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
            '/model/auv_box/stereo_front/right/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo'
        ],
        output='screen'
    )

    return LaunchDescription([
        image_bridge,
        parameter_bridge
    ])
