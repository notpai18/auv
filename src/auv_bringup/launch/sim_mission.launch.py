#!/usr/bin/env python3
"""
sim_mission.launch.py  —  Full AUV gate mission in Gazebo simulation.

SIMULATION ONLY — uses Gazebo, model spawning, and the Gazebo-ROS bridge.
For the real AUV, use mission.launch.py (hardware drivers replace Gazebo).

Launch order
------------
  1. Gazebo simulator         (gz sim)
  2. AUV model spawn          (one-shot, delayed 8 s for Gazebo startup)
  3. ROS <-> Gazebo bridge    (image + parameter topics)
  4. Stereo depth processing  (disparity from left/right cameras)
  5. Gate detector            (YOLOv8 — publishes /auv/gate_detection_2d)
  6. Gate localizer           (stereo 3D — publishes /auv/gate_position_3d)
  7. Mission logger           (CSV — ~/auv_ws/mission_logs/)
  8. Trail mapper             (trajectory — publishes /auv/trail_map)
  9. Gate navigator           (main FSM — publishes /model/auv_box/cmd_vel)
 10. RViz2                    (pre-configured visualizer)
 11. Image view               (live left camera feed)

Usage
-----
  source ~/auv_ws/install/setup.bash
  ros2 launch auv_bringup sim_mission.launch.py

Optional arguments
------------------
  world:=<path>     Path to .sdf world file
                    (default: ~/auv_ws/camera_test.sdf)
  spawn_x:=-5.0     AUV spawn X position (default: -5.0)
  spawn_y:=-4.0     AUV spawn Y position (default: -4.0)
  spawn_z:=0.15     AUV spawn Z position (default: 0.15)
  spawn_yaw:=2.7    AUV spawn yaw in radians (default: 2.7)
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    TimerAction,
    LogInfo,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── Fix: prevent Gazebo from permanently grabbing the mouse ──────────
    # SDL_MOUSE_RELATIVE_MODE_WARP : disables SDL's relative-grab mode that
    #   locks the pointer to the Gazebo window.
    # GDK_BACKEND=x11             : force X11 backend so xinput can always
    #   re-enable the device if the window dies without releasing the grab.
    os.environ.setdefault('SDL_MOUSE_RELATIVE_MODE_WARP', '0')
    os.environ.setdefault('SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS', '0')
    os.environ.setdefault('GDK_BACKEND', 'x11')

    # ── Launch arguments (override at CLI with arg:=value) ───────────────
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.expanduser('~/auv_ws/camera_test.sdf'),
        description='Path to Gazebo world SDF file'
    )
    spawn_x_arg   = DeclareLaunchArgument('spawn_x',   default_value='-5.0')
    spawn_y_arg   = DeclareLaunchArgument('spawn_y',   default_value='-4.0')
    spawn_z_arg   = DeclareLaunchArgument('spawn_z',   default_value='0.15')
    spawn_yaw_arg = DeclareLaunchArgument('spawn_yaw', default_value='2.7')

    world     = LaunchConfiguration('world')
    spawn_x   = LaunchConfiguration('spawn_x')
    spawn_y   = LaunchConfiguration('spawn_y')
    spawn_z   = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    # ── Package share paths ───────────────────────────────────────────────
    auv_desc_share  = get_package_share_directory('auv_description')
    model_sdf_path  = os.path.join(
        auv_desc_share, 'models', 'auv_box', 'model.sdf'
    )
    rviz_config     = os.path.join(auv_desc_share, 'rviz', 'auv.rviz')
    bridge_launch   = os.path.join(auv_desc_share, 'launch', 'bridge.launch.py')

    # ── 1. Gazebo ─────────────────────────────────────────────────────────
    # additional_env merges into the inherited environment (no need to copy
    # os.environ — ExecuteProcess already inherits it by default).
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world],
        output='screen',
        name='gazebo',
        additional_env={
            'SDL_MOUSE_RELATIVE_MODE_WARP': '0',   # disable SDL pointer grab
            'SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS': '0',
            'GDK_BACKEND': 'x11',                  # force X11 for reliable xinput
        },
    )

    # ── Cleanup hook: re-enable mouse after Gazebo exits ─────────────────
    # If Gazebo crashes or is killed before it can call XUngrabPointer(),
    # this runs `xinput --enable` on every pointer device to recover input.
    mouse_cleanup = RegisterEventHandler(
        OnProcessExit(
            target_action=gazebo,
            on_exit=[
                ExecuteProcess(
                    cmd=[
                        'bash', '-c',
                        'xinput list --id-only | xargs -I{} sh -c '
                        '"xinput get-prop {} \'Device Enabled\' 2>/dev/null '
                        '&& xinput enable {} 2>/dev/null || true"'
                    ],
                    output='screen',
                    name='mouse_cleanup',
                )
            ]
        )
    )

    # ── 2. Spawn AUV (delayed 8 s — wait for Gazebo to fully start) ───────
    spawn_auv = TimerAction(
        period=8.0,
        actions=[
            LogInfo(msg='[sim_mission] Spawning AUV into Gazebo world...'),
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'ros_gz_sim', 'create',
                    '-world', 'camera_world',
                    '-file',  model_sdf_path,
                    '-name',  'auv_box',
                    '-x', spawn_x,
                    '-y', spawn_y,
                    '-z', spawn_z,
                    '-Y', spawn_yaw,
                ],
                output='screen',
                name='spawn_auv'
            ),
        ]
    )

    # ── 3. ROS <-> Gazebo bridge (delayed 1 s after spawn) ────────────────
    bridge = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg='[sim_mission] Starting ROS <-> Gazebo bridge...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bridge_launch)
            ),
        ]
    )

    # ── 4. Stereo depth processing (delayed 12 s — after bridge) ──────────
    stereo_proc_share = get_package_share_directory('stereo_image_proc')
    stereo_proc = TimerAction(
        period=12.0,
        actions=[
            LogInfo(msg='[sim_mission] Starting stereo_image_proc...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(stereo_proc_share, 'launch',
                                 'stereo_image_proc.launch.py')
                ),
                launch_arguments={
                    'left_namespace':  '/model/auv_box/stereo_front/left',
                    'right_namespace': '/model/auv_box/stereo_front/right',
                }.items()
            ),
        ]
    )

    # ── 5–11. ROS nodes (delayed 14 s — after stereo proc is ready) ───────
    ros_nodes = TimerAction(
        period=14.0,
        actions=[
            LogInfo(msg='[sim_mission] Starting all mission nodes...'),

            # 5. Gate detector (YOLOv8)
            Node(
                package='auv_vision',
                executable='gate_detector_node',
                name='gate_detector_node',
                output='screen',
            ),

            # 6. Gate localizer (stereo 3D)
            Node(
                package='auv_vision',
                executable='gate_localizer_node',
                name='gate_localizer_node',
                output='screen',
            ),

            # 7. Mission logger (CSV telemetry)
            Node(
                package='auv_telemetry',
                executable='mission_logger_node',
                name='mission_logger_node',
                output='screen',
            ),

            # 8. Trail mapper (trajectory recorder)
            Node(
                package='auv_telemetry',
                executable='trail_mapper_node',
                name='trail_mapper_node',
                output='screen',
            ),

            # 9. Gate navigator (main FSM — starts last so sensors are ready)
            Node(
                package='auv_planner',
                executable='gate_navigator_node',
                name='gate_navigator_node',
                output='screen',
            ),

            # 10. RViz2 (pre-configured — no GUI setup needed)
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_config],
                output='screen',
                name='rviz2'
            ),

            # 11. Live left camera view
            Node(
                package='image_view',
                executable='image_view',
                name='camera_view',
                output='screen',
                remappings=[('image', '/model/auv_box/stereo_front/left/image_raw')],
            ),

            # 12. Mission monitor — opens in its own gnome-terminal window
            # Shows live startup status + mission state dashboard (1 Hz refresh)
            ExecuteProcess(
                cmd=[
                    'gnome-terminal',
                    '--title=AUV Mission Monitor',
                    '--geometry=65x24',
                    '--',
                    'bash', '-c',
                    'source ~/auv_ws/install/setup.bash && '
                    'ros2 run auv_telemetry mission_monitor_node; '
                    'read -p "Press Enter to close..."',
                ],
                output='screen',
                name='mission_monitor'
            ),
        ]
    )

    return LaunchDescription([
        # Arguments
        world_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_z_arg,
        spawn_yaw_arg,

        # Sequence
        LogInfo(msg='[sim_mission] Launching AUV gate mission (simulation)...'),
        gazebo,
        mouse_cleanup,   # re-enables mouse if Gazebo exits without releasing grab
        spawn_auv,
        bridge,
        stereo_proc,
        ros_nodes,
    ])
