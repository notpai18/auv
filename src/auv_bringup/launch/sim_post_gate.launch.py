#!/usr/bin/env python3
"""
sim_post_gate.launch.py  —  AUV green-mat mission starting AFTER gate crossing.

Use this when you want to test / tune the green-mat pipeline without running
the full gate sequence.  The AUV is spawned at a position past the gate and
the green navigator is triggered immediately (no gate nodes are launched).

Differences from sim_mission.launch.py
---------------------------------------
  X  gate_detector_node      — NOT started
  X  gate_localizer_node     — NOT started
  X  gate_navigator_v2_node  — NOT started
  ✓  green_detector_node     — started as normal
  ✓  green_localizer_node    — started as normal
  ✓  green_navigator_node    — started and IMMEDIATELY triggered with a
                               one-shot "DONE" publish on /auv/mission_state
                               so it skips its IDLE wait and enters SEARCH_GREEN.

Spawn position
--------------
  Default spawn is placed just past the gate, facing the green mat area.
  Override with:  spawn_x, spawn_y, spawn_z, spawn_yaw arguments.

Usage
-----
  source ~/auv_ws/install/setup.bash
  ros2 launch auv_bringup sim_post_gate.launch.py

Optional arguments
------------------
  world:=<path>       Path to .sdf world file (default: ~/auv_ws/camera_test.sdf)
  spawn_x:=10.0       AUV spawn X position (default: 10.0)
  spawn_y:=10.0       AUV spawn Y position (default: 10.0)
  spawn_z:=-4.238     AUV spawn Z position (default: -4.238)
  spawn_yaw:=1.57     AUV spawn yaw in radians (default: 1.57)
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
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── Prevent Gazebo from permanently grabbing the mouse ───────────────
    os.environ.setdefault('SDL_MOUSE_RELATIVE_MODE_WARP', '0')
    os.environ.setdefault('SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS', '0')
    os.environ.setdefault('GDK_BACKEND', 'x11')

    # ── Launch arguments ─────────────────────────────────────────────────
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.expanduser('~/auv_ws/camera_test.sdf'),
        description='Path to Gazebo world SDF file'
    )
    # Spawn the AUV on the far side of the gate, aimed toward the green mat.
    # The green mat is at world coords x=10, y=22 (from sim_params.yaml).
    # Default spawn is ~12 m from the mat so the navigator has room to search.
    spawn_x_arg   = DeclareLaunchArgument('spawn_x',   default_value='10.0',
                                          description='AUV spawn X (past the gate)')
    spawn_y_arg   = DeclareLaunchArgument('spawn_y',   default_value='10.0',
                                          description='AUV spawn Y (past the gate)')
    spawn_z_arg   = DeclareLaunchArgument('spawn_z',   default_value='-4.23800',
                                          description='AUV spawn Z (depth)')
    spawn_yaw_arg = DeclareLaunchArgument('spawn_yaw', default_value='1.57',
                                          description='AUV spawn yaw (radians)')

    world     = LaunchConfiguration('world')
    spawn_x   = LaunchConfiguration('spawn_x')
    spawn_y   = LaunchConfiguration('spawn_y')
    spawn_z   = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    # ── Package share paths ───────────────────────────────────────────────
    auv_desc_share    = get_package_share_directory('auv_description')
    auv_bringup_share = get_package_share_directory('auv_bringup')
    model_sdf_path    = os.path.join(auv_desc_share, 'models', 'auv_box', 'model.sdf')
    rviz_config       = os.path.join(auv_desc_share, 'rviz', 'auv.rviz')
    bridge_launch     = os.path.join(auv_desc_share, 'launch', 'bridge.launch.py')
    sim_params_yaml   = os.path.join(auv_bringup_share, 'config', 'sim_params.yaml')

    workspace_share_dir = os.path.dirname(auv_desc_share)

    # ── 1. Gazebo ─────────────────────────────────────────────────────────
    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world],
        output='screen',
        name='gazebo',
        additional_env={
            'SDL_MOUSE_RELATIVE_MODE_WARP': '0',
            'SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS': '0',
            'GDK_BACKEND': 'x11',
            'GZ_SIM_RESOURCE_PATH': workspace_share_dir,
        },
    )

    # ── Cleanup hook: re-enable mouse after Gazebo exits ─────────────────
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

    # ── 2. Spawn AUV past the gate (8 s delay — wait for Gazebo startup) ──
    spawn_auv = TimerAction(
        period=8.0,
        actions=[
            LogInfo(msg='[sim_post_gate] Spawning AUV past gate position...'),
            ExecuteProcess(
                cmd=[
                    'ros2', 'run', 'ros_gz_sim', 'create',
                    '-world', 'auv_pool_world',
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

    # ── 3. ROS <-> Gazebo bridge (10 s delay) ────────────────────────────
    bridge = TimerAction(
        period=10.0,
        actions=[
            LogInfo(msg='[sim_post_gate] Starting ROS <-> Gazebo bridge...'),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(bridge_launch)
            ),
        ]
    )

    # ── 4. Stereo depth processing (12 s delay — after bridge is up) ──────
    stereo_proc_share = get_package_share_directory('stereo_image_proc')
    stereo_proc = TimerAction(
        period=12.0,
        actions=[
            LogInfo(msg='[sim_post_gate] Starting stereo_image_proc...'),
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

    # ── 5–12. Mission nodes (14 s delay — gate nodes excluded) ────────────
    ros_nodes = TimerAction(
        period=14.0,
        actions=[
            LogInfo(msg='[sim_post_gate] Starting green-mat pipeline (no gate nodes)...'),

            # ── Green mat pipeline ───────────────────────────────────────

            # 5. Green detector (HSV colour segmentation — front + bottom cameras)
            Node(
                package='auv_vision',
                executable='green_detector_node',
                name='green_detector_node',
                output='screen',
                parameters=[sim_params_yaml],
            ),

            # 6. Green localizer (stereo Phase 1 + bottom Phase 2)
            Node(
                package='auv_vision',
                executable='green_localizer_node',
                name='green_localizer_node',
                output='screen',
                parameters=[sim_params_yaml],
            ),

            # ── Blue bin pipeline ────────────────────────────────────────

            # 7a. YOLO blue bin detector — handles the FRONT camera path only.
            #     (blue_bin_detection_front_2d → blue_bin_localizer_node Phase 1)
            Node(
                package='auv_vision',
                executable='blue_bin_detector_node',
                name='blue_bin_detector_node',
                output='screen',
                parameters=[sim_params_yaml],
            ),

            # 7b. HSV blue-in-green CV detector — handles the BOTTOM camera path.
            #     Publishes to /auv/blue_bin_detection_bottom_2d.
            #     Replaces the YOLO bottom path: structurally cannot false-positive
            #     on open blue water because it requires spatial overlap with the
            #     green mat in the same frame.
            Node(
                package='auv_vision',
                executable='blue_bin_cv_detector_node',
                name='blue_bin_cv_detector_node',
                output='screen',
                parameters=[sim_params_yaml],
            ),

            # 8. Blue bin localizer (fuses front + bottom detections into 3D position)
            Node(
                package='auv_vision',
                executable='blue_bin_localizer_node',
                name='blue_bin_localizer_node',
                output='screen',
                parameters=[sim_params_yaml],
            ),

            # ── Telemetry ────────────────────────────────────────────────

            # 9. Mission logger (CSV telemetry)
            Node(
                package='auv_telemetry',
                executable='mission_logger_node',
                name='mission_logger_node',
                output='screen',
                parameters=[sim_params_yaml],
            ),

            # 10. Trail mapper (trajectory recorder)
            Node(
                package='auv_telemetry',
                executable='trail_mapper_node',
                name='trail_mapper_node',
                output='screen',
                parameters=[sim_params_yaml],
            ),

            # ── Planner ──────────────────────────────────────────────────

            # 11. Green navigator — waits for /auv/mission_state == 'DONE'.
            #     The one-shot trigger below (at t=16 s) publishes DONE so
            #     the navigator immediately leaves IDLE -> SEARCH_GREEN.
            Node(
                package='auv_planner',
                executable='green_navigator_node',
                name='green_navigator_node',
                output='screen',
                parameters=[sim_params_yaml],
            ),

            # ── Visualisation ────────────────────────────────────────────

            # 12. RViz2
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_config],
                output='screen',
                name='rviz2'
            ),

            # 13. Live left camera view
            Node(
                package='image_view',
                executable='image_view',
                name='camera_view',
                output='screen',
                remappings=[('image', '/model/auv_box/stereo_front/left/image_raw')],
            ),

            # 14. Mission monitor (real-time status in terminal)
            Node(
                package='auv_telemetry',
                executable='mission_monitor_node',
                name='mission_monitor_node',
                output='screen',
            ),
        ]
    )

    # ── Trigger green navigator (16 s — 2 s after nodes start) ───────────
    # green_navigator_node idles in IDLE state until it receives
    # /auv/mission_state == 'DONE'.  Since we have no gate_navigator here,
    # we fire that signal once via ros2 topic pub --once.
    trigger_green = TimerAction(
        period=16.0,
        actions=[
            LogInfo(msg='[sim_post_gate] Publishing DONE → green_navigator entering SEARCH_GREEN...'),
            ExecuteProcess(
                cmd=[
                    'ros2', 'topic', 'pub',
                    '--once',
                    '/auv/mission_state',
                    'std_msgs/msg/String',
                    '{data: DONE}',
                ],
                output='screen',
                name='gate_done_trigger',
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
        LogInfo(msg='[sim_post_gate] Launching AUV post-gate green-mat mission (simulation)...'),
        gazebo,
        mouse_cleanup,
        spawn_auv,
        bridge,
        stereo_proc,
        ros_nodes,
        trigger_green,   # fires DONE so green_navigator skips IDLE immediately
    ])
