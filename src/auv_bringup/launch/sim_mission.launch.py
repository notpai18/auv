#!/usr/bin/env python3
"""
sim_mission.launch.py  —  Full AUV gate + green-mat mission in Gazebo simulation.

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
  7. Green detector           (HSV — publishes /auv/green_detection_*_2d)
  8. Green localizer          (stereo+bottom — publishes /auv/green_position_3d)
  9. Mission logger           (CSV — ~/auv_ws/mission_logs/)
 10. Trail mapper             (trajectory — publishes /auv/trail_map)
 11. Gate navigator           (gate FSM — drives AUV through gate, publishes DONE)
 12. Green navigator          (green FSM — waits for DONE, then finds green mat)
 13. RViz2                    (pre-configured visualizer)
 14. Image view               (live left camera feed)

Workflow
--------
  FORWARD (8m) → SEARCH (strafe) → TRACK → ALIGN → CROSS → DONE
    └─ triggers Green Navigator ──→ SEARCH_GREEN → APPROACH_GREEN
                                  → CENTER_GREEN → HOLD → FINAL_DONE

Usage
-----
  source ~/auv_ws/install/setup.bash
  ros2 launch auv_bringup sim_mission.launch.py

Water model (DAVE)
------------------
  The default world is auv_description/worlds/dave_pool.sdf — the competition
  pool carrying DAVE's ocean current plugins. Pool geometry, prop positions,
  lighting, fog and physics are unchanged from the old camera_test.sdf: gravity
  is still zero and buoyancy is off, so the AUV holds its spawn depth and the
  vision pipeline needs no retuning. DAVE graded buoyancy is present in that
  file but commented out, with the steps to enable it.

  This needs dave_ws for the current plugins. Their paths are injected below, so
  sourcing dave_ws by hand is optional but harmless.

  Either world can be used interchangeably:
    world:=$HOME/auv_ws/camera_test.sdf

Optional arguments
------------------
  world:=<path>     Path to .sdf world file
                    (default: auv_description/worlds/dave_pool.sdf)
  spawn_x:=1.0      AUV spawn X position (default: 1.0)
  spawn_y:=4.0      AUV spawn Y position (default: 4.0)
  spawn_z:=-4.23800 AUV spawn Z position (default: -4.23800)
  spawn_yaw:=0.0    AUV spawn yaw in radians (default: 0.0)
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
from launch.event_handlers import OnProcessExit, OnShutdown
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

    # ── DAVE workspace discovery ─────────────────────────────────────────
    # The DAVE world plugins live in a separate workspace. Rather than require
    # `source ~/dave_ws/install/setup.bash` before every run, find that install
    # tree here and inject the two things Gazebo and the plugins need:
    #
    #   GZ_SIM_SYSTEM_PLUGIN_PATH — so Gazebo can dlopen
    #     libOceanCurrentWorldPlugin.so and libOceanCurrentPlugin.so. Note these
    #     sit directly in <pkg>/lib/, not in the <pkg>/lib/<pkg>/ subdirectory
    #     the packages' own .dsv hooks point at, and those hooks are not even
    #     present in the installed share/ tree — so relying on them fails.
    #
    #   AMENT_PREFIX_PATH — OceanCurrentWorldPlugin calls
    #     get_package_share_directory("dave_worlds") while loading its transient
    #     current database, which throws if dave_worlds cannot be resolved.
    #
    # Override the location with DAVE_WS if the workspace lives elsewhere.
    dave_ws = os.environ.get('DAVE_WS', os.path.expanduser('~/dave_ws'))
    dave_install = os.path.join(dave_ws, 'install')

    dave_env = {}
    if os.path.isdir(dave_install):
        plugin_dirs = [
            os.path.join(dave_install, 'dave_gz_world_plugins', 'lib'),
            os.path.join(dave_install, 'dave_ros_gz_plugins', 'lib'),
            os.path.join(dave_install, 'dave_gz_model_plugins', 'lib'),
            os.path.join(dave_install, 'dave_gz_sensor_plugins', 'lib'),
        ]
        plugin_dirs = [d for d in plugin_dirs if os.path.isdir(d)]

        ament_dirs = [
            os.path.join(dave_install, pkg)
            for pkg in ('dave_worlds', 'dave_robot_models', 'dave_object_models',
                        'dave_sensor_models')
        ]
        ament_dirs = [d for d in ament_dirs if os.path.isdir(d)]

        def _prepend(var, dirs):
            existing = os.environ.get(var, '')
            parts = [p for p in existing.split(os.pathsep) if p]
            merged = dirs + [p for p in parts if p not in dirs]
            return os.pathsep.join(merged)

        if plugin_dirs:
            dave_env['GZ_SIM_SYSTEM_PLUGIN_PATH'] = _prepend(
                'GZ_SIM_SYSTEM_PLUGIN_PATH', plugin_dirs)
            # Plugins link against DAVE's own shared libs (gauss_markov_process,
            # tidal_oscillation, the msgs library) which sit alongside them.
            dave_env['LD_LIBRARY_PATH'] = _prepend('LD_LIBRARY_PATH', plugin_dirs)
        if ament_dirs:
            dave_env['AMENT_PREFIX_PATH'] = _prepend('AMENT_PREFIX_PATH', ament_dirs)

    # ── Launch arguments (override at CLI with arg:=value) ───────────────
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(
            get_package_share_directory('auv_description'), 'worlds', 'dave_pool.sdf'
        ),
        description='Path to Gazebo world SDF file (default: DAVE-water pool)'
    )
    spawn_x_arg   = DeclareLaunchArgument('spawn_x',   default_value='14.0')
    spawn_y_arg   = DeclareLaunchArgument('spawn_y',   default_value='4.0')
    spawn_z_arg   = DeclareLaunchArgument('spawn_z',   default_value='-4.23800')
    spawn_yaw_arg = DeclareLaunchArgument('spawn_yaw', default_value='1.57')

    world     = LaunchConfiguration('world')
    spawn_x   = LaunchConfiguration('spawn_x')
    spawn_y   = LaunchConfiguration('spawn_y')
    spawn_z   = LaunchConfiguration('spawn_z')
    spawn_yaw = LaunchConfiguration('spawn_yaw')

    # ── Package share paths ───────────────────────────────────────────────
    auv_desc_share  = get_package_share_directory('auv_description')
    auv_bringup_share = get_package_share_directory('auv_bringup')
    model_sdf_path  = os.path.join(
        auv_desc_share, 'models', 'auv_box', 'model.sdf'
    )
    rviz_config     = os.path.join(auv_desc_share, 'rviz', 'auv.rviz')
    bridge_launch   = os.path.join(auv_desc_share, 'launch', 'bridge.launch.py')

    # Sim-specific parameter file — passed to every mission node so that all
    # tunable values are configurable without editing source code.
    sim_params_yaml = os.path.join(auv_bringup_share, 'config', 'sim_params.yaml')

    # ── 1. Gazebo ─────────────────────────────────────────────────────────
    # additional_env merges into the inherited environment (no need to copy
    # os.environ — ExecuteProcess already inherits it by default).
    
    # Give Gazebo the path to the workspace's share directory so it can resolve package:// URIs
    workspace_share_dir = os.path.dirname(auv_desc_share)
    
    gazebo_env = {
        'SDL_MOUSE_RELATIVE_MODE_WARP': '0',   # disable SDL pointer grab
        'SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS': '0',
        'GDK_BACKEND': 'x11',                  # force X11 for reliable xinput
        'GZ_SIM_RESOURCE_PATH': workspace_share_dir,
    }
    # DAVE plugin/resource paths (empty dict if dave_ws was not found — Gazebo
    # then fails to load the ocean current plugins and logs an error, but the
    # world itself, including gravity and graded buoyancy, still runs).
    gazebo_env.update(dave_env)

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world],
        output='log',
        name='gazebo',
        additional_env=gazebo_env,
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
                    output='log',
                    name='mouse_cleanup',
                )
            ]
        )
    )

    # ── Shutdown notice ───────────────────────────────────────────────────
    # Every node above logs to file instead of the terminal (output='log'),
    # so once Ctrl+C is pressed the terminal goes silent while everything
    # shuts down — Gazebo in particular can take 10+ seconds to exit
    # cleanly (this is why run.md's manual workflow has its own
    # 'pkill -9 -f "gz sim"' cleanup step). With no on-screen feedback that
    # is easy to mistake for a hung launch, and pressing Ctrl+C again makes
    # ros2 launch escalate to an immediate SIGTERM/SIGKILL of every process
    # — which can catch trail_mapper_node mid-write and leave a truncated
    # map PNG/HTML. This message buys a moment of patience.
    shutdown_notice = RegisterEventHandler(
        OnShutdown(
            on_shutdown=[LogInfo(msg=(
                '\n[sim_mission] Shutting down — this can take up to '
                '15-20s (Gazebo exits slowly). Please WAIT, do not press '
                'Ctrl+C again — a second interrupt force-kills every node '
                'immediately and can truncate the saved map PNG/HTML.\n'
            ))]
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
                    '-world', 'auv_pool_world',
                    '-file',  model_sdf_path,
                    '-name',  'auv_box',
                    '-x', spawn_x,
                    '-y', spawn_y,
                    '-z', spawn_z,
                    '-Y', spawn_yaw,
                ],
                output='log',
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

            # ── Gate pipeline ────────────────────────────────────────────

            # 5. Gate detector (YOLOv8)
            Node(
                package='auv_vision',
                executable='gate_detector_node',
                name='gate_detector_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            # 6. Gate localizer (stereo 3D)
            Node(
                package='auv_vision',
                executable='gate_localizer_node',
                name='gate_localizer_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            # ── Green mat pipeline ───────────────────────────────────────

            # 7. Green detector (HSV colour segmentation — front + bottom cameras)
            Node(
                package='auv_vision',
                executable='green_detector_node',
                name='green_detector_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            # 8. Green localizer (stereo Phase 1 + bottom Phase 2)
            Node(
                package='auv_vision',
                executable='green_localizer_node',
                name='green_localizer_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            # ── Blue Bin pipeline ────────────────────────────────────────

            Node(
                package='auv_vision',
                executable='blue_bin_detector_node',
                name='blue_bin_detector_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            Node(
                package='auv_vision',
                executable='blue_bin_localizer_node',
                name='blue_bin_localizer_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            # ── Underwater view (display only) ───────────────────────────

            # Applies Beer-Lambert haze to the left camera using the disparity
            # stereo_image_proc produces, publishing /auv/underwater_view/image_raw.
            # This exists because Gazebo's <fog> is a no-op in Harmonic and the
            # world's water volume cannot tint a camera sitting inside it — see
            # the header of auv_vision/underwater_view_node.py.
            # Output is a SEPARATE topic: every detector still consumes the raw
            # feed, so no HSV or YOLO threshold changes.
            Node(
                package='auv_vision',
                executable='underwater_view_node',
                name='underwater_view_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            # ── Telemetry ────────────────────────────────────────────────

            # 9. Mission logger (CSV telemetry)
            Node(
                package='auv_telemetry',
                executable='mission_logger_node',
                name='mission_logger_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            # 10. Trail mapper (trajectory recorder)
            Node(
                package='auv_telemetry',
                executable='trail_mapper_node',
                name='trail_mapper_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            # ── Planners ─────────────────────────────────────────────────

            # 11. Gate navigator v2 — forward 8 m, strafe search, track, align, cross
            #     Publishes /auv/mission_state == 'DONE' when gate is crossed.
            #     To revert to round-trip navigator change executable to 'gate_navigator_node'
            Node(
                package='auv_planner',
                executable='gate_navigator_v2_node',
                name='gate_navigator_v2_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            # 12. Green navigator — IDLE until gate DONE, then finds green mat.
            #     Final states: HOLD → FINAL_DONE (full stop over green zone).
            Node(
                package='auv_planner',
                executable='green_navigator_node',
                name='green_navigator_node',
                output='log',
                parameters=[sim_params_yaml],
            ),

            # ── Visualisation ────────────────────────────────────────────

            # 13. RViz2 (pre-configured — no GUI setup needed)
            ExecuteProcess(
                cmd=['rviz2', '-d', rviz_config],
                output='log',
                name='rviz2'
            ),

            # 14. Live left camera view — shows the HAZED feed, so what you
            #     watch looks underwater. The detectors are unaffected; they
            #     still read the raw topic. To watch the raw feed instead,
            #     point this back at
            #     /model/auv_box/stereo_front/left/image_raw
            Node(
                package='image_view',
                executable='image_view',
                name='camera_view',
                output='log',
                remappings=[('image', '/auv/underwater_view/image_raw')],
            ),

            # 15. Mission monitor — prints real-time status in the launch terminal.
            # Kept on 'screen' (every other node above is 'log') so this is the
            # ONLY thing writing to the terminal — a clean, presentation-ready
            # live dashboard instead of 12 processes' logs interleaving with it.
            Node(
                package='auv_telemetry',
                executable='mission_monitor_node',
                name='mission_monitor_node',
                output='screen',
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
        shutdown_notice, # warns not to double-Ctrl+C while things shut down
        spawn_auv,
        bridge,
        stereo_proc,
        ros_nodes,
    ])
