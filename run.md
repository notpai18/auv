# AUV Gate Mission — Run Guide

# ════════════════════════════════════════════════════════════
# OPTION A1 — Full mission (gate → green mat)
# Starts everything automatically in the correct order.
# Wait ~14 seconds for all nodes to come up.
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 launch auv_bringup sim_mission.launch.py

# Custom spawn position (optional args):
# ros2 launch auv_bringup sim_mission.launch.py spawn_x:=14.0 spawn_y:=4.0 spawn_z:=-4.238 spawn_yaw:=1.57

# ════════════════════════════════════════════════════════════
# OPTION A2 — Post-gate only (green-mat pipeline, no gate nodes)
# AUV spawns past the gate; green navigator is triggered immediately.
# Use this to test/tune green detection without running the full mission.
# ════════════════════════════════════════════════════════════
# Custom spawn position past gate (optional args):
# ros2 launch auv_bringup sim_post_gate.launch.py spawn_x:=10.0 spawn_y:=10.0 spawn_z:=-4.238 spawn_yaw:=1.57

# ════════════════════════════════════════════════════════════════════════
# OPTION B — Manual terminals (use for debugging individual nodes)
# Open a NEW terminal for each numbered section. Run them IN ORDER.
# Wait for each terminal to be ready before moving to the next.
# ════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════
# TERMINAL 1 — Kill old Gazebo (run once before anything else)
# ════════════════════════════════════════════════════════════
pkill -9 -f "gz sim"

# ════════════════════════════════════════════════════════════
# TERMINAL 1 — Launch Gazebo
# Wait until the Gazebo window is fully open before Terminal 2
# ════════════════════════════════════════════════════════════
gz sim -r ~/auv_ws/camera_test.sdf

# ════════════════════════════════════════════════════════════
# TERMINAL 2 — Spawn the AUV into the world
# Wait for: "Entity creation successful."
#
# COMPLEX TEST SPAWN:
#   Position : x=-5, y=-4, z=0.15
#   Yaw      : 2.7 rad (~155°) → facing AWAY from gate
#   Distance : ~9.8m to gate at (4,0) → pushes YOLO/stereo to limit
#   Lateral  : 4m to the OPPOSITE side from original spawn
#   Challenge: AUV must SEARCH-spin ~180°, then track from far-left-rear
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 run ros_gz_sim create \
  -world camera_world \
  -file ~/auv_ws/install/auv_description/share/auv_description/models/auv_box/model.sdf \
  -name auv_box \
  -x -5 -y 0 -z 1.5 \
  -Y 0

# ════════════════════════════════════════════════════════════
# TERMINAL 3 — ROS <-> Gazebo bridge
# Wait for: bridge topics to appear (no error output)
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 launch auv_description bridge.launch.py

# ════════════════════════════════════════════════════════════
# TERMINAL 4 — Stereo depth processing (disparity)
# Wait for: no error output
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 launch stereo_image_proc stereo_image_proc.launch.py \
  left_namespace:=/model/auv_box/stereo_front/left \
  right_namespace:=/model/auv_box/stereo_front/right

# ════════════════════════════════════════════════════════════
# TERMINAL 5 — Gate detector (YOLOv8 v2 — whole gate, class 0)
# Publishes: /auv/gate_detection_2d → [detected, cx, cy, w, h, conf, 1.0]
# Wait for: "YOLOv8 model loaded successfully!"
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 run auv_vision gate_detector_node

# ════════════════════════════════════════════════════════════
# TERMINAL 6 — Gate localizer (stereo → 3D position)
# Publishes: /auv/gate_position_3d → [x_fwd, y_left, z_up, confidence]
# Filters:   detections below conf=0.40 are dropped
# Wait for: "Gate 3D Localizer Node started!"
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 run auv_vision gate_localizer_node

# ════════════════════════════════════════════════════════════
# TERMINAL 7 — Mission logger (CSV)
# Logs: all topics → ~/auv_ws/mission_logs/mission_run_<ts>.csv
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 run auv_telemetry mission_logger_node

# ════════════════════════════════════════════════════════════
# TERMINAL 8 — Gate navigator (main state machine)
# States: SEARCH → TRACK → ALIGN → CROSS → STOP → RETURN → DONE
# Publishes: /model/auv_box/cmd_vel  /auv/mission_state
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 run auv_planner gate_navigator_node

# ════════════════════════════════════════════════════════════
# TERMINAL 9 — Trail mapper
# Records AUV trajectory (every 0.1 m)
# Publishes: /auv/trail_map (nav_msgs/Path)
# Saves:     ~/auv_ws/mission_logs/trail_<ts>.csv on shutdown
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 run auv_telemetry trail_mapper_node

# ════════════════════════════════════════════════════════════
# TERMINAL 10 — RViz2 visualizer (pre-configured, no setup needed)
# Shows: AUV position (red arrow) + full trajectory trail (green line)
# Fixed frame: map | Background: dark
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
rviz2 -d ~/auv_ws/src/auv_description/rviz/auv.rviz

# ════════════════════════════════════════════════════════════
# TERMINAL 11 — Live camera view (left stereo camera)
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 run image_view image_view --ros-args -r image:=/model/auv_box/stereo_front/left/image_raw

# ════════════════════════════════════════════════════════════
# TERMINAL 12 — (Optional) Monitor mission state live
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 topic echo /auv/mission_state
