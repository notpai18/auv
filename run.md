# AUV Gate Mission — Run Guide
# Open a NEW terminal for each numbered section. Run them IN ORDER.
# Wait for each terminal to be ready before moving to the next.

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
  -x -5 -y -4 -z 0.15 \
  -Y 2.7

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
# TERMINAL 7 — Mission logger
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 run auv_telemetry mission_logger_node

# ════════════════════════════════════════════════════════════
# TERMINAL 8 — Gate navigator (main state machine)
# States: SEARCH → TRACK → ALIGN → CROSS → STOP
# Publishes: /model/auv_box/cmd_vel  /auv/mission_state
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 run auv_planner gate_navigator_node

# ════════════════════════════════════════════════════════════
# TERMINAL 9 — Live camera view (left stereo camera)
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 run image_view image_view --ros-args -r image:=/model/auv_box/stereo_front/left/image_raw

# ════════════════════════════════════════════════════════════
# TERMINAL 10 — (Optional) Monitor mission state live
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 topic echo /auv/mission_state

# ════════════════════════════════════════════════════════════
# TERMINAL 11 — (Optional) Monitor 3D gate position live
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 topic echo /auv/gate_position_3d

# ════════════════════════════════════════════════════════════
# TERMINAL 12 — YOLO live detection feed
# Shows raw YOLO output: [detected, cx, cy, w, h, conf, 1.0]
# If "detected" = 0.0 always → YOLO cannot see the gate
# If "conf" stays below 0.30 → too far or model struggling
# ════════════════════════════════════════════════════════════
source ~/auv_ws/install/setup.bash
ros2 topic echo /auv/gate_detection_2d

# ════════════════════════════════════════════════════════════
# TERMINAL 13 — Debug image (YOLO annotated frame)
# YOLO saves its annotated image here every frame — open it
# in a viewer to see what the model is detecting in real time
# ════════════════════════════════════════════════════════════
eog /home/pai/yolo_debug/latest_detection.jpg
# OR use:
# feh --reload 1 /home/pai/yolo_debug/latest_detection.jpg
