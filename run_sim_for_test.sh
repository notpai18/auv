#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/pai/auv_ws/install/setup.bash

pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "ros2 launch" 2>/dev/null
sleep 2

nohup ros2 launch auv_bringup sim_mission.launch.py > /tmp/sim_test.log 2>&1 &
LAUNCH_PID=$!
echo "Launch PID: $LAUNCH_PID"

echo "Waiting for 25 seconds for simulator to start..."
sleep 25

echo "Killing planner nodes to prevent AUV from moving..."
pkill -f "gate_navigator_node"
pkill -f "green_navigator_node"

echo "Running python altitude test script..."
python3 /home/pai/auv_ws/src/auv_vision/auv_vision/bottom_cam_altitude_test.py
