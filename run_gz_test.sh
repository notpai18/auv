#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/pai/auv_ws/install/setup.bash

export DISPLAY=:1
export GDK_BACKEND=x11
export SDL_MOUSE_RELATIVE_MODE_WARP=0
export SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS=0
export GZ_SIM_RESOURCE_PATH="$(ros2 pkg prefix auv_description)/share"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export EGL_PLATFORM=x11

pkill -9 -f "gz sim" 2>/dev/null
sleep 2

WORLD=/home/pai/auv_ws/camera_test.sdf

echo "Starting Gazebo with camera_test pool world..."
echo "  World: $WORLD"
nohup gz sim -r "$WORLD" > /tmp/gz_test.log 2>&1 &
GZ_PID=$!
echo "Gazebo PID: $GZ_PID"
sleep 10
echo "Tailing log:"
tail -n 20 /tmp/gz_test.log

echo "Starting image bridge..."
nohup ros2 run ros_gz_image image_bridge /model/auv_box/bottom/image_raw > /tmp/bridge_test.log 2>&1 &
sleep 5
echo "Topic list:"
ros2 topic list | grep bottom
