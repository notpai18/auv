#!/bin/bash
source /opt/ros/jazzy/setup.bash
source /home/pai/auv_ws/install/setup.bash

# Source dave_ws so OceanCurrentWorldPlugin and OceanCurrentPlugin can be found
if [ -f /home/pai/dave_ws/install/setup.bash ]; then
  source /home/pai/dave_ws/install/setup.bash
fi

export DISPLAY=:1
export GDK_BACKEND=x11
export SDL_MOUSE_RELATIVE_MODE_WARP=0
export SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS=0
export GZ_SIM_RESOURCE_PATH="$(ros2 pkg prefix auv_description)/share"
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export EGL_PLATFORM=x11

# Inject Dave plugin paths (same as sim_mission.launch.py does automatically)
DAVE_INSTALL=/home/pai/dave_ws/install
export GZ_SIM_SYSTEM_PLUGIN_PATH="${DAVE_INSTALL}/dave_gz_world_plugins/lib:${DAVE_INSTALL}/dave_gz_model_plugins/lib/dave_gz_model_plugins:${DAVE_INSTALL}/dave_ros_gz_plugins/lib/dave_ros_gz_plugins:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
export AMENT_PREFIX_PATH="${DAVE_INSTALL}/dave_worlds:${AMENT_PREFIX_PATH}"

pkill -9 -f "gz sim" 2>/dev/null
sleep 2

# Use dave_pool.sdf — has water volume + Dave animated ocean surface + ocean current plugin.
# To fall back to the old world without water: change path to camera_test.sdf
WORLD=$(ros2 pkg prefix auv_description)/share/auv_description/worlds/dave_pool.sdf

echo "Starting Gazebo with Dave pool world..."
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
