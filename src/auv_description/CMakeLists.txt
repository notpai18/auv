cmake_minimum_required(VERSION 3.8)
project(auv_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# We only need ament_cmake to build the package. 
# The other dependencies are runtime only, so they don't need find_package()
find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf rviz models meshes worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()