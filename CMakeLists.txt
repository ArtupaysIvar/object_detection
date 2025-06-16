cmake_minimum_required(VERSION 3.5)
project(Object_detection)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(orb_node src/orb_node.cpp)
ament_target_dependencies(orb_node rclcpp sensor_msgs cv_bridge)
target_link_libraries(orb_node ${OpenCV_LIBRARIES})

install(TARGETS
  orb_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
