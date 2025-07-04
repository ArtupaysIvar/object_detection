cmake_minimum_required(VERSION 3.10)
project(object_detection)

find_package(image_transport REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(orb_node src/orb_node.cpp)
ament_target_dependencies(orb_node rclcpp cv_bridge std_msgs sensor_msgs cv_bridge OpenCV)
target_link_libraries(orb_node ${OpenCV_LIBRARIES})

add_executable(trial_node src/trial_node.cpp)
ament_target_dependencies(trial_node rclcpp cv_bridge std_msgs sensor_msgs cv_bridge OpenCV image_transport)
target_link_libraries(trial_node ${OpenCV_LIBRARIES})

add_executable(compressed_node src/compressed_node.cpp)
ament_target_dependencies(compressed_node rclcpp cv_bridge std_msgs sensor_msgs cv_bridge OpenCV image_transport)
target_link_libraries(compressed_node ${OpenCV_LIBRARIES})


install(TARGETS
  orb_node
  trial_node
  compressed_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
