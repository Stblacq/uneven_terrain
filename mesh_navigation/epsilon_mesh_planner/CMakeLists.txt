cmake_minimum_required(VERSION 3.8)
project(epsilon_mesh_planner)

# Default release
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

find_package(ament_cmake_ros REQUIRED)
find_package(mbf_mesh_core REQUIRED)
find_package(mbf_msgs REQUIRED)
find_package(mbf_utility REQUIRED)
find_package(mesh_map REQUIRED)
find_package(rclcpp REQUIRED)
find_package(LVR2 REQUIRED)
find_package(Eigen3 REQUIRED)

# Adjust paths if you installed TensorFlow C API and cnpy system-wide
set(TENSORFLOW_INCLUDE_DIR /usr/local/include)
set(TENSORFLOW_LIBRARY /usr/local/lib/libtensorflow.so)
set(CNPY_INCLUDE_DIR /usr/local/include)
set(CNPY_LIBRARY /usr/local/lib/libcnpy.a)

pluginlib_export_plugin_description_file(mbf_mesh_core epsilon_mesh_planner.xml)

add_library(${PROJECT_NAME}
  src/epsilon_mesh_planner.cpp
  src/data_collector.cpp
  src/wavefront.cpp
  src/terrain_dynamics.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${LVR2_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${TENSORFLOW_INCLUDE_DIR}
  ${CNPY_INCLUDE_DIR}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE "EPSILON_MESH_PLANNER_BUILDING_LIBRARY")

ament_target_dependencies(${PROJECT_NAME}
  mbf_mesh_core
  mbf_msgs
  mbf_utility
  mesh_map
  rclcpp
)

target_link_libraries(${PROJECT_NAME}
  ${LVR2_LIBRARIES}
  ${MPI_CXX_LIBRARIES}
  ${TENSORFLOW_LIBRARY}
  ${CNPY_LIBRARY}
)

install(DIRECTORY include DESTINATION include)
install(TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(test_terrain_dynamics test/test_terrain_dynamics.cpp)
target_link_libraries(test_terrain_dynamics ${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(mbf_mesh_core mbf_msgs mbf_utility mesh_map rclcpp)

ament_package()
