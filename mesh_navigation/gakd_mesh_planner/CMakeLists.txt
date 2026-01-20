cmake_minimum_required(VERSION 3.8)
project(gakd_mesh_planner)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Optimization flags for native GP performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake_ros REQUIRED)
find_package(mbf_mesh_core REQUIRED)
find_package(mbf_msgs REQUIRED)
find_package(mbf_utility REQUIRED)
find_package(mesh_map REQUIRED)
find_package(rclcpp REQUIRED)
find_package(LVR2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)

set(CNPY_INCLUDE_DIR /usr/local/include)
set(CNPY_LIBRARY /usr/local/lib/libcnpy.a)

pluginlib_export_plugin_description_file(mbf_mesh_core gakd_mesh_planner.xml)

# Native GP library
add_library(native_gp SHARED
  src/native_gp.cpp
)

target_include_directories(native_gp PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
  ${CNPY_INCLUDE_DIR}
)

target_link_libraries(native_gp
  ${CNPY_LIBRARY}
  z
)

# Main library
add_library(${PROJECT_NAME}
  src/gakd_mesh_planner.cpp
  src/data_collector.cpp
  src/wavefront.cpp
  src/terrain_dynamics.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${LVR2_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${CNPY_INCLUDE_DIR}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE "GAKD_MESH_PLANNER_BUILDING_LIBRARY")

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
  ${CNPY_LIBRARY}
  native_gp
  CGAL::CGAL
  z
)

install(DIRECTORY include DESTINATION include)

install(TARGETS ${PROJECT_NAME} native_gp
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(test_terrain_dynamics test/test_terrain_dynamics.cpp)
target_link_libraries(test_terrain_dynamics ${PROJECT_NAME} native_gp)

add_executable(validate_terrain_dynamics test/validate_terrain_dynamics.cpp)
target_link_libraries(validate_terrain_dynamics ${PROJECT_NAME} native_gp)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME})
ament_export_dependencies(mbf_mesh_core mbf_msgs mbf_utility mesh_map rclcpp)

ament_package()