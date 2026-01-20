# GAKD Demo

This repository contains a demonstration of our recently published work:

> **A Genetic Approach to Gradient-Free Kinodynamic Planning in Uneven Terrains**  
> Otobong Jerome, Alexandr Klimchik, Alexander Maloletov, Geesara Kulathunga  
> *IEEE Robotics and Automation Letters (RA-L), Vol. 10, No. 6, pp. 5521-5528, 2025*  
> [DOI: 10.1109/LRA.2025.3560883](https://doi.org/10.1109/LRA.2025.3560883)

The demo showcases kinodynamic planning on mesh-based terrain representations using genetic heuristics for the **AgileX Scout Mini** robot.

## Overview

This package integrates components from several open-source projects to enable motion planning on uneven terrains.

- **Base Framework:**  
  Adapted from [mesh_navigation_tutorials](https://github.com/naturerobots/mesh_navigation_tutorials)

- **Robot Platform:**  
  [Scout ROS 2 Interface](https://github.com/westonrobot/scout_ros2)

- **Terrain Representation & Navigation Core:**  
  [Mesh Navigation](https://github.com/naturerobots/mesh_navigation)

- **Planning Framework:**  
  [Move Base Flex](https://github.com/naturerobots/move_base_flex)

## Dependencies

This demo is designed for **ROS 2 Humble** and requires the following:

- `ros2_control` and compatible robot controllers
- `ros2_topic_tools`
- Git Large File Storage (LFS)

## Setup Instructions

1. **Clone the Repository**

   ```bash
   mkdir -p ~/mesh_ws/src
   cd ~/mesh_ws/src
   git clone https://github.com/Stblacq/mesh_planner.git
   cd mesh_planner
   git lfs pull
   ```

2. **Install Dependencies**

   Ensure all required packages are sourced and installed. Install external dependencies (`scout_ros2`, `mesh_navigation`, `move_base_flex`) into the same workspace or overlayed ones.

   ```
    # Core ROS 2 Build Tools
    sudo apt install ros-humble-ament-cmake

    # ASIO Library (for ugv_sdk)
    sudo apt install libasio-dev

    # GSL (GNU Scientific Library) for lvr2
    sudo apt install libgsl-dev

    # OpenCL for mesh processing
    sudo apt install ocl-icd-opencl-dev opencl-headers

    # CGAL for mesh algorithms
    sudo apt install libcgal-dev

    # Gazebo simulation
    sudo apt install ros-humble-ros-gz-sim ros-humble-ros-gz

    # ROS 2 Control
    sudo apt install ros-humble-controller-manager
    sudo apt install ros-humble-gz-ros2-control
    sudo apt install ros-humble-joint-state-broadcaster ros-humble-diff-drive-controller

    # Other ROS packages
    sudo apt install ros-humble-topic-tools
    sudo apt install ros-humble-robot-localization

    # TensorFlow C library (manual install)
    wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.10.0.tar.gz
    sudo tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.10.0.tar.gz
    sudo ldconfig

    # cnpy library (built from source)
    cd ~/workspaces/RL_ws/src
    git clone https://github.com/rogersce/cnpy.git
    cd cnpy && mkdir build && cd build
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
    make && sudo make install
    sudo ldconfig

    ```
3. **Build the Workspace**

   ```bash
   cd ~/mesh_ws
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install
   ```

4. **Launch the Demo**

   ```bash
   source ~/mesh_ws/install/setup.bash
   ros2 launch terrain_nav terrain_nav_launch.py world_name:=desert
   ```

## Notes

### Collision Avoidance
This demo does **not** include collision avoidance. For collision-aware planning, integrate the **Wavefront Planner** from [Mesh Navigation](https://github.com/naturerobots/mesh_navigation) as a global planner by:

- Using vertex costs for obstacle-aware planning
- Setting up semantic layers for parts of the mesh (e.g., traversability)

Refer to the [Mesh Navigation documentation](https://github.com/naturerobots/mesh_navigation) for details on setting up semantic layers and planners.

## Videos
[![Genetic Kinodynamic Planning Demo](https://img.youtube.com/vi/IYRplkzxtIA/0.jpg)](https://youtu.be/IYRplkzxtIA)


### 🎥 Gradient-Free Kinodynamic Planning

[![Gradient-Free Kinodynamic Planning](https://img.youtube.com/vi/KI9RNQK9xW4/0.jpg)](https://youtu.be/KI9RNQK9xW4)

## License

This project is shared for academic and demonstration purposes in conjunction with our publication. Refer to each included project's license for further details.

## Contributions

We welcome contributions and feedback. For questions, open an issue.

Kindly star ⭐ this project if it helps you.