# Terrain-Adaptive ε-Constraint MPC for Uneven Terrain

This repository contains the ROS 2 implementation of the planner from:

> **A Terrain-Adaptive ε-Constraint MPC for Uneven Terrain Kinodynamic Planning**
> Otobong Jerome, Geesara Kulathunga, Tiago Nascimento
> *IEEE Robotics and Automation Letters (RA-L), 2026. To be presented at ICRA 2027.*
> [arXiv:2605.21188](https://arxiv.org/abs/2605.21188)

The planner drives a car-like robot across terrain represented as a triangle mesh. It treats path efficiency and vehicle stability as two separate objectives and tightens or relaxes the stability budget (ε) according to the local terrain, so the robot takes direct routes on gentle ground and conservative ones on rough or steep ground.

It runs as a [Move Base Flex](https://github.com/naturerobots/move_base_flex) planner plugin and was tested in Gazebo and on two physical robots, an **AgileX Scout Mini** and a **JetAuto**.

## Video

[![Terrain-Adaptive ε-Constraint MPC](https://img.youtube.com/vi/w6QKhI-CfOY/0.jpg)](https://youtu.be/w6QKhI-CfOY)

## What is in this repository

My own packages:

| Package | Purpose |
|---|---|
| `mesh_navigation/epsilon_mesh_planner` | The ε-constraint MPC planner (C++), plugin `epsilon_mesh_planner/EpsilonMeshPlanner` |
| `mesh_navigation/gakd_mesh_planner` | GAKD baseline, the genetic planner from our RA-L 2025 paper |
| `mesh_navigation/mppi_mesh_planner` | MPPI baseline |
| `terrain_nav/terrain_nav` | Launch files, Move Base Flex configuration, maps and RViz setup |
| `terrain_nav/terrain_nav_sim` | Gazebo worlds (`desert`, `inno`), robot model and ground-truth localisation |

Third-party packages included so the workspace builds in one step: [lvr2](https://github.com/uos/lvr2), [mesh_navigation](https://github.com/naturerobots/mesh_navigation), [mesh_tools](https://github.com/naturerobots/mesh_tools), [move_base_flex](https://github.com/naturerobots/move_base_flex), [scout_ros2](https://github.com/westonrobot/scout_ros2) and [ugv_sdk](https://github.com/westonrobot/ugv_sdk). The setup is adapted from [mesh_navigation_tutorials](https://github.com/naturerobots/mesh_navigation_tutorials).

## How the planner works

Each planning cycle, `epsilon_mesh_planner`:

1. Computes a guidance vector field over the mesh with a wavefront / Fast Marching pass from the goal.
2. Samples candidate control sequences over a receding horizon, biased toward the guidance field and warm-started from the previous solution.
3. Rolls each candidate forward through the vehicle-terrain model: analytical dynamics on the mesh plus a sparse Gaussian-process correction (Matérn 5/2 kernel with ARD, evaluated natively in C++ with Eigen).
4. Scores the candidates on two objectives: progress to the goal (f1) and tip-over stability relative to the terrain normal (f2).
5. Keeps the best candidate for f1 among those whose f2 stays within the budget ε, where ε is reduced from `epsilon_base` according to the local roughness and slope.

Terrain descriptors (normal, roughness, Gaussian and mean curvature) are precomputed per vertex and loaded at start-up. Mesh projection queries use a CGAL AABB tree.

## Dependencies

Built and tested on **ROS 2 Humble**.

```bash
# Build tools
sudo apt install ros-humble-ament-cmake

# Libraries
sudo apt install libasio-dev          # ugv_sdk
sudo apt install libgsl-dev           # lvr2
sudo apt install ocl-icd-opencl-dev opencl-headers
sudo apt install libcgal-dev          # mesh queries in the planners
sudo apt install libeigen3-dev

# Gazebo
sudo apt install ros-humble-ros-gz-sim ros-humble-ros-gz

# ROS 2 Control
sudo apt install ros-humble-controller-manager ros-humble-gz-ros2-control \
                 ros-humble-joint-state-broadcaster ros-humble-diff-drive-controller

# Other ROS packages
sudo apt install ros-humble-topic-tools ros-humble-robot-localization
```

The planners read their model files with [cnpy](https://github.com/rogersce/cnpy), which is built from source and expected under `/usr/local`:

```bash
git clone https://github.com/rogersce/cnpy.git
cd cnpy && mkdir build && cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
make && sudo make install
sudo ldconfig
```

Git LFS is required for the mesh and map files.

## Setup

```bash
mkdir -p ~/mesh_ws/src
cd ~/mesh_ws/src
git clone https://github.com/Stblacq/uneven_terrain.git
cd uneven_terrain
git lfs pull

cd ~/mesh_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
```

### Point the planner at its data files

`terrain_nav/terrain_nav/config/mbf_mesh_nav.yaml` refers to the planner's data files by absolute path. Before the first run, change these four entries under `epsilon_planner` to match your workspace:

```yaml
descriptors_file: '<workspace>/src/uneven_terrain/mesh_navigation/epsilon_mesh_planner/data/vertex_descriptors.npy'
mesh_file:        '<workspace>/src/uneven_terrain/mesh_navigation/epsilon_mesh_planner/data/mesh_export.obj'
gp_params_file:   '<workspace>/src/uneven_terrain/mesh_navigation/epsilon_mesh_planner/data/gp_matrices.npz'
curvature_file:   '<workspace>/src/uneven_terrain/mesh_navigation/epsilon_mesh_planner/data/curvatures.npz'
```

The `gakd_planner` and `mppi_planner` sections have the same kind of entries if you want to run the baselines.

## Running

```bash
source ~/mesh_ws/install/setup.bash
ros2 launch terrain_nav terrain_nav_launch.py world_name:=desert
```

Launch arguments:

| Argument | Values | Default |
|---|---|---|
| `world_name` | `desert`, `inno` | |
| `map_name` | any map in `terrain_nav/maps` | same as `world_name` |
| `localization_type` | `ground_truth`, `rmcl_micpl` | `ground_truth` |
| `start_rviz` | `True`, `False` | `True` |

Set a goal pose in RViz to start planning.

### Switching planners

The active planner is chosen in `mbf_mesh_nav.yaml`:

```yaml
planners: ['epsilon_planner']   # or 'gakd_planner', 'mppi_planner'
```

### Main parameters of the ε-constraint planner

| Parameter | Default | Meaning |
|---|---|---|
| `num_candidates` | 80 | Candidate control sequences per cycle |
| `time_horizon` | 10 | Horizon length in steps |
| `control_dt` | 0.1 | Step length in seconds |
| `epsilon_base` | 0.5 | Stability budget before terrain adaptation |
| `roughness_adaptation_strength` | 0.5 | How strongly roughness tightens ε (0 to 1) |
| `slope_adaptation_strength` | 0.3 | How strongly slope tightens ε (0 to 1) |
| `alpha_safe` | 0.5236 | Largest allowed deviation from the terrain normal, in radians (30°) |
| `fmm_bias_strength` | 0.7 | Sampling bias toward the guidance field (0 is random, 1 follows the field) |
| `kappa` | 10.0 | Penalty coefficient for stability violations |
| `use_simplified_dynamics` | false | If true, skips the terrain model and GP correction |

To record rollouts for retraining the Gaussian process on a new robot, set `enable_data_collection: true`. Data is written to `~/robot_dataset.csv` unless `data_collection_filepath` is set.

## Results

Real-world runs reported in the paper:

| Robot | Method | Success | Max. orientation deviation | Planning time |
|---|---|---|---|---|
| AgileX Scout Mini | MPPI | 67% | 32.8° | 42.3 ms |
| | GAKD | 73% | 30.2° | 51.6 ms |
| | **ε-MPC** | **80%** | **26.4°** | 48.7 ms |
| JetAuto | MPPI | 58% | 35.7° | 44.1 ms |
| | GAKD | 66% | 32.5° | 53.2 ms |
| | **ε-MPC** | **74%** | **28.6°** | 50.4 ms |

In simulation the planner reaches a 94% success rate. See the paper for the full evaluation, ablations and failure analysis.

## Limitations

The planner works from the static terrain mesh and does not avoid dynamic obstacles. Lethal regions of the mesh are handled through the mesh-map cost layers configured in `mbf_mesh_nav.yaml`.

## Citation

```bibtex
@article{jerome2026epsilon,
  author  = {Jerome, Otobong and Kulathunga, Geesara and Nascimento, Tiago},
  title   = {A Terrain-Adaptive $\epsilon$-Constraint {MPC} for Uneven Terrain Kinodynamic Planning},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}
```

The GAKD baseline is described in:

```bibtex
@article{jerome2025gakd,
  author  = {Jerome, Otobong and Klimchik, Alexandr and Maloletov, Alexander and Kulathunga, Geesara},
  title   = {A Genetic Approach to Gradient-Free Kinodynamic Planning in Uneven Terrains},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {10},
  number  = {6},
  pages   = {5521--5528},
  year    = {2025},
  doi     = {10.1109/LRA.2025.3560883}
}
```

Its original demo is at [Stblacq/mesh_planner](https://github.com/Stblacq/mesh_planner).

## License

The planner packages are released under the BSD 3-Clause licence. The included third-party packages keep their own licences.

## Contributing

Questions and suggestions are welcome through the issue tracker.
