#include "nsga2_mesh_planner/nsga2_mesh_planner.h"
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/util/Meap.hpp>
#include <mesh_map/util.h>
#include <pluginlib/class_list_macros.hpp>
#include <cnpy.h>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <iostream>

PLUGINLIB_EXPORT_CLASS(nsga2_mesh_planner::NSGA2MeshPlanner, mbf_mesh_core::MeshPlanner);

namespace nsga2_mesh_planner
{

NSGA2MeshPlanner::NSGA2MeshPlanner() : cancel_planning_(false)
{
}

NSGA2MeshPlanner::~NSGA2MeshPlanner()
{
}

bool NSGA2MeshPlanner::initialize(const std::string& plugin_name, 
                                 const std::shared_ptr<mesh_map::MeshMap>& mesh_map_ptr,
                                 const rclcpp::Node::SharedPtr& node)
{
  mesh_map_ = mesh_map_ptr;
  name_ = plugin_name;
  map_frame_ = mesh_map_->mapFrame();
  node_ = node;

  path_pub_ = node->create_publisher<nav_msgs::msg::Path>("~/path", rclcpp::QoS(1).transient_local());
  
  publish_vector_field_ = node->declare_parameter(name_ + ".publish_vector_field", false);
  publish_face_vectors_ = node->declare_parameter(name_ + ".publish_face_vectors", false);
  
  // Data collection
  node->declare_parameter("enable_data_collection", false);
  node->declare_parameter("data_collection_filepath", "");
  
  bool enable_collection = node->get_parameter("enable_data_collection").as_bool();
  std::string filepath = node->get_parameter("data_collection_filepath").as_string();
  
  data_collector_ = std::make_shared<DataCollector>(filepath, node->get_logger());
  data_collector_->setEnabled(enable_collection);
  
  if (enable_collection) {
    RCLCPP_INFO(node->get_logger(), "Data collection enabled. Saving to: %s", 
                data_collector_->getFilePath().c_str());
  }

  // Wavefront configuration
  WaveFrontConfig wf_config;
  wf_config.cost_limit = node->declare_parameter(plugin_name + ".cost_limit", 1.0);
  wf_config.goal_dist_offset = node->declare_parameter(plugin_name + ".goal_dist_offset", 0.3);
  
  std::string update_method = node->declare_parameter(name_ + ".wavefront_update_method", "with_s");

  if (update_method == "with_s") {
    wf_config.update_method = WaveFrontConfig::UpdateMethod::WITH_S;
  } else if (update_method == "fmm") {
    wf_config.update_method = WaveFrontConfig::UpdateMethod::FMM;
  } else {
    wf_config.update_method = WaveFrontConfig::UpdateMethod::STANDARD;
  }

  wavefront_ = std::make_shared<WaveFront>(mesh_map_, wf_config, node->get_logger());

  // Terrain dynamics
  use_simplified_dynamics_ = node->declare_parameter(name_ + ".use_simplified_dynamics", false);
  
  std::string mesh_file = node->declare_parameter(name_ + ".mesh_file",
      "/home/otto/workspaces/RL_ws/src/mesh_navigation/nsga2_mesh_planner/data/mesh_export.obj");
  std::string corrections_file = node->declare_parameter(name_ + ".corrections_file",
      "/home/otto/workspaces/RL_ws/src/mesh_navigation/nsga2_mesh_planner/data/vertex_corrections.npy");
  
  double dt = node->declare_parameter(name_ + ".dynamics_dt", 0.1);
  double epsilon = node->declare_parameter(name_ + ".dynamics_epsilon", 0.5);
  
  if (!use_simplified_dynamics_) {
    try {
      RCLCPP_INFO(node->get_logger(), "Initializing TerrainDynamics...");
      terrain_dynamics_ = std::make_shared<nsga2_mesh_planner::TerrainDynamics>(
          mesh_file, corrections_file, dt, epsilon);
      RCLCPP_INFO(node->get_logger(), "TerrainDynamics initialized successfully");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node->get_logger(), "Failed to initialize TerrainDynamics: %s", e.what());
      RCLCPP_WARN(node->get_logger(), "Falling back to simplified dynamics model");
      terrain_dynamics_ = nullptr;
      use_simplified_dynamics_ = true;
    }
  } else {
    RCLCPP_INFO(node->get_logger(), "Using simplified 2D dynamics model");
    terrain_dynamics_ = nullptr;
  }
  
  // Control constraints
  min_velocity_ = node->declare_parameter(name_ + ".min_velocity", -1.5);
  max_velocity_ = node->declare_parameter(name_ + ".max_velocity", 1.5);
  min_omega_ = node->declare_parameter(name_ + ".min_omega", -1.21832);
  max_omega_ = node->declare_parameter(name_ + ".max_omega", 1.2832);
  
  // NSGA-II parameters
  population_size_ = node->declare_parameter(name_ + ".population_size", 100);
  num_generations_ = node->declare_parameter(name_ + ".num_generations", 50);
  time_horizon_ = node->declare_parameter(name_ + ".time_horizon", 10);
  mutation_rate_ = node->declare_parameter(name_ + ".mutation_rate", 0.02);
  tournament_size_ = node->declare_parameter(name_ + ".tournament_size", 3);
  mutation_stddev_ = node->declare_parameter(name_ + ".mutation_stddev", 0.1);
  fmm_bias_strength_ = node->declare_parameter(name_ + ".fmm_bias_strength", 0.3);
  
  // Stability parameters (for F2)
  alpha_safe_ = node->declare_parameter(name_ + ".alpha_safe", 0.349066);  // 20 degrees
  alpha_max_ = node->declare_parameter(name_ + ".alpha_max", 0.523599);    // 30 degrees
  kappa_ = node->declare_parameter(name_ + ".kappa", 10.0);
  s_exp_ = node->declare_parameter(name_ + ".s_exp", 2.0);
  
  // Execution parameters
  control_dt_ = node->declare_parameter(name_ + ".control_dt", 0.1);
  max_steps_ = node->declare_parameter(name_ + ".max_steps", 1000);
  goal_threshold_ = node->declare_parameter(name_ + ".goal_threshold", 0.5);
  position_projection_radius_ = node->declare_parameter(name_ + ".position_projection_radius", 0.4f);

  // Load precomputed vertex descriptors (REQUIRED)
  std::string descriptors_file = node->declare_parameter(name_ + ".descriptors_file",
      "/home/otto/workspaces/RL_ws/src/mesh_navigation/nsga2_mesh_planner/data/vertex_descriptors.npy");
  
  bool descriptors_loaded = false;
  
  // TerrainDynamics always has descriptors built-in, we just need separate array for fallback
  if (terrain_dynamics_) {
    RCLCPP_INFO(node->get_logger(), "Using descriptors from TerrainDynamics");
    descriptors_loaded = true;
  }
  
  // Always try to load vertex_descriptors as fallback
  {
    try {
      RCLCPP_INFO(node->get_logger(), "Loading vertex descriptors from: %s", descriptors_file.c_str());
      cnpy::NpyArray arr = cnpy::npy_load(descriptors_file);
      
      if (arr.shape.size() != 2 || arr.shape[1] != 8) {
        throw std::runtime_error("Invalid descriptors file shape. Expected (N, 8)");
      }
      
      size_t n_vertices = arr.shape[0];
      double* data = arr.data<double>();
      vertex_descriptors_.resize(n_vertices);
      
      for (size_t i = 0; i < n_vertices; ++i) {
        for (int j = 0; j < 8; ++j) {
          vertex_descriptors_[i][j] = data[i * 8 + j];
        }
      }
      
      RCLCPP_INFO(node->get_logger(), "Loaded %zu vertex descriptors", n_vertices);
      if (!terrain_dynamics_) {
        descriptors_loaded = true;
      }
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node->get_logger(), "Failed to load vertex descriptors: %s", e.what());
      if (!terrain_dynamics_) {
        RCLCPP_ERROR(node->get_logger(), "No fallback available - terrain_dynamics is also unavailable");
      }
    }
  }
  
  if (!descriptors_loaded) {
    RCLCPP_ERROR(node->get_logger(), "NSGA-II planner REQUIRES terrain descriptors!");
    RCLCPP_ERROR(node->get_logger(), "Please provide valid descriptors_file parameter.");
    return false;
  }

  return true;
}

bool NSGA2MeshPlanner::cancel()
{
  if (wavefront_) {
    wavefront_->cancel();
  }
  cancel_planning_ = true;
  return true;
}

State NSGA2MeshPlanner::dynamics(const State& x, const Control& u, double dt)
{
  if (terrain_dynamics_) {
    auto next = terrain_dynamics_->dynamics(x, u);
    return next;
  }
  
  // Fallback to simple dynamics
  double x_pos = x[0];
  double y_pos = x[1];
  double z_pos = x[2];
  double phi = x[3];
  double theta = x[4];
  double psi = x[5];

  double u0 = u[0];
  double u1 = u[1];

  double x_next = x_pos + u0 * std::cos(psi) * dt;
  double y_next = y_pos + u0 * std::sin(psi) * dt;
  double psi_next = psi + u1 * dt;

  return {x_next, y_next, z_pos, phi, theta, psi_next};
}

Control NSGA2MeshPlanner::randomControl()
{
  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dis_velocity(min_velocity_, max_velocity_);
  std::uniform_real_distribution<> dis_omega(min_omega_, max_omega_);
  return {dis_velocity(gen), dis_omega(gen)};
}

Control NSGA2MeshPlanner::fmmGuidedControl(
    const mesh_map::Vector& position,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  auto dir_opt = calculateDirectionAtPosition(position, vector_map);
  
  if (dir_opt) {
    mesh_map::Vector dir = dir_opt.get().normalized();
    
    // Velocity magnitude biased toward forward motion
    static thread_local std::mt19937 gen(std::random_device{}());
    std::normal_distribution<> vel_dist(0.7 * max_velocity_, 0.2);
    double v = std::clamp(vel_dist(gen), min_velocity_, max_velocity_);
    
    // Angular velocity based on desired direction
    double desired_yaw = std::atan2(dir.y, dir.x);
    std::normal_distribution<> omega_dist(0.0, 0.3);
    double omega = std::clamp(omega_dist(gen), min_omega_, max_omega_);
    
    return {v, omega};
  }
  
  return randomControl();
}

boost::optional<mesh_map::Vector> NSGA2MeshPlanner::calculateDirectionAtPosition(
    const mesh_map::Vector& position,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  const auto& mesh = mesh_map_->mesh();
  mesh_map::Vector pos_copy = position;

  auto goal_face_opt = mesh_map_->getContainingFace(pos_copy, position_projection_radius_);
  if (!goal_face_opt) {
    return boost::none;
  }

  const auto goal_face = goal_face_opt.unwrap();
  const auto& vertex_handles = mesh->getVerticesOfFace(goal_face);

  std::array<float, 3> barycentric_coords;
  float dist;
  mesh_map_->projectedBarycentricCoords(position, goal_face, barycentric_coords, dist);

  boost::optional<mesh_map::Vector> direction =
      mesh_map_->directionAtPosition(vector_map, vertex_handles, barycentric_coords);

  return direction;
}

float NSGA2MeshPlanner::computeDirectionCost(
    const mesh_map::Vector& current,
    const mesh_map::Vector& next,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  mesh_map::Vector move_dir = (next - current).normalized();
  auto ideal_dir_opt = calculateDirectionAtPosition(current, vector_map);
  
  if (!ideal_dir_opt) {
    return 1.0f;
  }
  
  mesh_map::Vector ideal_dir = ideal_dir_opt.get();
  float dot = move_dir.dot(ideal_dir);
  return 1.0f - std::max(0.0f, dot);
}

TerrainDescriptor NSGA2MeshPlanner::getTerrainDescriptor(const mesh_map::Vector& position)
{
  TerrainDescriptor descriptor;

  // Use TerrainDynamics KD-tree lookup (fast O(log n))
  if (terrain_dynamics_) {
    auto desc = terrain_dynamics_->getDescriptorAtPosition(position.x, position.y, position.z);
    
    descriptor.n_x = desc[0];
    descriptor.n_y = desc[1];
    descriptor.n_z = desc[2];
    descriptor.roughness = desc[3];
    descriptor.gaussian_curv = desc[4];
    descriptor.mean_curv = desc[5];
    descriptor.total_area = desc[6];
    descriptor.inclination = std::acos(std::clamp(descriptor.n_z, -1.0, 1.0));
    
    return descriptor;
  }

  // Use precomputed descriptors with brute-force search (slower O(n))
  if (!vertex_descriptors_.empty()) {
    const auto& mesh = mesh_map_->mesh();
    
    size_t closest_vertex_idx = 0;
    double min_dist_sq = std::numeric_limits<double>::max();
    
    for (const auto& vh : mesh->vertices()) {
      auto vertex_pos = mesh->getVertexPosition(vh);
      size_t idx = vh.idx();
      if (idx < vertex_descriptors_.size()) {
        double dx = vertex_pos.x - position.x;
        double dy = vertex_pos.y - position.y;
        double dz = vertex_pos.z - position.z;
        double dist_sq = dx*dx + dy*dy + dz*dz;
        
        if (dist_sq < min_dist_sq) {
          min_dist_sq = dist_sq;
          closest_vertex_idx = idx;
        }
      }
    }
    
    const auto& desc = vertex_descriptors_[closest_vertex_idx];
    descriptor.n_x = desc[0];
    descriptor.n_y = desc[1];
    descriptor.n_z = desc[2];
    descriptor.roughness = desc[3];
    descriptor.gaussian_curv = desc[4];
    descriptor.mean_curv = desc[5];
    descriptor.total_area = desc[6];
    descriptor.inclination = std::acos(std::clamp(descriptor.n_z, -1.0, 1.0));
    
    return descriptor;
  }

  // No descriptors available - this planner cannot function
  throw std::runtime_error("NSGA-II planner requires terrain descriptors - none available!");
}

std::pair<double, double> NSGA2MeshPlanner::computeTerrainAlignedOrientation(
    const TerrainDescriptor& descriptor)
{
  mesh_map::Vector normal = descriptor.getNormal();
  
  // Terrain-aligned roll: rotation about x-axis to align with terrain
  double theta_terrain = std::atan2(normal.y, normal.z);
  
  // Terrain-aligned pitch: rotation about y-axis to align with terrain
  double phi_terrain = std::atan2(-normal.x, std::sqrt(normal.y * normal.y + normal.z * normal.z));
  
  return {theta_terrain, phi_terrain};
}

double NSGA2MeshPlanner::stabilityViolation(double delta_angle)
{
  if (std::abs(delta_angle) <= alpha_safe_) {
    return 0.0;
  }
  
  double excess = std::abs(delta_angle) - alpha_safe_;
  return excess;
}

Trajectory NSGA2MeshPlanner::simulateForward(
    const State& state_init,
    const ControlSequence& control_sequence)
{
  Trajectory trajectory;
  State current_state = state_init;
  trajectory.push_back(current_state);
  
  for (const auto& control : control_sequence) {
    current_state = dynamics(current_state, control, control_dt_);
    trajectory.push_back(current_state);
  }
  
  return trajectory;
}

std::vector<TerrainDescriptor> NSGA2MeshPlanner::extractTerrainDescriptors(
    const Trajectory& trajectory)
{
  std::vector<TerrainDescriptor> descriptors;
  descriptors.reserve(trajectory.size());
  
  for (const auto& state : trajectory) {
    mesh_map::Vector position(state[0], state[1], state[2]);
    descriptors.push_back(getTerrainDescriptor(position));
  }
  
  return descriptors;
}

double NSGA2MeshPlanner::objectiveF1(
    const Trajectory& trajectory,
    const State& state_target,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  double cost = 0.0;
  
  // Path length and direction cost
  for (size_t i = 1; i < trajectory.size(); ++i) {
    mesh_map::Vector current(trajectory[i-1][0], trajectory[i-1][1], trajectory[i-1][2]);
    mesh_map::Vector next(trajectory[i][0], trajectory[i][1], trajectory[i][2]);
    
    double dist = (next - current).length();
    double dir_cost = computeDirectionCost(current, next, vector_map);
    
    cost += dist * (1.0 + 0.5 * dir_cost);  // Penalize deviations from FMM
  }
  
  // Terminal cost: distance to goal
  const State& final_state = trajectory.back();
  double terminal_dist = std::sqrt(
      std::pow(final_state[0] - state_target[0], 2) +
      std::pow(final_state[1] - state_target[1], 2) +
      std::pow(final_state[2] - state_target[2], 2));
  
  cost += 5.0 * terminal_dist;  // Heavy penalty for not reaching goal
  
  return cost;
}

double NSGA2MeshPlanner::objectiveF2(
    const Trajectory& trajectory,
    const std::vector<TerrainDescriptor>& descriptors)
{
  double total_violation = 0.0;
  
  // Distance to goal (for proximity scaling)
  const State& final_state = trajectory.back();
  
  for (size_t i = 0; i < trajectory.size(); ++i) {
    const State& state = trajectory[i];
    const TerrainDescriptor& desc = descriptors[i];
    
    // Compute terrain-aligned orientation
    auto [theta_terrain, phi_terrain] = computeTerrainAlignedOrientation(desc);
    
    // Compute deviations
    double delta_roll = state[3] - theta_terrain;
    double delta_pitch = state[4] - phi_terrain;
    
    // Stability violations
    double violation_roll = stabilityViolation(delta_roll);
    double violation_pitch = stabilityViolation(delta_pitch);
    
    total_violation += (violation_roll + violation_pitch);
  }
  
  // Apply penalty coefficient
  return kappa_ * total_violation;
}

bool NSGA2MeshPlanner::checkHardConstraints(
    const Trajectory& trajectory,
    const std::vector<TerrainDescriptor>& descriptors)
{
  for (size_t i = 0; i < trajectory.size(); ++i) {
    const State& state = trajectory[i];
    const TerrainDescriptor& desc = descriptors[i];
    
    // Compute terrain-aligned orientation
    auto [theta_terrain, phi_terrain] = computeTerrainAlignedOrientation(desc);
    
    // Compute deviations
    double delta_roll = std::abs(state[3] - theta_terrain);
    double delta_pitch = std::abs(state[4] - phi_terrain);
    
    // Check hard constraint: |Δα| < α_max
    if (delta_roll >= alpha_max_ || delta_pitch >= alpha_max_) {
      return false;  // Violates hard constraint
    }
  }
  
  return true;  // No violations
}

void NSGA2MeshPlanner::evaluateIndividual(
    Individual& ind,
    const State& state_init,
    const State& state_target,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  // Simulate trajectory
  ind.trajectory = simulateForward(state_init, ind.controls);
  
  // Extract terrain descriptors
  ind.descriptors = extractTerrainDescriptors(ind.trajectory);
  
  // Check hard constraints
  ind.violates_hard_constraints = !checkHardConstraints(ind.trajectory, ind.descriptors);
  
  if (ind.violates_hard_constraints) {
    // Assign infinite penalty
    ind.f1 = std::numeric_limits<double>::infinity();
    ind.f2 = std::numeric_limits<double>::infinity();
  } else {
    // Evaluate objectives
    ind.f1 = objectiveF1(ind.trajectory, state_target, vector_map);
    ind.f2 = objectiveF2(ind.trajectory, ind.descriptors);
  }
}

bool NSGA2MeshPlanner::dominates(const Individual& a, const Individual& b)
{
  // Handle constraint violations
  if (a.violates_hard_constraints && !b.violates_hard_constraints) {
    return false;  // b dominates a
  }
  if (!a.violates_hard_constraints && b.violates_hard_constraints) {
    return true;   // a dominates b
  }
  if (a.violates_hard_constraints && b.violates_hard_constraints) {
    return false;  // Neither dominates if both violate
  }
  
  // Standard Pareto dominance for feasible solutions
  bool at_least_as_good = (a.f1 <= b.f1) && (a.f2 <= b.f2);
  bool strictly_better = (a.f1 < b.f1) || (a.f2 < b.f2);
  
  return at_least_as_good && strictly_better;
}

void NSGA2MeshPlanner::fastNonDominatedSort(std::vector<Individual>& population)
{
  int n = population.size();
  
  // Reset NSGA-II attributes
  for (auto& ind : population) {
    ind.domination_count = 0;
    ind.dominated_solutions.clear();
    ind.rank = 0;
  }
  
  std::vector<std::vector<int>> fronts;
  std::vector<int> first_front;
  
  // Compute domination relationships
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      if (p == q) continue;
      
      if (dominates(population[p], population[q])) {
        population[p].dominated_solutions.push_back(q);
      } else if (dominates(population[q], population[p])) {
        population[p].domination_count++;
      }
    }
    
    if (population[p].domination_count == 0) {
      population[p].rank = 0;
      first_front.push_back(p);
    }
  }
  
  fronts.push_back(first_front);
  
  // Build subsequent fronts
  int current_front = 0;
  while (!fronts[current_front].empty()) {
    std::vector<int> next_front;
    
    for (int p : fronts[current_front]) {
      for (int q : population[p].dominated_solutions) {
        population[q].domination_count--;
        if (population[q].domination_count == 0) {
          population[q].rank = current_front + 1;
          next_front.push_back(q);
        }
      }
    }
    
    if (!next_front.empty()) {
      fronts.push_back(next_front);
      current_front++;
    } else {
      break;
    }
  }
}

void NSGA2MeshPlanner::calculateCrowdingDistance(std::vector<Individual>& front)
{
  int n = front.size();
  if (n == 0) return;
  
  // Initialize crowding distances
  for (auto& ind : front) {
    ind.crowding_distance = 0.0;
  }
  
  // For each objective
  for (int obj = 0; obj < 2; ++obj) {
    // Sort by objective
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::sort(indices.begin(), indices.end(), [&](int i, int j) {
      double val_i = (obj == 0) ? front[i].f1 : front[i].f2;
      double val_j = (obj == 0) ? front[j].f1 : front[j].f2;
      return val_i < val_j;
    });
    
    // Boundary points get infinite distance
    front[indices[0]].crowding_distance = std::numeric_limits<double>::infinity();
    front[indices[n-1]].crowding_distance = std::numeric_limits<double>::infinity();
    
    // Compute range
    double f_min = (obj == 0) ? front[indices[0]].f1 : front[indices[0]].f2;
    double f_max = (obj == 0) ? front[indices[n-1]].f1 : front[indices[n-1]].f2;
    double range = f_max - f_min;
    
    if (range < 1e-10) range = 1.0;  // Avoid division by zero
    
    // Calculate crowding distance
    for (int i = 1; i < n - 1; ++i) {
      double f_next = (obj == 0) ? front[indices[i+1]].f1 : front[indices[i+1]].f2;
      double f_prev = (obj == 0) ? front[indices[i-1]].f1 : front[indices[i-1]].f2;
      
      front[indices[i]].crowding_distance += (f_next - f_prev) / range;
    }
  }
}

std::vector<Individual> NSGA2MeshPlanner::tournamentSelection(
    const std::vector<Individual>& population,
    int tournament_size,
    int num_parents)
{
  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<> dis(0, population.size() - 1);
  
  std::vector<Individual> selected;
  selected.reserve(num_parents);
  
  for (int i = 0; i < num_parents; ++i) {
    std::vector<int> tournament_indices;
    for (int j = 0; j < tournament_size; ++j) {
      tournament_indices.push_back(dis(gen));
    }
    
    // Select best from tournament (lower rank, or higher crowding distance if same rank)
    int best_idx = tournament_indices[0];
    for (int idx : tournament_indices) {
      if (population[idx].rank < population[best_idx].rank) {
        best_idx = idx;
      } else if (population[idx].rank == population[best_idx].rank &&
                 population[idx].crowding_distance > population[best_idx].crowding_distance) {
        best_idx = idx;
      }
    }
    
    selected.push_back(population[best_idx]);
  }
  
  return selected;
}

std::pair<ControlSequence, ControlSequence> NSGA2MeshPlanner::uniformCrossover(
    const ControlSequence& parent1,
    const ControlSequence& parent2)
{
  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  
  ControlSequence child1 = parent1;
  ControlSequence child2 = parent2;
  
  for (size_t i = 0; i < parent1.size(); ++i) {
    // For each control dimension
    for (int d = 0; d < 2; ++d) {
      if (dis(gen) < 0.5) {
        std::swap(child1[i][d], child2[i][d]);
      }
    }
  }
  
  return {child1, child2};
}

ControlSequence NSGA2MeshPlanner::gaussianMutation(
    const ControlSequence& controls,
    double mutation_rate,
    double mutation_stddev)
{
  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_real_distribution<> prob_dis(0.0, 1.0);
  std::normal_distribution<> mut_dis(0.0, mutation_stddev);
  
  ControlSequence mutated = controls;
  
  for (auto& control : mutated) {
    if (prob_dis(gen) < mutation_rate) {
      control[0] = std::clamp(control[0] + mut_dis(gen), min_velocity_, max_velocity_);
      control[1] = std::clamp(control[1] + mut_dis(gen), min_omega_, max_omega_);
    }
  }
  
  return mutated;
}

std::vector<Individual> NSGA2MeshPlanner::initializePopulation(
    const State& current_state,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map,
    const std::optional<ControlSequence>& prev_best_controls)
{
  std::vector<Individual> population;
  population.reserve(population_size_);
  
  int num_fmm_guided = static_cast<int>(fmm_bias_strength_ * population_size_);
  int num_random = population_size_ - num_fmm_guided;
  
  // Add warm-started individual if available
  if (prev_best_controls) {
    Individual warm_start;
    warm_start.controls = prev_best_controls.value();
    
    // Temporal shifting: remove first control, add zero control at end
    if (!warm_start.controls.empty()) {
      warm_start.controls.erase(warm_start.controls.begin());
      warm_start.controls.push_back({0.0, 0.0});
    }
    
    population.push_back(warm_start);
    num_random--;
  }
  
  // FMM-guided initialization
  for (int i = 0; i < num_fmm_guided; ++i) {
    Individual ind;
    ind.controls.reserve(time_horizon_);
    
    State state = current_state;
    for (int t = 0; t < time_horizon_; ++t) {
      mesh_map::Vector pos(state[0], state[1], state[2]);
      Control control = fmmGuidedControl(pos, vector_map);
      ind.controls.push_back(control);
      
      // Simulate one step for next position
      state = dynamics(state, control, control_dt_);
    }
    
    population.push_back(ind);
  }
  
  // Random initialization
  for (int i = 0; i < num_random; ++i) {
    Individual ind;
    ind.controls.reserve(time_horizon_);
    
    for (int t = 0; t < time_horizon_; ++t) {
      ind.controls.push_back(randomControl());
    }
    
    population.push_back(ind);
  }
  
  return population;
}

int NSGA2MeshPlanner::selectSolutionFromParetoFront(
    const std::vector<Individual>& pareto_front,
    const State& state_target)
{
  if (pareto_front.empty()) {
    return -1;
  }
  
  if (pareto_front.size() == 1) {
    return 0;
  }
  
  // Compute terrain-adaptive weights for each solution
  std::vector<double> alpha_path(pareto_front.size());
  std::vector<double> alpha_stab(pareto_front.size());
  
  for (size_t j = 0; j < pareto_front.size(); ++j) {
    double min_weight = 1.0;
    
    for (const auto& desc : pareto_front[j].descriptors) {
      double weight = 1.0 - (desc.inclination / alpha_max_);
      min_weight = std::min(min_weight, weight);
    }
    
    alpha_path[j] = min_weight;
    alpha_stab[j] = 1.0 - min_weight;
  }
  
  // Normalize objectives across Pareto front
  double f1_min = std::numeric_limits<double>::max();
  double f1_max = std::numeric_limits<double>::lowest();
  double f2_min = std::numeric_limits<double>::max();
  double f2_max = std::numeric_limits<double>::lowest();
  
  for (const auto& ind : pareto_front) {
    f1_min = std::min(f1_min, ind.f1);
    f1_max = std::max(f1_max, ind.f1);
    f2_min = std::min(f2_min, ind.f2);
    f2_max = std::max(f2_max, ind.f2);
  }
  
  double f1_range = (f1_max - f1_min) > 1e-6 ? (f1_max - f1_min) : 1.0;
  double f2_range = (f2_max - f2_min) > 1e-6 ? (f2_max - f2_min) : 1.0;
  
  // Select solution with minimum weighted sum
  int best_idx = 0;
  double best_score = std::numeric_limits<double>::max();
  
  for (size_t j = 0; j < pareto_front.size(); ++j) {
    double f1_norm = (pareto_front[j].f1 - f1_min) / f1_range;
    double f2_norm = (pareto_front[j].f2 - f2_min) / f2_range;
    
    double score = alpha_path[j] * f1_norm + alpha_stab[j] * f2_norm;
    
    if (score < best_score) {
      best_score = score;
      best_idx = j;
    }
  }
  
  return best_idx;
}

std::vector<Individual> NSGA2MeshPlanner::nsga2Optimization(
    const State& state_init,
    const State& state_target,
    const std::optional<ControlSequence>& prev_best_controls,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  // Initialize population
  std::vector<Individual> population = initializePopulation(
      state_init, vector_map, prev_best_controls);
  
  // Evaluate initial population
  for (auto& ind : population) {
    evaluateIndividual(ind, state_init, state_target, vector_map);
  }
  
  // Main NSGA-II loop
  for (int gen = 0; gen < num_generations_; ++gen) {
    if (cancel_planning_) break;
    
    // Non-dominated sorting
    fastNonDominatedSort(population);
    
    // Calculate crowding distance for each front
    std::map<int, std::vector<Individual*>> fronts;
    for (auto& ind : population) {
      fronts[ind.rank].push_back(&ind);
    }
    
    for (auto& [rank, front_ptrs] : fronts) {
      std::vector<Individual> front;
      for (auto* ptr : front_ptrs) {
        front.push_back(*ptr);
      }
      calculateCrowdingDistance(front);
      
      // Copy back crowding distances
      for (size_t i = 0; i < front.size(); ++i) {
        front_ptrs[i]->crowding_distance = front[i].crowding_distance;
      }
    }
    
    // Generate offspring
    std::vector<Individual> offspring;
    offspring.reserve(population_size_);
    
    while (offspring.size() < population_size_) {
      // Select parents
      auto parents = tournamentSelection(population, tournament_size_, 2);
      
      // Crossover
      auto [child1_controls, child2_controls] = uniformCrossover(
          parents[0].controls, parents[1].controls);
      
      // Mutation
      child1_controls = gaussianMutation(child1_controls, mutation_rate_, mutation_stddev_);
      child2_controls = gaussianMutation(child2_controls, mutation_rate_, mutation_stddev_);
      
      // Create offspring individuals
      Individual child1, child2;
      child1.controls = child1_controls;
      child2.controls = child2_controls;
      
      offspring.push_back(child1);
      if (offspring.size() < population_size_) {
        offspring.push_back(child2);
      }
    }
    
    // Evaluate offspring
    for (auto& ind : offspring) {
      evaluateIndividual(ind, state_init, state_target, vector_map);
    }
    
    // Combine parent and offspring populations
    std::vector<Individual> combined = population;
    combined.insert(combined.end(), offspring.begin(), offspring.end());
    
    // Sort combined population
    fastNonDominatedSort(combined);
    
    // Select next generation
    population.clear();
    population.reserve(population_size_);
    
    int current_rank = 0;
    while (population.size() < population_size_) {
      // Get individuals in current front
      std::vector<Individual> current_front;
      for (const auto& ind : combined) {
        if (ind.rank == current_rank) {
          current_front.push_back(ind);
        }
      }
      
      if (current_front.empty()) break;
      
      // If entire front fits, add all
      if (population.size() + current_front.size() <= population_size_) {
        population.insert(population.end(), current_front.begin(), current_front.end());
      } else {
        // Front doesn't fit entirely, select by crowding distance
        calculateCrowdingDistance(current_front);
        std::sort(current_front.begin(), current_front.end(),
                 [](const Individual& a, const Individual& b) {
                   return a.crowding_distance > b.crowding_distance;
                 });
        
        int remaining = population_size_ - population.size();
        population.insert(population.end(), current_front.begin(), 
                         current_front.begin() + remaining);
        break;
      }
      
      current_rank++;
    }
  }
  
  // Extract Pareto front (rank 0)
  std::vector<Individual> pareto_front;
  for (const auto& ind : population) {
    if (ind.rank == 0) {
      pareto_front.push_back(ind);
    }
  }
  
  return pareto_front;
}

Trajectory NSGA2MeshPlanner::nsga2Planner(
    const State& start_point,
    const State& goal_point,
    int time_horizon,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  auto twist_pub = node_->create_publisher<geometry_msgs::msg::TwistStamped>(
      "~/cmd_vel", 10);
  auto point_pub = node_->create_publisher<geometry_msgs::msg::PointStamped>(
      "~/current_point", 10);
  auto ideal_dir_pub = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      "~/ideal_direction", 10);
  
  // Odometry subscription for state updates
  std::mutex odom_mutex;
  bool odometry_received = false;
  nav_msgs::msg::Odometry current_state;
  nav_msgs::msg::Odometry previous_state;
  bool previous_state_valid = false;
  
  auto odom_sub = node_->create_subscription<nav_msgs::msg::Odometry>(
      "/odometry/filtered", 10,
      [&](const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(odom_mutex);
        current_state = *msg;
        odometry_received = true;
      });
  
  double dt = control_dt_;
  double goal_threshold = goal_threshold_;
  Trajectory trajectory;
  State x_current = start_point;
  trajectory.push_back(x_current);
  
  ControlSequence last_control;
  bool control_available = false;
  
  std_msgs::msg::Header header;
  header.frame_id = map_frame_;
  
  for (int step = 0; step < max_steps_ && !cancel_planning_; ++step) {
    // Publish current position
    geometry_msgs::msg::PointStamped point_msg;
    point_msg.header = header;
    point_msg.header.stamp = node_->now();
    point_msg.point.x = x_current[0];
    point_msg.point.y = x_current[1];
    point_msg.point.z = x_current[2];
    point_pub->publish(point_msg);
    
    // Publish ideal FMM direction for visualization
    mesh_map::Vector current_vec(x_current[0], x_current[1], x_current[2]);
    auto ideal_dir_opt = calculateDirectionAtPosition(current_vec, vector_map);
    
    if (ideal_dir_opt) {
      mesh_map::Vector dir = ideal_dir_opt.get();
      geometry_msgs::msg::PoseStamped dir_msg;
      dir_msg.header = header;
      dir_msg.header.stamp = node_->now();
      dir_msg.pose.position.x = x_current[0];
      dir_msg.pose.position.y = x_current[1];
      dir_msg.pose.position.z = x_current[2];
      
      double yaw = std::atan2(dir.y, dir.x);
      tf2::Quaternion q;
      q.setRPY(0, 0, yaw);
      dir_msg.pose.orientation = tf2::toMsg(q);
      
      ideal_dir_pub->publish(dir_msg);
    }
    
    // NSGA-II optimization
    std::vector<Individual> pareto_front = nsga2Optimization(
        x_current, goal_point, prev_best_controls_, vector_map);
    
    if (pareto_front.empty()) {
      RCLCPP_ERROR(node_->get_logger(), "NSGA-II failed to find any feasible solution!");
      break;
    }
    
    // Select solution from Pareto front using terrain-adaptive strategy
    int best_idx = selectSolutionFromParetoFront(pareto_front, goal_point);
    
    if (best_idx < 0) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to select solution from Pareto front!");
      break;
    }
    
    ControlSequence optimal_control = pareto_front[best_idx].controls;
    
    // Store for warm start
    prev_best_controls_ = optimal_control;
    
    // Publish control
    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header = header;
    twist_msg.header.stamp = node_->now();
    twist_msg.twist.linear.x = optimal_control[0][0];
    twist_msg.twist.angular.z = optimal_control[0][1];
    twist_pub->publish(twist_msg);
    
    // Update state
    {
      std::lock_guard<std::mutex> lock(odom_mutex);
      if (odometry_received) {
        if (previous_state_valid && control_available &&
            data_collector_ && data_collector_->isEnabled()) {
          // Convert Odometry to PoseStamped for data collector
          geometry_msgs::msg::PoseStamped prev_pose, curr_pose;
          prev_pose.header = previous_state.header;
          prev_pose.pose = previous_state.pose.pose;
          curr_pose.header = current_state.header;
          curr_pose.pose = current_state.pose.pose;
          
          data_collector_->collectTransition(
              prev_pose, curr_pose,
              last_control[0][0], last_control[0][1]);
        }
        
        previous_state = current_state;
        previous_state_valid = true;
        
        x_current[0] = current_state.pose.pose.position.x;
        x_current[1] = current_state.pose.pose.position.y;
        x_current[2] = current_state.pose.pose.position.z;
        
        tf2::Quaternion quat(
            current_state.pose.pose.orientation.x,
            current_state.pose.pose.orientation.y,
            current_state.pose.pose.orientation.z,
            current_state.pose.pose.orientation.w);
        double roll, pitch, yaw;
        tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        x_current[3] = roll;
        x_current[4] = pitch;
        x_current[5] = yaw;
      } else {
        RCLCPP_WARN(node_->get_logger(),
                   "No odometry at step %d. Using dynamics.", step);
        x_current = dynamics(x_current, optimal_control[0], control_dt_);
      }
    }
    
    // Shift for warm start
    if (prev_best_controls_) {
      prev_best_controls_->erase(prev_best_controls_->begin());
      prev_best_controls_->push_back({0.0, 0.0});
    }
    
    last_control = optimal_control;
    control_available = true;
    
    trajectory.push_back(x_current);
    
    // Check goal
    double dist = std::sqrt(
        std::pow(x_current[0] - goal_point[0], 2) +
        std::pow(x_current[1] - goal_point[1], 2) +
        std::pow(x_current[2] - goal_point[2], 2));
    
    if (dist < goal_threshold_) {
      RCLCPP_INFO(node_->get_logger(), "Goal reached!");
      if (data_collector_ && data_collector_->isEnabled()) {
        data_collector_->flush();
      }
      return trajectory;
    }
  }
  
  RCLCPP_WARN(node_->get_logger(), "Max steps reached without reaching goal.");
  if (data_collector_ && data_collector_->isEnabled()) {
    data_collector_->flush();
  }
  return {};
}

std::tuple<double, double, double> NSGA2MeshPlanner::getRPYFromQuaternion(
    const geometry_msgs::msg::Quaternion& q)
{
  tf2::Quaternion quat(q.x, q.y, q.z, q.w);
  tf2::Matrix3x3 m(quat);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  return std::make_tuple(roll, pitch, yaw);
}

uint32_t NSGA2MeshPlanner::makePlan(
    const geometry_msgs::msg::PoseStamped& start,
    const geometry_msgs::msg::PoseStamped& goal, 
    double tolerance, 
    std::vector<geometry_msgs::msg::PoseStamped>& plan,
    double& cost, 
    std::string& message)
{
  mesh_map::Vector start_vec = mesh_map::toVector(start.pose.position);
  mesh_map::Vector goal_vec = mesh_map::toVector(goal.pose.position);

  if (!wavefront_->compute(start_vec, goal_vec, message)) {
    RCLCPP_ERROR_STREAM(node_->get_logger(),
                       "Wavefront computation failed: " << message);
    return mbf_msgs::action::GetPath::Result::NO_PATH_FOUND;
  }
  
  const auto& vector_map = wavefront_->getVectorMap();

  std_msgs::msg::Header header;
  header.stamp = node_->now();
  header.frame_id = map_frame_;
  
  if (publish_vector_field_) {
    mesh_map_->publishVectorField("vector_field", vector_map, publish_face_vectors_);
  }

  double roll_start, pitch_start, yaw_start;
  std::tie(roll_start, pitch_start, yaw_start) = 
      getRPYFromQuaternion(start.pose.orientation);
  State start_state = {start.pose.position.x, start.pose.position.y,
                      start.pose.position.z, roll_start, pitch_start, yaw_start};

  double roll_goal, pitch_goal, yaw_goal;
  std::tie(roll_goal, pitch_goal, yaw_goal) = 
      getRPYFromQuaternion(goal.pose.orientation);
  State goal_state = {goal.pose.position.x, goal.pose.position.y,
                     goal.pose.position.z, roll_goal, pitch_goal, yaw_goal};

  Trajectory traj = nsga2Planner(start_state, goal_state, time_horizon_, vector_map);

  if (cancel_planning_) {
    message = "Planning was canceled";
    RCLCPP_WARN_STREAM(node_->get_logger(), message);
    return mbf_msgs::action::GetPath::Result::CANCELED;
  }

  if (traj.empty()) {
    message = "No path found to the goal";
    RCLCPP_WARN_STREAM(node_->get_logger(), message);
    return mbf_msgs::action::GetPath::Result::NO_PATH_FOUND;
  }

  RCLCPP_INFO_STREAM(node_->get_logger(),
                    "Path found with " << traj.size() << " points");

  header.stamp = node_->now();
  header.frame_id = map_frame_;
  cost = 0.0;
  mesh_map::Vector current = start_vec;
  nav_msgs::msg::Path path_msg;
  path_msg.header = header;

  for (size_t i = 0; i < traj.size(); ++i) {
    geometry_msgs::msg::PoseStamped pose;
    pose.header = header;
    const auto& next_state = traj[i];

    geometry_msgs::msg::Point next_point;
    next_point.x = next_state[0];
    next_point.y = next_state[1];
    next_point.z = next_state[2];
    mesh_map::Vector next = mesh_map::toVector(next_point);

    pose.pose.position = next_point;
    pose.pose.orientation = start.pose.orientation;
    float dir_length = (next - current).length();

    cost += dir_length;
    current = next;
    plan.push_back(pose);
    path_msg.poses.push_back(pose);
  }

  geometry_msgs::msg::PoseStamped goal_pose;
  goal_pose.header = header;
  goal_pose.pose = goal.pose;
  plan.push_back(goal_pose);
  path_msg.poses.push_back(goal_pose);

  path_pub_->publish(path_msg);

  RCLCPP_INFO_STREAM(node_->get_logger(), "Path length: " << cost << "m");
  return mbf_msgs::action::GetPath::Result::SUCCESS;
}

}  // namespace nsga2_mesh_planner