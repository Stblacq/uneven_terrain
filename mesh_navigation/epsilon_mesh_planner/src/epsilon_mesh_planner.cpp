#include "epsilon_mesh_planner/epsilon_mesh_planner.h"
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
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <iostream>

PLUGINLIB_EXPORT_CLASS(epsilon_mesh_planner::EpsilonMeshPlanner, mbf_mesh_core::MeshPlanner);

namespace epsilon_mesh_planner
{

EpsilonMeshPlanner::EpsilonMeshPlanner() : cancel_planning_(false)
{
}

EpsilonMeshPlanner::~EpsilonMeshPlanner()
{
}

bool EpsilonMeshPlanner::initialize(const std::string& plugin_name,
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

  data_collector_ = std::make_shared<epsilon_mesh_planner::DataCollector>(filepath, node->get_logger());
  data_collector_->setEnabled(enable_collection);

  if (enable_collection) {
    RCLCPP_INFO(node->get_logger(), "Data collection enabled. Saving to: %s",
                data_collector_->getFilePath().c_str());
  }

  // Wavefront configuration
  epsilon_mesh_planner::WaveFrontConfig wf_config;
  wf_config.cost_limit = node->declare_parameter(plugin_name + ".cost_limit", 1.0);
  wf_config.goal_dist_offset = node->declare_parameter(plugin_name + ".goal_dist_offset", 0.3);

  std::string update_method = node->declare_parameter(name_ + ".wavefront_update_method", "with_s");

  if (update_method == "with_s") {
    wf_config.update_method = epsilon_mesh_planner::WaveFrontConfig::UpdateMethod::WITH_S;
  } else if (update_method == "fmm") {
    wf_config.update_method = epsilon_mesh_planner::WaveFrontConfig::UpdateMethod::FMM;
  } else {
    wf_config.update_method = epsilon_mesh_planner::WaveFrontConfig::UpdateMethod::STANDARD;
  }

  wavefront_ = std::make_shared<epsilon_mesh_planner::WaveFront>(mesh_map_, wf_config, node->get_logger());

  // Terrain dynamics
  use_simplified_dynamics_ = node->declare_parameter(name_ + ".use_simplified_dynamics", false);

  std::string mesh_file = node->declare_parameter(name_ + ".mesh_file","data/mesh_export.obj");
  std::string curvature_file = node->declare_parameter(name_ + ".curvature_file", "/data/curvatures.npz");
  std::string gp_params_file = node->declare_parameter(name_ + ".gp_params_file","/data/gp_matrices.npz");

  double dt = node->declare_parameter(name_ + ".dynamics_dt", 0.1);
  double epsilon = node->declare_parameter(name_ + ".dynamics_epsilon", 0.5);
  double neighborhood_radius = node->declare_parameter(name_ + ".neighborhood_radius", 0.16);

  if (!use_simplified_dynamics_) {
    try {
      RCLCPP_INFO(node->get_logger(), "Initializing TerrainDynamics with CGAL...");
      terrain_dynamics_ = std::make_shared<epsilon_mesh_planner::TerrainDynamics>(
          mesh_file, curvature_file, gp_params_file, dt, epsilon, neighborhood_radius);
      RCLCPP_INFO(node->get_logger(), "TerrainDynamics initialized successfully");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node->get_logger(), "Failed to initialize TerrainDynamics: %s", e.what());
      RCLCPP_WARN(node->get_logger(), "Falling back to simple dynamics model");
      terrain_dynamics_ = nullptr;
      use_simplified_dynamics_ = true;
    }
  } else {
    RCLCPP_INFO(node->get_logger(), "Using simplified dynamics");
    terrain_dynamics_ = nullptr;
  }

  // Epsilon-constraint parameters
  num_candidates_ = node->declare_parameter(name_ + ".num_candidates", 100);
  epsilon_base_ = node->declare_parameter(name_ + ".epsilon_base", 0.5);
  alpha_safe_ = node->declare_parameter(name_ + ".alpha_safe", 0.523599);  // 30 degrees - alpha_max (hard limit)
  alpha_max_ = alpha_safe_;  // Same value, used as hard constraint
  noise_stddev_v_ = node->declare_parameter(name_ + ".noise_stddev_v", 0.5);
  noise_stddev_w_ = node->declare_parameter(name_ + ".noise_stddev_w", 0.3);
  
  // FMM biasing parameter
  fmm_bias_strength_ = node->declare_parameter(name_ + ".fmm_bias_strength", 0.5);  // Blend 50% FMM, 50% random

  // Adaptive epsilon parameters 
  // Note: expected_max_roughness_ and expected_max_inclination_ are computed from data, not config
  roughness_adaptation_strength_ = node->declare_parameter(name_ + ".roughness_adaptation_strength", 0.5);
  slope_adaptation_strength_ = node->declare_parameter(name_ + ".slope_adaptation_strength", 0.3);

  // Control constraints
  min_velocity_ = node->declare_parameter(name_ + ".min_velocity", -1.5);
  max_velocity_ = node->declare_parameter(name_ + ".max_velocity", 1.5);
  min_omega_ = node->declare_parameter(name_ + ".min_omega", -1.21832);
  max_omega_ = node->declare_parameter(name_ + ".max_omega", 1.2832);

  // Stability parameters (for F2) - terrain-relative formulation
  max_roll_ = node->declare_parameter(name_ + ".max_roll", 0.349066);  // 20 degrees - alpha_safe (no penalty below)
  max_pitch_ = max_roll_;  // Same as roll
  kappa_ = node->declare_parameter(name_ + ".kappa", 10.0);  // Penalty coefficient
  s_exp_ = node->declare_parameter(name_ + ".s_exp", 2.0);  // Goal proximity scaling exponent

  // Execution parameters
  time_horizon_ = node->declare_parameter(name_ + ".time_horizon", 10);
  control_dt_ = node->declare_parameter(name_ + ".control_dt", 0.1);
  max_steps_ = node->declare_parameter(name_ + ".max_steps", 1000);
  goal_threshold_ = node->declare_parameter(name_ + ".goal_threshold", 0.5);
  position_projection_radius_ = node->declare_parameter(name_ + ".position_projection_radius", 0.4f);

  // Load precomputed vertex descriptors
  std::string descriptors_file = node->declare_parameter(name_ + ".descriptors_file", "/data/vertex_descriptors.npy");
  
  bool descriptors_loaded = false;
  
  if (terrain_dynamics_ && terrain_dynamics_->hasDescriptors()) {
    RCLCPP_INFO(node->get_logger(), "Using descriptors from TerrainDynamics");
    descriptors_loaded = true;
  } else {
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
      descriptors_loaded = true;
      
      // Compute expected max values from actual data
      expected_max_roughness_ = 0.0;
      expected_max_inclination_ = 0.0;
      
      for (size_t i = 0; i < n_vertices; ++i) {
        double roughness = vertex_descriptors_[i][3];  // Index 3 is roughness
        double n_z = vertex_descriptors_[i][2];        // Index 2 is n_z
        double inclination = std::acos(std::clamp(n_z, -1.0, 1.0));
        
        expected_max_roughness_ = std::max(expected_max_roughness_, roughness);
        expected_max_inclination_ = std::max(expected_max_inclination_, inclination);
      }
      
      // Add small safety margin to avoid division by zero
      if (expected_max_roughness_ < 1e-6) expected_max_roughness_ = 0.1;
      if (expected_max_inclination_ < 1e-6) expected_max_inclination_ = M_PI_4;  // 45 degrees minimum
      
      RCLCPP_INFO(node->get_logger(), 
                  "Computed from terrain data: max_roughness=%.4f, max_inclination=%.2f° (%.3f rad)",
                  expected_max_roughness_, 
                  expected_max_inclination_ * 180.0 / M_PI,
                  expected_max_inclination_);
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node->get_logger(), "Failed to load vertex descriptors: %s", e.what());
    }
  }
  
  if (!descriptors_loaded) {
    RCLCPP_ERROR(node->get_logger(), "Epsilon-Constraint planner REQUIRES terrain descriptors!");
    RCLCPP_ERROR(node->get_logger(), "Please provide valid descriptors_file parameter.");
    return false;
  }
  
  // If using TerrainDynamics, compute expected max values from that data source
  if (terrain_dynamics_ && terrain_dynamics_->hasDescriptors() && vertex_descriptors_.empty()) {
    RCLCPP_WARN(node->get_logger(), "Using TerrainDynamics descriptors - cannot pre-compute max values");
    RCLCPP_WARN(node->get_logger(), "Using conservative defaults: max_roughness=0.1, max_inclination=60°");
    expected_max_roughness_ = 0.1;
    expected_max_inclination_ = M_PI / 3.0;  // 60 degrees
  }

  RCLCPP_INFO(node->get_logger(), "Epsilon-Constraint Planner initialized (terrain-relative stability with adaptive epsilon):");
  RCLCPP_INFO(node->get_logger(), "  Candidates: %d, Horizon: %d", num_candidates_, time_horizon_);
  RCLCPP_INFO(node->get_logger(), "  Epsilon_base: %.3f (adaptive), Alpha_safe (penalty-free): %.3f rad, Alpha_max (hard limit): %.3f rad",
              epsilon_base_, max_roll_, alpha_safe_);
  RCLCPP_INFO(node->get_logger(), "  Adaptive params: roughness_strength=%.2f, slope_strength=%.2f",
              roughness_adaptation_strength_, slope_adaptation_strength_);
  RCLCPP_INFO(node->get_logger(), "  Terrain ranges: roughness=[0, %.4f], inclination=[0, %.2f°]",
              expected_max_roughness_, expected_max_inclination_ * 180.0 / M_PI);
  RCLCPP_INFO(node->get_logger(), "  FMM bias strength: %.1f%% (0=random, 100=pure FMM)",
              fmm_bias_strength_ * 100.0);

  return true;
}

bool EpsilonMeshPlanner::cancel()
{
  if (wavefront_) {
    wavefront_->cancel();
  }
  cancel_planning_ = true;
  return true;
}

State EpsilonMeshPlanner::dynamics(const State& x, const Control& u, double dt)
{
  if (terrain_dynamics_) {
    std::array<double, 6> state = {x[0], x[1], x[2], x[3], x[4], x[5]};
    std::array<double, 2> control = {u[0], u[1]};
    auto next = terrain_dynamics_->dynamics(state, control);
    return {next[0], next[1], next[2], next[3], next[4], next[5]};
  }

  // Simplified dynamics
  double x_pos = x[0];
  double y_pos = x[1];
  double z_pos = x[2];
  double psi = x[5];

  double u0 = u[0];
  double u1 = u[1];

  double x_next = x_pos + u0 * cos(psi) * dt;
  double y_next = y_pos + u0 * sin(psi) * dt;
  double psi_next = psi + u1 * dt;

  return {x_next, y_next, z_pos, x[3], x[4], psi_next};
}

TerrainDescriptor EpsilonMeshPlanner::getTerrainDescriptor(const mesh_map::Vector& position)
{
  TerrainDescriptor descriptor;

  // Use TerrainDynamics KD-tree lookup (fast O(log n))
  if (terrain_dynamics_ && terrain_dynamics_->hasDescriptors()) {
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
  throw std::runtime_error("Epsilon planner requires terrain descriptors - none available!");
}

std::pair<double, double> EpsilonMeshPlanner::computeTerrainAlignedOrientation(
    const TerrainDescriptor& descriptor)
{
  // Compute theta_terrain and phi_terrain from terrain normal using atan2
  // theta_terrain = atan2(n_y, n_z)
  // phi_terrain = atan2(-n_x, n_z)
  double theta_terrain = std::atan2(descriptor.n_y, descriptor.n_z);
  double phi_terrain = std::atan2(-descriptor.n_x, descriptor.n_z);
  
  return {theta_terrain, phi_terrain};
}

Trajectory EpsilonMeshPlanner::simulateForward(
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

std::vector<TerrainDescriptor> EpsilonMeshPlanner::extractTerrainDescriptors(
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

double EpsilonMeshPlanner::computeAdaptiveEpsilon(const TerrainDescriptor& current_terrain_descriptor)
{
  // Compute adaptive epsilon based on current robot position's terrain characteristics
  // More challenging terrain (rough or steep) → tighter epsilon → more conservative planning
  
  double roughness = current_terrain_descriptor.roughness;
  double inclination = current_terrain_descriptor.inclination;
  
  // Normalize roughness [0, expected_max] → [0, 1]
  double normalized_roughness = std::min(1.0, roughness / expected_max_roughness_);
  
  // Normalize inclination [0, π/2] → [0, 1]
  double normalized_inclination = std::min(1.0, inclination / expected_max_inclination_);
  
  // Compute reduction factors (higher terrain difficulty → smaller factor → tighter epsilon)
  // roughness_factor: 1.0 (flat) → 0.5 (very rough)
  double roughness_factor = 1.0 - (roughness_adaptation_strength_ * normalized_roughness);
  
  // slope_factor: 1.0 (flat) → 0.7 (steep)
  double slope_factor = 1.0 - (slope_adaptation_strength_ * normalized_inclination);
  
  // Take the more conservative (smaller) factor
  double combined_factor = std::min(roughness_factor, slope_factor);
  
  // Ensure we don't go below a minimum threshold
  combined_factor = std::max(0.3, combined_factor);
  
  double adaptive_epsilon = epsilon_base_ * combined_factor;
  
  RCLCPP_DEBUG(node_->get_logger(), 
               "Adaptive epsilon: roughness=%.4f (norm=%.2f), inclination=%.2f° (norm=%.2f), "
               "factors: rough=%.2f, slope=%.2f → ε=%.3f (base=%.3f)",
               roughness, normalized_roughness, 
               inclination * 180.0 / M_PI, normalized_inclination,
               roughness_factor, slope_factor,
               adaptive_epsilon, epsilon_base_);
  
  return adaptive_epsilon;
}

double EpsilonMeshPlanner::stabilityViolation(double delta_angle)
{
  // v_tilt(Δα) as defined in paper - measures deviation from terrain-aligned orientation
  // Δα = actual_angle - terrain_aligned_angle
  double abs_delta = std::abs(delta_angle);
  
  if (abs_delta <= max_roll_) { // alpha_safe (safe deviation from terrain-normal)
    return 0.0;
  } else if (abs_delta < alpha_safe_) { // alpha_max (maximum allowable deviation)
    // Quadratic penalty: κ(|Δα| - α_safe)^2
    return kappa_ * std::pow(abs_delta - max_roll_, 2.0);
  } else {
    // Hard constraint violation - guaranteed tip-over
    return std::numeric_limits<double>::infinity();
  }
}

double EpsilonMeshPlanner::objectiveF2(
    const Trajectory& trajectory,
    const std::vector<TerrainDescriptor>& descriptors)
{
  // F2: Sum of stability violations (terrain-relative orientation deviations)
  // Measures deviation from terrain-aligned orientation at each state
  
  if (trajectory.size() != descriptors.size()) {
    RCLCPP_ERROR(node_->get_logger(), 
                 "Trajectory and descriptor sizes mismatch: %zu vs %zu",
                 trajectory.size(), descriptors.size());
    return std::numeric_limits<double>::infinity();
  }
  
  double total_violation = 0.0;

  for (size_t i = 0; i < trajectory.size(); ++i) {
    const auto& state = trajectory[i];
    const auto& descriptor = descriptors[i];
    
    // Get actual robot orientation
    double theta_actual = state[4];  // pitch
    double phi_actual = state[3];    // roll
    
    // Compute terrain-aligned orientation from terrain normal
    auto [theta_terrain, phi_terrain] = computeTerrainAlignedOrientation(descriptor);
    
    // Compute deviations from terrain-aligned orientation
    double delta_theta = theta_actual - theta_terrain;
    double delta_phi = phi_actual - phi_terrain;
    
    // Compute violations
    double violation_theta = stabilityViolation(delta_theta);
    double violation_phi = stabilityViolation(delta_phi);
    
    if (std::isinf(violation_theta) || std::isinf(violation_phi)) {
      return std::numeric_limits<double>::infinity(); // Hard constraint violated
    }
    
    total_violation += violation_theta + violation_phi;
  }

  return total_violation;
}

boost::optional<mesh_map::Vector> EpsilonMeshPlanner::calculateDirectionAtPosition(
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

float EpsilonMeshPlanner::computeDirectionCost(
    const mesh_map::Vector& current,
    const mesh_map::Vector& next,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  auto mesh_dir_opt = calculateDirectionAtPosition(current, vector_map);
  if (!mesh_dir_opt) return std::numeric_limits<float>::infinity();

  mesh_map::Vector mesh_dir = mesh_dir_opt.get().normalized();
  mesh_map::Vector movement_dir = next - current;
  float movement_length = movement_dir.length();

  if (movement_length < 1e-6f)
    return std::numeric_limits<float>::infinity();

  float cos_theta = movement_dir.normalized().dot(mesh_dir);
  float direction_cost = 1.0f - cos_theta;

  return direction_cost;
}

double EpsilonMeshPlanner::objectiveF1(
    const Trajectory& trajectory,
    const State& state_target,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  // F1: Cumulative directional misalignment with FMM field, scaled by goal proximity
  // Uses computeDirectionCost which already computes: 1 - cos(θ)
  
  if (trajectory.size() < 2) {
    return 0.0;
  }
  
  double total_cost = 0.0;
  
  // Compute d_max once (max distance from any vertex to goal)
  double d_max = 0.0;
  const auto& mesh = mesh_map_->mesh();
  mesh_map::Vector p_goal(state_target[0], state_target[1], state_target[2]);
  
  for (const auto& vh : mesh->vertices()) {
    auto vertex_pos = mesh->getVertexPosition(vh);
    mesh_map::Vector v_pos(vertex_pos.x, vertex_pos.y, vertex_pos.z);
    double dist = (v_pos - p_goal).length();
    d_max = std::max(d_max, dist);
  }
  
  if (d_max < 1e-6) d_max = 1.0; // Avoid division by zero

  for (size_t i = 1; i < trajectory.size(); ++i) {
    mesh_map::Vector p_prev(trajectory[i-1][0], trajectory[i-1][1], trajectory[i-1][2]);
    mesh_map::Vector p_curr(trajectory[i][0], trajectory[i][1], trajectory[i][2]);
    
    // Use computeDirectionCost to get directional misalignment
    float misalignment = computeDirectionCost(p_prev, p_curr, vector_map);
    
    if (!std::isfinite(misalignment)) {
      return std::numeric_limits<double>::infinity();
    }
    
    // Goal proximity scaling: s(p) = 1 - (||p - p_goal|| / d_max)^s_exp
    double dist_to_goal = (p_curr - p_goal).length();
    double s_p = 1.0 - std::pow(dist_to_goal / d_max, s_exp_);
    s_p = std::max(0.0, std::min(1.0, s_p)); // Clamp to [0,1]
    
    total_cost += misalignment * s_p;
  }

  return total_cost;
}

bool EpsilonMeshPlanner::checkHardConstraints(
    const Trajectory& trajectory,
    const std::vector<TerrainDescriptor>& descriptors)
{
  if (trajectory.size() != descriptors.size()) {
    return false;
  }
  
  const auto& mesh = mesh_map_->mesh();
  
  // Check collision-free and orientation constraints
  for (size_t i = 0; i < trajectory.size(); ++i) {
    const auto& state = trajectory[i];
    mesh_map::Vector position(state[0], state[1], state[2]);
    
    // Check collision using mesh_map directly
    mesh_map::Vector pos_copy = position;
    auto face_opt = mesh_map_->getContainingFace(pos_copy, position_projection_radius_);
    if (!face_opt) {
      return false;
    }
    
    const auto face = face_opt.unwrap();
    const auto& vertex_handles = mesh->getVerticesOfFace(face);
    
    std::array<float, 3> barycentric_coords;
    float dist;
    mesh_map_->projectedBarycentricCoords(position, face, barycentric_coords, dist);
    
    float cost = mesh_map_->costAtPosition(vertex_handles, barycentric_coords);
    if (!std::isfinite(cost) || cost >= 1.0) {
      return false;
    }
    
    // Check hard orientation constraints (|Δα| < alpha_max)
    const auto& descriptor = descriptors[i];
    auto [theta_terrain, phi_terrain] = computeTerrainAlignedOrientation(descriptor);
    
    double delta_theta = std::abs(state[4] - theta_terrain);
    double delta_phi = std::abs(state[3] - phi_terrain);
    
    if (delta_theta >= alpha_safe_ || delta_phi >= alpha_safe_) {
      return false;  // Would cause tip-over
    }
  }

  return true;
}

std::vector<ControlSequence> EpsilonMeshPlanner::generateCandidateTrajectories(
    const State& current_state,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map,
    const std::optional<ControlSequence>& prev_best_controls)
{
  std::vector<ControlSequence> candidates(num_candidates_);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> noise_v(0, noise_stddev_v_);
  std::normal_distribution<> noise_w(0, noise_stddev_w_);

  for (int i = 0; i < num_candidates_; ++i) {
    ControlSequence candidate(time_horizon_);

    if (prev_best_controls && i < num_candidates_ / 2) {
      // Strategy 1: Warm start from previous solution
      for (int t = 0; t < time_horizon_; ++t) {
        if (t < static_cast<int>(prev_best_controls->size())) {
          candidate[t][0] = (*prev_best_controls)[t][0] + noise_v(gen);
          candidate[t][1] = (*prev_best_controls)[t][1] + noise_w(gen);
        } else {
          candidate[t][0] = noise_v(gen);
          candidate[t][1] = noise_w(gen);
        }

        // Clamp to constraints
        candidate[t][0] = std::clamp(candidate[t][0], min_velocity_, max_velocity_);
        candidate[t][1] = std::clamp(candidate[t][1], min_omega_, max_omega_);
      }
    } else {
      // Strategy 2: Random exploration with optional FMM biasing
      std::uniform_real_distribution<> dis_v(min_velocity_, max_velocity_);
      std::uniform_real_distribution<> dis_w(min_omega_, max_omega_);

      // Simulate forward to get positions for FMM lookup (if biasing enabled)
      State state = current_state;
      
      for (int t = 0; t < time_horizon_; ++t) {
        // Generate random control
        double random_v = dis_v(gen);
        double random_w = dis_w(gen);
        
        if (fmm_bias_strength_ > 0.0) {
          // Get FMM direction and compute FMM-suggested control
          mesh_map::Vector position(state[0], state[1], state[2]);
          auto fmm_dir_opt = calculateDirectionAtPosition(position, vector_map);
          
          if (fmm_dir_opt) {
            mesh_map::Vector fmm_dir = fmm_dir_opt.get();
            
            // Compute desired yaw from FMM direction
            double desired_yaw = std::atan2(fmm_dir.y, fmm_dir.x);
            double current_yaw = state[5];
            double yaw_error = desired_yaw - current_yaw;
            
            // Normalize to [-pi, pi]
            while (yaw_error > M_PI) yaw_error -= 2.0 * M_PI;
            while (yaw_error < -M_PI) yaw_error += 2.0 * M_PI;
            
            // FMM-suggested control
            double fmm_v = max_velocity_ * 0.7;  // Conservative forward speed
            double fmm_w = std::clamp(yaw_error * 2.0, min_omega_, max_omega_);
            
            // Blend FMM with random based on bias strength
            candidate[t][0] = fmm_bias_strength_ * fmm_v + (1.0 - fmm_bias_strength_) * random_v;
            candidate[t][1] = fmm_bias_strength_ * fmm_w + (1.0 - fmm_bias_strength_) * random_w;
          } else {
            // FMM lookup failed, use pure random
            candidate[t][0] = random_v;
            candidate[t][1] = random_w;
          }
        } else {
          // No FMM biasing, pure random
          candidate[t][0] = random_v;
          candidate[t][1] = random_w;
        }
        
        // Clamp to constraints
        candidate[t][0] = std::clamp(candidate[t][0], min_velocity_, max_velocity_);
        candidate[t][1] = std::clamp(candidate[t][1], min_omega_, max_omega_);
        
        // Simulate forward one step for next FMM lookup
        if (fmm_bias_strength_ > 0.0 && t < time_horizon_ - 1) {
          state = dynamics(state, candidate[t], control_dt_);
        }
      }
    }

    candidates[i] = candidate;
  }

  return candidates;
}

ControlSequence EpsilonMeshPlanner::epsilonConstraintOptimization(
    const State& state_init,
    const State& state_target,
    const std::optional<ControlSequence>& prev_best_controls,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  // Compute adaptive epsilon based on CURRENT robot position's terrain
  mesh_map::Vector current_position(state_init[0], state_init[1], state_init[2]);
  TerrainDescriptor current_terrain = getTerrainDescriptor(current_position);
  double adaptive_epsilon = computeAdaptiveEpsilon(current_terrain);
  
  // Generate candidate trajectories with FMM biasing
  auto candidates = generateCandidateTrajectories(state_init, vector_map, prev_best_controls);

  struct CandidateEvaluation {
    int index;
    double f1;
    double f2;
    bool feasible;
  };

  std::vector<CandidateEvaluation> evaluations;
  evaluations.reserve(num_candidates_);

  // Evaluate all candidates
  for (int j = 0; j < num_candidates_; ++j) {
    CandidateEvaluation eval;
    eval.index = j;

    // Simulate forward
    Trajectory trajectory = simulateForward(state_init, candidates[j]);

    // Extract terrain descriptors
    auto descriptors = extractTerrainDescriptors(trajectory);

    // Compute objectives
    eval.f1 = objectiveF1(trajectory, state_target, vector_map);
    eval.f2 = objectiveF2(trajectory, descriptors);

    // Check constraints (using adaptive epsilon based on current terrain)
    bool hard_ok = checkHardConstraints(trajectory, descriptors);
    bool stability_ok = eval.f2 <= adaptive_epsilon;
    eval.feasible = hard_ok && stability_ok;

    evaluations.push_back(eval);
  }

  // Find feasible candidates
  std::vector<int> feasible_indices;
  for (const auto& eval : evaluations) {
    if (eval.feasible) {
      feasible_indices.push_back(eval.index);
    }
  }

  if (!feasible_indices.empty()) {
    // Select best feasible candidate (minimum f1)
    int best_idx = feasible_indices[0];
    double best_f1 = evaluations[best_idx].f1;

    for (int idx : feasible_indices) {
      if (evaluations[idx].f1 < best_f1) {
        best_f1 = evaluations[idx].f1;
        best_idx = idx;
      }
    }

    RCLCPP_DEBUG(node_->get_logger(), "Found %zu feasible trajectories (ε=%.3f), best f1: %.3f",
                feasible_indices.size(), adaptive_epsilon, best_f1);
    return candidates[best_idx];
  } else {
    // No feasible solution - return least violating or safe fallback
    RCLCPP_WARN(node_->get_logger(), "No feasible trajectory found (ε=%.3f), using safe fallback",
                adaptive_epsilon);
    
    // Find candidate with minimum constraint violation
    int least_violating = 0;
    double min_violation = evaluations[0].f2 - adaptive_epsilon;
    
    for (size_t j = 1; j < evaluations.size(); ++j) {
      double violation = evaluations[j].f2 - adaptive_epsilon;
      if (violation < min_violation) {
        min_violation = violation;
        least_violating = j;
      }
    }
    
    return candidates[least_violating];
  }
}

Trajectory EpsilonMeshPlanner::epsilonPlanner(
    const State& start_point,
    const State& goal_point,
    int time_horizon,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  tf2_ros::Buffer tf_buffer(node_->get_clock());
  tf2_ros::TransformListener tf_listener(tf_buffer);

  auto point_pub = node_->create_publisher<geometry_msgs::msg::PointStamped>(
      "~/current_point", rclcpp::QoS(10));
  auto twist_pub = node_->create_publisher<geometry_msgs::msg::TwistStamped>(
      "~/cmd_vel", rclcpp::QoS(10));
  auto ideal_dir_pub = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
      "~/ideal_direction", rclcpp::QoS(10));

  geometry_msgs::msg::PoseStamped current_state;
  geometry_msgs::msg::PoseStamped previous_state;
  bool odometry_received = false;
  bool previous_state_valid = false;

  std::mutex odom_mutex;

  auto pose_sub = node_->create_subscription<nav_msgs::msg::Odometry>(
      "/scout_base_controller/odom",
      rclcpp::QoS(10),
      [&current_state, &odometry_received, &odom_mutex, &tf_buffer, this](
          const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(odom_mutex);
        geometry_msgs::msg::PoseStamped pose_in_odom;
        pose_in_odom.header = msg->header;
        pose_in_odom.pose = msg->pose.pose;
        try {
          geometry_msgs::msg::PoseStamped pose_in_map;
          tf_buffer.transform(pose_in_odom, pose_in_map, mesh_map_->mapFrame(),
                            tf2::durationFromSec(1.0));
          current_state = pose_in_map;
          odometry_received = true;
        } catch (tf2::TransformException& ex) {
          RCLCPP_WARN(node_->get_logger(), "Transform failed: %s", ex.what());
        }
      });

  Trajectory trajectory;
  State x_current = start_point;
  trajectory.push_back(x_current);

  ControlSequence last_control;
  bool control_available = false;

  std_msgs::msg::Header header;
  header.frame_id = map_frame_;

  for (int step = 0; step < max_steps_ && !cancel_planning_; ++step) {
    // Publish current point
    geometry_msgs::msg::PointStamped point_msg;
    point_msg.header = header;
    point_msg.header.stamp = node_->now();
    point_msg.point.x = x_current[0];
    point_msg.point.y = x_current[1];
    point_msg.point.z = x_current[2];
    point_pub->publish(point_msg);

    // Publish ideal direction
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

    // Epsilon-constraint optimization
    ControlSequence optimal_control = epsilonConstraintOptimization(
        x_current, goal_point, prev_best_controls_, vector_map);

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
          data_collector_->collectTransition(
              previous_state, current_state,
              last_control[0][0], last_control[0][1]);
        }

        previous_state = current_state;
        previous_state_valid = true;

        x_current[0] = current_state.pose.position.x;
        x_current[1] = current_state.pose.position.y;
        x_current[2] = current_state.pose.position.z;

        tf2::Quaternion quat(
            current_state.pose.orientation.x,
            current_state.pose.orientation.y,
            current_state.pose.orientation.z,
            current_state.pose.orientation.w);
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

std::tuple<double, double, double> EpsilonMeshPlanner::getRPYFromQuaternion(
    const geometry_msgs::msg::Quaternion& q)
{
  tf2::Quaternion quat(q.x, q.y, q.z, q.w);
  tf2::Matrix3x3 m(quat);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  return std::make_tuple(roll, pitch, yaw);
}

uint32_t EpsilonMeshPlanner::makePlan(
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

  Trajectory traj = epsilonPlanner(start_state, goal_state, time_horizon_, vector_map);

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

}  // namespace epsilon_mesh_planner