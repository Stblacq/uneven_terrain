#include "mppi_mesh_planner/mppi_mesh_planner.h"
#include <lvr2/geometry/Handles.hpp>
#include <lvr2/util/Meap.hpp>
#include <mesh_map/util.h>
#include <pluginlib/class_list_macros.hpp>
#include <random>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <iostream>

PLUGINLIB_EXPORT_CLASS(mppi_mesh_planner::MPPIMeshPlanner, mbf_mesh_core::MeshPlanner);

namespace mppi_mesh_planner
{

MPPIMeshPlanner::MPPIMeshPlanner() : cancel_planning_(false)
{
}

MPPIMeshPlanner::~MPPIMeshPlanner()
{
}

bool MPPIMeshPlanner::initialize(const std::string& plugin_name,
                                 const std::shared_ptr<mesh_map::MeshMap>& mesh_map_ptr,
                                 const rclcpp::Node::SharedPtr& node)
{
  mesh_map_ = mesh_map_ptr;
  name_ = plugin_name;
  map_frame_ = mesh_map_->mapFrame();
  node_ = node;

  path_pub_ = node->create_publisher<nav_msgs::msg::Path>("~/path", rclcpp::QoS(1).transient_local());

  // Visualization parameters
  publish_vector_field_ = node->declare_parameter(name_ + ".publish_vector_field", false);
  publish_face_vectors_ = node->declare_parameter(name_ + ".publish_face_vectors", false);

  // Data collection
  node->declare_parameter("enable_data_collection", false);
  node->declare_parameter("data_collection_filepath", "");

  bool enable_collection = node->get_parameter("enable_data_collection").as_bool();
  std::string filepath = node->get_parameter("data_collection_filepath").as_string();

  data_collector_ = std::make_shared<mppi_mesh_planner::DataCollector>(filepath, node->get_logger());
  data_collector_->setEnabled(enable_collection);

  if (enable_collection) {
    RCLCPP_INFO(node->get_logger(), "Data collection enabled. Saving to: %s",
                data_collector_->getFilePath().c_str());
  }

  // Wavefront configuration
  mppi_mesh_planner::WaveFrontConfig wf_config;
  wf_config.cost_limit = node->declare_parameter(plugin_name + ".cost_limit", 1.0);
  wf_config.goal_dist_offset = node->declare_parameter(plugin_name + ".goal_dist_offset", 0.3);

  std::string update_method = node->declare_parameter(name_ + ".wavefront_update_method", "with_s");

  if (update_method == "with_s") {
    wf_config.update_method = mppi_mesh_planner::WaveFrontConfig::UpdateMethod::WITH_S;
  } else if (update_method == "fmm") {
    wf_config.update_method = mppi_mesh_planner::WaveFrontConfig::UpdateMethod::FMM;
  } else {
    wf_config.update_method = mppi_mesh_planner::WaveFrontConfig::UpdateMethod::STANDARD;
  }

  wavefront_ = std::make_shared<mppi_mesh_planner::WaveFront>(mesh_map_, wf_config, node->get_logger());

  // Terrain dynamics
  use_simplified_dynamics_ = node->declare_parameter(name_ + ".use_simplified_dynamics", false);

  std::string mesh_file = node->declare_parameter(name_ + ".mesh_file", "data/mesh_export.obj");
  std::string corrections_file = node->declare_parameter(name_ + ".corrections_file","/data/vertex_corrections.npy");

  double dt = node->declare_parameter(name_ + ".dynamics_dt", 0.1);
  double epsilon = node->declare_parameter(name_ + ".dynamics_epsilon", 0.5);

  if (!use_simplified_dynamics_) {
    try {
      RCLCPP_INFO(node->get_logger(), "Initializing TerrainDynamics...");
      terrain_dynamics_ = std::make_shared<mppi_mesh_planner::TerrainDynamics>(
          mesh_file, corrections_file, dt, epsilon);
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

  // MPPI parameters
  num_samples_ = node->declare_parameter(name_ + ".num_samples", 1000);
  num_iterations_ = node->declare_parameter(name_ + ".num_iterations", 5);
  time_horizon_ = node->declare_parameter(name_ + ".time_horizon", 10);
  lambda_ = node->declare_parameter(name_ + ".lambda", 1.0);
  noise_stddev_v_ = node->declare_parameter(name_ + ".noise_stddev_v", 0.5);
  noise_stddev_w_ = node->declare_parameter(name_ + ".noise_stddev_w", 0.3);

  // Control constraints
  min_velocity_ = node->declare_parameter(name_ + ".min_velocity", -1.5);
  max_velocity_ = node->declare_parameter(name_ + ".max_velocity", 1.5);
  min_omega_ = node->declare_parameter(name_ + ".min_omega", -1.21832);
  max_omega_ = node->declare_parameter(name_ + ".max_omega", 1.2832);

  // Cost weights
  weight_direction_ = node->declare_parameter(name_ + ".weight_direction", 1.0);
  weight_smoothness_ = node->declare_parameter(name_ + ".weight_smoothness", 0.1);
  weight_terminal_ = node->declare_parameter(name_ + ".weight_terminal", 10.0);

  // Execution parameters
  control_dt_ = node->declare_parameter(name_ + ".control_dt", 0.1);
  max_steps_ = node->declare_parameter(name_ + ".max_steps", 1000);
  goal_threshold_ = node->declare_parameter(name_ + ".goal_threshold", 0.5);
  position_projection_radius_ = node->declare_parameter(name_ + ".position_projection_radius", 0.4f);

  RCLCPP_INFO(node->get_logger(), "MPPI Planner initialized with:");
  RCLCPP_INFO(node->get_logger(), "  Samples: %d, Iterations: %d, Horizon: %d",
              num_samples_, num_iterations_, time_horizon_);
  RCLCPP_INFO(node->get_logger(), "  Lambda: %.2f, Noise: [%.2f, %.2f]",
              lambda_, noise_stddev_v_, noise_stddev_w_);

  return true;
}

bool MPPIMeshPlanner::cancel()
{
  if (wavefront_) {
    wavefront_->cancel();
  }
  cancel_planning_ = true;
  return true;
}

State MPPIMeshPlanner::dynamics(const State& x, const Control& u, double dt)
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

float MPPIMeshPlanner::calculateCostAtPosition(const mesh_map::Vector& position)
{
  const auto& mesh = mesh_map_->mesh();
  mesh_map::Vector pos_copy = position;

  auto goal_face_opt = mesh_map_->getContainingFace(pos_copy, position_projection_radius_);
  if (!goal_face_opt) {
    return std::numeric_limits<float>::infinity();
  }

  const auto goal_face = goal_face_opt.unwrap();
  const auto& vertex_handles = mesh->getVerticesOfFace(goal_face);

  std::array<float, 3> barycentric_coords;
  float dist;
  mesh_map_->projectedBarycentricCoords(position, goal_face, barycentric_coords, dist);

  float cost = mesh_map_->costAtPosition(vertex_handles, barycentric_coords);

  if (std::isnan(cost)) {
    return std::numeric_limits<float>::infinity();
  }

  return cost;
}

boost::optional<mesh_map::Vector> MPPIMeshPlanner::calculateDirectionAtPosition(
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

float MPPIMeshPlanner::computeDirectionCost(
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
    return std::numeric_limits<double>::infinity();

  float cos_theta = movement_dir.normalized().dot(mesh_dir);
  float direction_cost = 1.0f - cos_theta;

  return direction_cost;
}

double MPPIMeshPlanner::costFunction(
    const State& current_state,
    const State& next_state,
    const State& state_target,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  mesh_map::Vector current_vector(current_state[0], current_state[1], current_state[2]);
  mesh_map::Vector next_vector(next_state[0], next_state[1], next_state[2]);

  float direction_cost = computeDirectionCost(current_vector, next_vector, vector_map);

  if (!std::isfinite(direction_cost)) {
    return std::numeric_limits<double>::infinity();
  }

  return weight_direction_ * direction_cost;
}

double MPPIMeshPlanner::terminalCost(const State& state, const State& goal)
{
  double dx = state[0] - goal[0];
  double dy = state[1] - goal[1];
  double dz = state[2] - goal[2];
  return weight_terminal_ * (dx * dx + dy * dy + dz * dz);
}

std::vector<ControlSequence> MPPIMeshPlanner::sampleTrajectories(
    const ControlSequence& nominal_sequence)
{
  std::vector<ControlSequence> sampled(num_samples_);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> noise_v(0, noise_stddev_v_);
  std::normal_distribution<> noise_w(0, noise_stddev_w_);

  for (int i = 0; i < num_samples_; ++i) {
    sampled[i].resize(time_horizon_);
    for (int t = 0; t < time_horizon_; ++t) {
      double v = nominal_sequence[t][0] + noise_v(gen);
      double w = nominal_sequence[t][1] + noise_w(gen);
      
      // Clamp to constraints
      v = std::clamp(v, min_velocity_, max_velocity_);
      w = std::clamp(w, min_omega_, max_omega_);
      
      sampled[i][t] = {v, w};
    }
  }

  return sampled;
}

double MPPIMeshPlanner::evaluateTrajectory(
    const State& state_init,
    const ControlSequence& control_sequence,
    const State& state_target,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  State current_state = state_init;
  double total_cost = 0.0;

  for (int t = 0; t < time_horizon_; ++t) {
    State next_state = dynamics(current_state, control_sequence[t], control_dt_);
    
    double stage_cost = costFunction(current_state, next_state, state_target, vector_map);
    
    // Smoothness cost (penalize large control changes)
    if (t > 0) {
      double dv = control_sequence[t][0] - control_sequence[t-1][0];
      double dw = control_sequence[t][1] - control_sequence[t-1][1];
      stage_cost += weight_smoothness_ * (dv * dv + dw * dw);
    }
    
    total_cost += stage_cost;
    current_state = next_state;
    
    if (!std::isfinite(total_cost)) {
      return std::numeric_limits<double>::infinity();
    }
  }

  // Terminal cost
  total_cost += terminalCost(current_state, state_target);

  return total_cost;
}

ControlSequence MPPIMeshPlanner::updateNominalSequence(
    const ControlSequence& nominal_sequence,
    const std::vector<ControlSequence>& sampled_trajectories,
    const std::vector<double>& costs)
{
  // Find minimum cost for normalization
  double min_cost = *std::min_element(costs.begin(), costs.end());
  
  // Compute weights using exponential transform
  std::vector<double> weights(num_samples_);
  double weight_sum = 0.0;
  
  for (int i = 0; i < num_samples_; ++i) {
    if (std::isfinite(costs[i])) {
      weights[i] = std::exp(-lambda_ * (costs[i] - min_cost));
      weight_sum += weights[i];
    } else {
      weights[i] = 0.0;
    }
  }
  
  // Normalize weights
  if (weight_sum > 0) {
    for (int i = 0; i < num_samples_; ++i) {
      weights[i] /= weight_sum;
    }
  }
  
  // Compute weighted average control sequence
  ControlSequence new_nominal(time_horizon_);
  for (int t = 0; t < time_horizon_; ++t) {
    double v_sum = 0.0;
    double w_sum = 0.0;
    
    for (int i = 0; i < num_samples_; ++i) {
      v_sum += weights[i] * sampled_trajectories[i][t][0];
      w_sum += weights[i] * sampled_trajectories[i][t][1];
    }
    
    new_nominal[t] = {v_sum, w_sum};
  }
  
  return new_nominal;
}

ControlSequence MPPIMeshPlanner::mppiOptimization(
    const State& state_init,
    const State& state_target,
    const ControlSequence& nominal_sequence,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  ControlSequence current_nominal = nominal_sequence;
  
  for (int iter = 0; iter < num_iterations_; ++iter) {
    // Sample trajectories around nominal
    auto sampled_trajectories = sampleTrajectories(current_nominal);
    
    // Evaluate all sampled trajectories
    std::vector<double> costs(num_samples_);
    for (int i = 0; i < num_samples_; ++i) {
      costs[i] = evaluateTrajectory(state_init, sampled_trajectories[i], 
                                    state_target, vector_map);
    }
    
    // Update nominal sequence
    current_nominal = updateNominalSequence(current_nominal, sampled_trajectories, costs);
  }
  
  return current_nominal;
}

Trajectory MPPIMeshPlanner::mppiPlanner(
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

  // Initialize nominal sequence (zero controls)
  ControlSequence nominal_sequence(time_horizon_, {0.0, 0.0});
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

    // MPPI optimization
    ControlSequence optimal_control = mppiOptimization(
        x_current, goal_point, nominal_sequence, vector_map);

    // Publish control
    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header = header;
    twist_msg.header.stamp = node_->now();
    twist_msg.twist.linear.x = optimal_control[0][0];
    twist_msg.twist.angular.z = optimal_control[0][1];
    twist_pub->publish(twist_msg);

    // Update state from odometry or dynamics
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

    // Shift nominal sequence (warm start for next iteration)
    nominal_sequence.erase(nominal_sequence.begin());
    nominal_sequence.push_back({0.0, 0.0});

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

std::tuple<double, double, double> MPPIMeshPlanner::getRPYFromQuaternion(
    const geometry_msgs::msg::Quaternion& q)
{
  tf2::Quaternion quat(q.x, q.y, q.z, q.w);
  tf2::Matrix3x3 m(quat);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  return std::make_tuple(roll, pitch, yaw);
}

uint32_t MPPIMeshPlanner::makePlan(
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

  Trajectory traj = mppiPlanner(start_state, goal_state, time_horizon_, vector_map);

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

}  // namespace mppi_mesh_planner