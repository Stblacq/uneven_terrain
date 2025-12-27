#include <lvr2/geometry/Handles.hpp>
#include <lvr2/util/Meap.hpp>
#include <mesh_map/util.h>
#include <pluginlib/class_list_macros.hpp>
#include <random>
#include <cmath>
#include <chrono>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include "gakd_mesh_planner/gakd_mesh_planner.h"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <iostream>

PLUGINLIB_EXPORT_CLASS(gakd_mesh_planner::GAKDMeshPlanner, mbf_mesh_core::MeshPlanner);

namespace gakd_mesh_planner
{

GAKDMeshPlanner::GAKDMeshPlanner() : cancel_planning_(false)
{
}

GAKDMeshPlanner::~GAKDMeshPlanner()
{
}

bool GAKDMeshPlanner::initialize(const std::string& plugin_name, 
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

  std::string mesh_file = node->declare_parameter(name_ + ".mesh_file","/data/mesh_export.obj");
  std::string curvature_file = node->declare_parameter(name_ + ".curvature_file", "/data/curvatures.npz");
  std::string model_dir = node->declare_parameter(name_ + ".model_dir", "/data/gp_residual_export");
  
  double dt = node->declare_parameter(name_ + ".dynamics_dt", 0.1);
  double epsilon = node->declare_parameter(name_ + ".dynamics_epsilon", 0.5);
  double neighborhood_radius = node->declare_parameter(name_ + ".neighborhood_radius", 0.16);
  
  // Dynamics model selection
  bool use_simplified_dynamics = node->declare_parameter(name_ + ".use_simplified_dynamics", false);
  
  // Control constraints
  double min_velocity = node->declare_parameter(name_ + ".min_velocity", -1.5);
  double max_velocity = node->declare_parameter(name_ + ".max_velocity", 1.5);
  double min_omega = node->declare_parameter(name_ + ".min_omega", -1.21832);
  double max_omega = node->declare_parameter(name_ + ".max_omega", 1.2832);
  
  // GA parameters
  int population_size = node->declare_parameter(name_ + ".population_size", 100);
  int num_generations = node->declare_parameter(name_ + ".num_generations", 50);
  int time_horizon = node->declare_parameter(name_ + ".time_horizon", 10);
  double mutation_rate = node->declare_parameter(name_ + ".mutation_rate", 0.01);
  int tournament_size = node->declare_parameter(name_ + ".tournament_size", 3);
  double mutation_stddev = node->declare_parameter(name_ + ".mutation_stddev", 0.1);
  
  // Execution parameters
  double control_dt = node->declare_parameter(name_ + ".control_dt", 0.1);
  int max_steps = node->declare_parameter(name_ + ".max_steps", 1000);
  double goal_threshold = node->declare_parameter(name_ + ".goal_threshold", 0.5);

  // Mesh parameters
  float position_projection_radius = node->declare_parameter(name_ + ".position_projection_radius", 0.4f);
  
  // Store in member variables
  use_simplified_dynamics_ = use_simplified_dynamics;
  min_velocity_ = min_velocity;
  max_velocity_ = max_velocity;
  min_omega_ = min_omega;
  max_omega_ = max_omega;
  population_size_ = population_size;
  num_generations_ = num_generations;
  time_horizon_ = time_horizon;
  mutation_rate_ = mutation_rate;
  tournament_size_ = tournament_size;
  mutation_stddev_ = mutation_stddev;
  control_dt_ = control_dt;
  max_steps_ = max_steps;
  goal_threshold_ = goal_threshold;
  position_projection_radius_ = position_projection_radius;

// Initialize terrain dynamics only if not using simplified model
  if (!use_simplified_dynamics_) {
    try {
      RCLCPP_INFO(node->get_logger(), "Initializing TerrainDynamics with GP corrections...");
      RCLCPP_INFO(node->get_logger(), "  mesh_file: %s", mesh_file.c_str());
      RCLCPP_INFO(node->get_logger(), "  curvature_file: %s", curvature_file.c_str());
      RCLCPP_INFO(node->get_logger(), "  model_dir: %s", model_dir.c_str());
      RCLCPP_INFO(node->get_logger(), "  dt: %.3f, epsilon: %.3f, radius: %.3f", dt, epsilon, neighborhood_radius);
      
      terrain_dynamics_ = std::make_shared<gakd_mesh_planner::TerrainDynamics>(
          mesh_file,
          curvature_file,
          model_dir,
          dt,
          epsilon,
          neighborhood_radius
      );
      RCLCPP_INFO(node->get_logger(), "TerrainDynamics initialized successfully");
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node->get_logger(), "Failed to initialize TerrainDynamics: %s", e.what());
      RCLCPP_WARN(node->get_logger(), "Falling back to simplified dynamics model");
      terrain_dynamics_ = nullptr;
      use_simplified_dynamics_ = true;
    }
  } else {
    RCLCPP_INFO(node->get_logger(), "Using simplified 2D dynamics model (no terrain corrections)");
    terrain_dynamics_ = nullptr;
  }
  RCLCPP_INFO(node->get_logger(), "use_simplified_dynamics: %s", use_simplified_dynamics_ ? "true" : "false");

  return true;
}


bool GAKDMeshPlanner::cancel()
{
  if (wavefront_)
  {
    wavefront_->cancel();
  }
  cancel_planning_ = true;
  return true;
}


State GAKDMeshPlanner::dynamics(const State& x, const Control& u, double dt)
{
  if (terrain_dynamics_) {
    std::array<double, 6> state = {x[0], x[1], x[2], x[3], x[4], x[5]};
    std::array<double, 2> control = {u[0], u[1]};
    auto next = terrain_dynamics_->dynamics(state, control);
    return {next[0], next[1], next[2], next[3], next[4], next[5]};
  }
  
  // Fallback to simple dynamics if terrain_dynamics_ not available
  double x_pos = x[0];
  double y_pos = x[1];
  double z_pos = x[2];
  double phi = x[3];
  double theta = x[4];
  double psi = x[5];

  double u0 = u[0];
  double u1 = u[1];

  double x_next = x_pos + u0 * cos(psi) * dt;
  double y_next = y_pos + u0 * sin(psi) * dt;
  double psi_next = psi + u1 * dt;

  return {x_next, y_next, z_pos, phi, theta, psi_next};
}

Control GAKDMeshPlanner::randomControl()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_velocity(min_velocity_, max_velocity_);
  std::uniform_real_distribution<> dis_steering(min_omega_, max_omega_);
  return {dis_velocity(gen), dis_steering(gen)};
}

std::vector<ControlSequence> GAKDMeshPlanner::generatePopulation(int size, int horizon)
{
  std::vector<ControlSequence> population(size, ControlSequence(horizon));
  for (auto& individual : population)
  {
    for (auto& controls : individual)
    {
      controls = randomControl();
    }
  }
  return population;
}

std::pair<ControlSequence, ControlSequence> GAKDMeshPlanner::selectParents(
    const std::vector<ControlSequence>& population,
    const std::vector<double>& fitnesses)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, population.size() - 1);
  std::vector<std::pair<ControlSequence, double>> tournament;
  for (int i = 0; i < tournament_size_; ++i)
  {
    int idx = dis(gen);
    tournament.push_back({population[idx], fitnesses[idx]});
  }
  std::sort(tournament.begin(), tournament.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
  return {tournament[0].first, tournament[1].first};
}

std::pair<ControlSequence, ControlSequence> GAKDMeshPlanner::crossover(
    const ControlSequence& parent1, 
    const ControlSequence& parent2)
{
  int horizon = parent1.size();
  ControlSequence offspring1, offspring2;
  offspring1.reserve(horizon);
  offspring2.reserve(horizon);
  for (int i = 0; i < horizon; ++i)
  {
    if (rand() % 2 == 0)
    {
      offspring1.push_back(parent1[i]);
      offspring2.push_back(parent2[i]);
    }
    else
    {
      offspring1.push_back(parent2[i]);
      offspring2.push_back(parent1[i]);
    }
  }
  return {offspring1, offspring2};
}

ControlSequence GAKDMeshPlanner::mutate(ControlSequence control_sequence, double mutation_rate)
{
  std::random_device rd;
  std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, mutation_stddev_);
  for (auto& control : control_sequence)
  {
    for (auto& val : control)
    {
      if (static_cast<double>(rand()) / RAND_MAX < mutation_rate)
      {
        val += dis(gen);
      }
    }
  }
  return control_sequence;
}

float GAKDMeshPlanner::calculateCostAtPosition(const mesh_map::Vector& position)
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

boost::optional<mesh_map::Vector> GAKDMeshPlanner::calculateDirectionAtPosition(
    const mesh_map::Vector& position,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  const auto& mesh = mesh_map_->mesh();
  
  mesh_map::Vector pos_copy = position;
  
  auto goal_face_opt = mesh_map_->getContainingFace(pos_copy, 0.4);
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

float GAKDMeshPlanner::computeDirectionCost(
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

double GAKDMeshPlanner::costFunction(
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

  return direction_cost;
}

ControlSequence GAKDMeshPlanner::geneticAlgorithm(
    const State& state_init, 
    const State& state_target,
    double dt, 
    int pop_size, 
    int generations,
    int time_horizon, 
    double mutation_rate,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  auto population = generatePopulation(pop_size, time_horizon);
  std::vector<double> fitnesses(pop_size);

  for (int generation = 0; generation < generations; ++generation)
  {
    for (int i = 0; i < pop_size; ++i)
    {
      State current_state = state_init;
      double total_cost = 0.0;
      for (const auto& control : population[i])
      {
        State next_state = dynamics(current_state, control, dt);
        total_cost += costFunction(current_state, next_state, state_target, vector_map);
        current_state = next_state;
      }
      fitnesses[i] = total_cost;
    }

    auto min_fitness = *std::min_element(fitnesses.begin(), fitnesses.end());
    if (min_fitness < 1e-8)
    {
      break;
    }

    std::vector<ControlSequence> new_population;
    new_population.reserve(pop_size);
    for (int i = 0; i < pop_size / 2; ++i)
    {
      auto [parent1, parent2] = selectParents(population, fitnesses);
      auto [offspring1, offspring2] = crossover(parent1, parent2);
      offspring1 = mutate(offspring1, mutation_rate);
      offspring2 = mutate(offspring2, mutation_rate);
      new_population.push_back(offspring1);
      new_population.push_back(offspring2);
    }
    population = new_population;
  }

  int best_idx = std::min_element(fitnesses.begin(), fitnesses.end()) - fitnesses.begin();
  return population[best_idx];
}


Trajectory GAKDMeshPlanner::gaPlanner(
    const State& start_point, 
    const State& goal_point,
    double time_horizon,
    const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map)
{
  tf2_ros::Buffer tf_buffer(node_->get_clock());
  tf2_ros::TransformListener tf_listener(tf_buffer);

  auto point_pub = node_->create_publisher<geometry_msgs::msg::PointStamped>("~/current_point", rclcpp::QoS(10));
  auto twist_pub = node_->create_publisher<geometry_msgs::msg::TwistStamped>("~/cmd_vel", rclcpp::QoS(10));
  auto ideal_dir_pub = node_->create_publisher<geometry_msgs::msg::PoseStamped>("~/ideal_direction", rclcpp::QoS(10));
  
  geometry_msgs::msg::PoseStamped current_state;
  geometry_msgs::msg::PoseStamped previous_state;
  bool odometry_received = false;
  bool previous_state_valid = false;

  std::mutex odom_mutex;

  auto pose_sub = node_->create_subscription<nav_msgs::msg::Odometry>(
      "/scout_base_controller/odom", 
      rclcpp::QoS(10),
      [&current_state, &odometry_received, &odom_mutex, &tf_buffer, this](const nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(odom_mutex);
        geometry_msgs::msg::PoseStamped pose_in_odom;
        pose_in_odom.header = msg->header;
        pose_in_odom.pose = msg->pose.pose;
        try {
          geometry_msgs::msg::PoseStamped pose_in_map;
          tf_buffer.transform(pose_in_odom, pose_in_map, mesh_map_->mapFrame(), tf2::durationFromSec(1.0));
          current_state = pose_in_map;
          odometry_received = true;
        } catch (tf2::TransformException &ex) {
          RCLCPP_WARN(node_->get_logger(), "Transform failed: %s", ex.what());
        }
      });

  double dt = control_dt_;
  int max_steps = max_steps_;
  double goal_threshold = goal_threshold_;
  Trajectory trajectory;
  State x_current = start_point;
  trajectory.push_back(x_current);

  ControlSequence last_control;
  bool control_available = false;

  std_msgs::msg::Header header;
  header.frame_id = map_frame_;

  for (int step = 0; step < max_steps && !cancel_planning_; ++step)
  {
    geometry_msgs::msg::PointStamped point_msg;
    point_msg.header = header;
    point_msg.header.stamp = node_->now();
    point_msg.point.x = x_current[0];
    point_msg.point.y = x_current[1];
    point_msg.point.z = x_current[2];
    point_pub->publish(point_msg);

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

    ControlSequence optimal_control = geneticAlgorithm(x_current, goal_point, dt, population_size_,
       num_generations_, time_horizon_, mutation_rate_,vector_map);
    auto control_time = node_->now();

    geometry_msgs::msg::TwistStamped twist_msg;
    twist_msg.header = header;
    twist_msg.header.stamp = control_time;
    twist_msg.twist.linear.x = optimal_control[0][0];
    twist_msg.twist.angular.z = optimal_control[0][1];
    twist_pub->publish(twist_msg);

    {
      std::lock_guard<std::mutex> lock(odom_mutex);
      if (odometry_received) {
        if (previous_state_valid && control_available && data_collector_ && data_collector_->isEnabled()) {
          data_collector_->collectTransition(
              previous_state,
              current_state,
              last_control[0][0],
              last_control[0][1]
          );
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
            current_state.pose.orientation.w
        );
        double roll, pitch, yaw;
        tf2::Matrix3x3(quat).getRPY(roll, pitch, yaw);
        x_current[3] = roll;
        x_current[4] = pitch;
        x_current[5] = yaw;
      } else {
        RCLCPP_WARN(node_->get_logger(), "No odometry data received at step %d. Using dynamics.", step);
        x_current = dynamics(x_current, optimal_control[0], dt);
      }
    }
    
    last_control = optimal_control;
    control_available = true;

    trajectory.push_back(x_current);

    double dist = std::sqrt(
        std::pow(x_current[0] - goal_point[0], 2) +
        std::pow(x_current[1] - goal_point[1], 2) +
        std::pow(x_current[2] - goal_point[2], 2));

    if (dist < goal_threshold)
    {
      point_msg.header.stamp = node_->now();
      point_msg.point.x = x_current[0];
      point_msg.point.y = x_current[1];
      point_msg.point.z = x_current[2];
      point_pub->publish(point_msg);
      
      RCLCPP_INFO(node_->get_logger(), "Goal reached!");
      if (data_collector_ && data_collector_->isEnabled()) {
        data_collector_->flush();
        RCLCPP_INFO(node_->get_logger(), "Data collection saved to: %s", 
                   data_collector_->getFilePath().c_str());
      }
      return trajectory;
    }
  }

  RCLCPP_WARN(node_->get_logger(), "Max steps reached without reaching the goal.");
  if (data_collector_ && data_collector_->isEnabled()) {
    data_collector_->flush();
  }
  return {};
}

std::tuple<double, double, double> GAKDMeshPlanner::getRPYFromQuaternion(
    const geometry_msgs::msg::Quaternion& q)
{
  tf2::Quaternion quat(q.x, q.y, q.z, q.w);
  tf2::Matrix3x3 m(quat);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);
  return std::make_tuple(roll, pitch, yaw);
}

uint32_t GAKDMeshPlanner::makePlan(
    const geometry_msgs::msg::PoseStamped& start,
    const geometry_msgs::msg::PoseStamped& goal, 
    double tolerance, 
    std::vector<geometry_msgs::msg::PoseStamped>& plan,
    double& cost, 
    std::string& message)
{
  mesh_map::Vector start_vec = mesh_map::toVector(start.pose.position);
  mesh_map::Vector goal_vec = mesh_map::toVector(goal.pose.position);

  if (!wavefront_->compute(start_vec, goal_vec, message))
  {
    RCLCPP_ERROR_STREAM(node_->get_logger(), "Wavefront computation failed: " << message);
    return mbf_msgs::action::GetPath::Result::NO_PATH_FOUND;
  }
  
  const auto& vector_map = wavefront_->getVectorMap();

  std_msgs::msg::Header header;
  header.stamp = node_->now();
  header.frame_id = map_frame_;
  
  if (publish_vector_field_)
  {
    mesh_map_->publishVectorField("vector_field", vector_map, publish_face_vectors_);
  }

  double roll_start, pitch_start, yaw_start;
  std::tie(roll_start, pitch_start, yaw_start) = getRPYFromQuaternion(start.pose.orientation);
  State start_state = {start.pose.position.x, start.pose.position.y, start.pose.position.z, 
                       roll_start, pitch_start, yaw_start};

  double roll_goal, pitch_goal, yaw_goal;
  std::tie(roll_goal, pitch_goal, yaw_goal) = getRPYFromQuaternion(goal.pose.orientation);
  State goal_state = {goal.pose.position.x, goal.pose.position.y, goal.pose.position.z, 
                      roll_goal, pitch_goal, yaw_goal};

  Trajectory traj = gaPlanner(start_state, goal_state, time_horizon_, vector_map);

  if (cancel_planning_)
  {
    message = "Planning was canceled";
    RCLCPP_WARN_STREAM(node_->get_logger(), message);
    return mbf_msgs::action::GetPath::Result::CANCELED;
  }

  if (traj.empty())
  {
    message = "No path found to the goal";
    RCLCPP_WARN_STREAM(node_->get_logger(), message);
    return mbf_msgs::action::GetPath::Result::NO_PATH_FOUND;
  }

  RCLCPP_INFO_STREAM(node_->get_logger(), "Path found with " << traj.size() << " points");

  header.stamp = node_->now();
  header.frame_id = map_frame_;
  cost = 0.0;
  mesh_map::Vector current = start_vec;
  nav_msgs::msg::Path path_msg;
  path_msg.header = header;

  for (size_t i = 0; i < traj.size(); ++i)
  {
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

}  // namespace gakd_mesh_planner