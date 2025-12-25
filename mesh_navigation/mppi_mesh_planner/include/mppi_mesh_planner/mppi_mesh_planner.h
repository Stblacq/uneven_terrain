#ifndef MPPI_MESH_PLANNER_H
#define MPPI_MESH_PLANNER_H

#include <mbf_mesh_core/mesh_planner.h>
#include <mesh_map/mesh_map.h>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <vector>
#include <array>
#include <memory>
#include "mppi_mesh_planner/wavefront.h"
#include "mppi_mesh_planner/terrain_dynamics.hpp"
#include "mppi_mesh_planner/data_collector.h"
#include <mbf_msgs/action/get_path.hpp>


namespace mppi_mesh_planner
{

using State = std::array<double, 6>;  // [x, y, z, roll, pitch, yaw]
using Control = std::array<double, 2>;  // [linear_velocity, angular_velocity]
using ControlSequence = std::vector<Control>;
using Trajectory = std::vector<State>;

class MPPIMeshPlanner : public mbf_mesh_core::MeshPlanner
{
public:
  MPPIMeshPlanner();
  virtual ~MPPIMeshPlanner();

  virtual bool initialize(const std::string& plugin_name,
                         const std::shared_ptr<mesh_map::MeshMap>& mesh_map_ptr,
                         const rclcpp::Node::SharedPtr& node) override;

  virtual bool cancel() override;

  virtual uint32_t makePlan(const geometry_msgs::msg::PoseStamped& start,
                           const geometry_msgs::msg::PoseStamped& goal,
                           double tolerance,
                           std::vector<geometry_msgs::msg::PoseStamped>& plan,
                           double& cost,
                           std::string& message) override;

private:
  // Core MPPI functions
  ControlSequence mppiOptimization(
      const State& state_init,
      const State& state_target,
      const ControlSequence& nominal_sequence,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  std::vector<ControlSequence> sampleTrajectories(
      const ControlSequence& nominal_sequence);

  double evaluateTrajectory(
      const State& state_init,
      const ControlSequence& control_sequence,
      const State& state_target,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  ControlSequence updateNominalSequence(
      const ControlSequence& nominal_sequence,
      const std::vector<ControlSequence>& sampled_trajectories,
      const std::vector<double>& costs);

  // Dynamics and cost
  State dynamics(const State& x, const Control& u, double dt);
  
  double costFunction(const State& current_state,
                     const State& next_state,
                     const State& state_target,
                     const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  double terminalCost(const State& state, const State& goal);
  
  float calculateCostAtPosition(const mesh_map::Vector& position);
  
  boost::optional<mesh_map::Vector> calculateDirectionAtPosition(
      const mesh_map::Vector& position,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  float computeDirectionCost(const mesh_map::Vector& current,
                            const mesh_map::Vector& next,
                            const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  // Main planning loop
  Trajectory mppiPlanner(const State& start_point,
                        const State& goal_point,
                        int time_horizon,
                        const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  std::tuple<double, double, double> getRPYFromQuaternion(
      const geometry_msgs::msg::Quaternion& q);

  // Member variables
  std::shared_ptr<mesh_map::MeshMap> mesh_map_;
  std::shared_ptr<mppi_mesh_planner::WaveFront> wavefront_;
  std::shared_ptr<mppi_mesh_planner::TerrainDynamics> terrain_dynamics_;
  std::shared_ptr<mppi_mesh_planner::DataCollector> data_collector_;
  
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  
  std::string name_;
  std::string map_frame_;
  std::atomic<bool> cancel_planning_;
  
  bool publish_vector_field_;
  bool publish_face_vectors_;
  bool use_simplified_dynamics_;

  // MPPI parameters
  int num_samples_;
  int num_iterations_;
  int time_horizon_;
  double lambda_;  // Temperature parameter
  double noise_stddev_v_;  // Noise std for linear velocity
  double noise_stddev_w_;  // Noise std for angular velocity
  
  // Control constraints
  double min_velocity_;
  double max_velocity_;
  double min_omega_;
  double max_omega_;
  
  // Cost weights
  double weight_direction_;
  double weight_smoothness_;
  double weight_terminal_;
  
  // Execution parameters
  double control_dt_;
  int max_steps_;
  double goal_threshold_;
  float position_projection_radius_;
};

}  // namespace mppi_mesh_planner

#endif  // MPPI_MESH_PLANNER_H