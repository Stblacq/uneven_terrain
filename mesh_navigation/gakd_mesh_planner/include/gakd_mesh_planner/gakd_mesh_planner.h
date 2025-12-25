#ifndef MESH_NAVIGATION__MESH_PLANNER_H
#define MESH_NAVIGATION__MESH_PLANNER_H

#include <mbf_mesh_core/mesh_planner.h>
#include <mbf_msgs/action/get_path.hpp>
#include <mesh_map/mesh_map.h>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include "gakd_mesh_planner/data_collector.h"
#include "gakd_mesh_planner/wavefront.h"
#include "gakd_mesh_planner/terrain_dynamics.hpp"

namespace gakd_mesh_planner
{

using State = std::vector<double>;
using Control = std::array<double, 2>;
using ControlSequence = std::vector<Control>;
using Trajectory = std::vector<State>;

class GAKDMeshPlanner : public mbf_mesh_core::MeshPlanner
{
public:
  typedef std::shared_ptr<gakd_mesh_planner::GAKDMeshPlanner> Ptr;

  GAKDMeshPlanner();
  virtual ~GAKDMeshPlanner();

  virtual uint32_t makePlan(const geometry_msgs::msg::PoseStamped& start, 
                           const geometry_msgs::msg::PoseStamped& goal,
                           double tolerance, 
                           std::vector<geometry_msgs::msg::PoseStamped>& plan, 
                           double& cost,
                           std::string& message) override;

  virtual bool cancel() override;

  virtual bool initialize(const std::string& plugin_name, 
                         const std::shared_ptr<mesh_map::MeshMap>& mesh_map_ptr,
                         const rclcpp::Node::SharedPtr& node) override;

private:
  mesh_map::MeshMap::Ptr mesh_map_;
  std::string name_;
  rclcpp::Node::SharedPtr node_;
  std::atomic_bool cancel_planning_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  std::string map_frame_;
  std::shared_ptr<DataCollector> data_collector_;
  std::shared_ptr<WaveFront> wavefront_;
  std::shared_ptr<TerrainDynamics> terrain_dynamics_;
  bool publish_vector_field_;
  bool publish_face_vectors_;
  
  // Dynamics model
  bool use_simplified_dynamics_;
  
  // Control constraints
  double min_velocity_;
  double max_velocity_;
  double min_omega_;
  double max_omega_;
  
  // GA parameters
  int population_size_;
  int num_generations_;
  int time_horizon_;
  double mutation_rate_;
  int tournament_size_;
  double mutation_stddev_;
  
  // Execution parameters
  double control_dt_;
  int max_steps_;
  double goal_threshold_;
  
  // Mesh parameters
  float position_projection_radius_;

  State dynamics(const State& x, const Control& u, double dt);
  
  Control randomControl();
  std::vector<ControlSequence> generatePopulation(int size, int horizon);
  std::pair<ControlSequence, ControlSequence> selectParents(
      const std::vector<ControlSequence>& population,
      const std::vector<double>& fitnesses);
  std::pair<ControlSequence, ControlSequence> crossover(
      const ControlSequence& parent1, 
      const ControlSequence& parent2);
  ControlSequence mutate(ControlSequence control_sequence, double mutation_rate);
  
  float calculateCostAtPosition(const mesh_map::Vector& position);
  boost::optional<mesh_map::Vector> calculateDirectionAtPosition(
      const mesh_map::Vector& position,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  float computeDirectionCost(const mesh_map::Vector& current, 
                             const mesh_map::Vector& next,
                             const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  
  double costFunction(const State& current_state, 
                     const State& next_state,
                     const State& state_target,
                     const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  
  ControlSequence geneticAlgorithm(const State& state_init, 
                                   const State& state_target,
                                   double dt, 
                                   int pop_size, 
                                   int generations,
                                   int time_horizon, 
                                   double mutation_rate,
                                   const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  
  Trajectory gaPlanner(const State& start_point, 
                      const State& goal_point,
                      double time_horizon,
                      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  
  std::tuple<double, double, double> getRPYFromQuaternion(
      const geometry_msgs::msg::Quaternion& q);
};

}  // namespace gakd_mesh_planner

#endif  // MESH_NAVIGATION__MESH_PLANNER_H