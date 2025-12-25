#ifndef EPSILON_MESH_PLANNER_H
#define EPSILON_MESH_PLANNER_H

#include <mbf_mesh_core/mesh_planner.h>
#include <mbf_msgs/action/get_path.hpp>
#include <mesh_map/mesh_map.h>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <vector>
#include <array>
#include <memory>
#include <optional>
#include "epsilon_mesh_planner/wavefront.h"
#include "epsilon_mesh_planner/terrain_dynamics.hpp"
#include "epsilon_mesh_planner/data_collector.h"

namespace epsilon_mesh_planner
{

using State = std::array<double, 6>;  // [x, y, z, roll, pitch, yaw]
using Control = std::array<double, 2>;  // [linear_velocity, angular_velocity]
using ControlSequence = std::vector<Control>;
using Trajectory = std::vector<State>;

// Terrain descriptor structure (matches precomputed descriptors)
struct TerrainDescriptor {
  double n_x;           // Normal x component
  double n_y;           // Normal y component
  double n_z;           // Normal z component (vertical)
  double roughness;     // Local roughness (σz)
  double gaussian_curv; // Gaussian curvature (K)
  double mean_curv;     // Mean curvature (H)
  double total_area;    // Total area
  double inclination;   // Computed from n_z: acos(n_z)
  
  mesh_map::Vector getNormal() const {
    return mesh_map::Vector(n_x, n_y, n_z);
  }
};

class EpsilonMeshPlanner : public mbf_mesh_core::MeshPlanner
{
public:
  EpsilonMeshPlanner();
  virtual ~EpsilonMeshPlanner();

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
  // Core epsilon-constraint optimization
  ControlSequence epsilonConstraintOptimization(
      const State& state_init,
      const State& state_target,
      const std::optional<ControlSequence>& prev_best_controls,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  std::vector<ControlSequence> generateCandidateTrajectories(
      const State& current_state,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map,
      const std::optional<ControlSequence>& prev_best_controls);

  Trajectory simulateForward(
      const State& state_init,
      const ControlSequence& control_sequence);

  std::vector<TerrainDescriptor> extractTerrainDescriptors(
      const Trajectory& trajectory);

  TerrainDescriptor getTerrainDescriptor(const mesh_map::Vector& position);

  // Adaptive epsilon computation (NEW)
  double computeAdaptiveEpsilon(const TerrainDescriptor& current_terrain_descriptor);

  // Objective functions (as defined in paper)
  double objectiveF1(
      const Trajectory& trajectory,
      const State& state_target,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  double objectiveF2(
      const Trajectory& trajectory,
      const std::vector<TerrainDescriptor>& descriptors);

  bool checkHardConstraints(
      const Trajectory& trajectory,
      const std::vector<TerrainDescriptor>& descriptors);

  // Stability violation for F2 (terrain-relative)
  double stabilityViolation(double delta_angle);
  
  // Compute terrain-aligned orientation from terrain normal
  std::pair<double, double> computeTerrainAlignedOrientation(
      const TerrainDescriptor& descriptor);

  // Dynamics and helpers
  State dynamics(const State& x, const Control& u, double dt);

  boost::optional<mesh_map::Vector> calculateDirectionAtPosition(
      const mesh_map::Vector& position,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  float computeDirectionCost(const mesh_map::Vector& current,
                            const mesh_map::Vector& next,
                            const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  // Main planning loop
  Trajectory epsilonPlanner(const State& start_point,
                           const State& goal_point,
                           int time_horizon,
                           const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);

  std::tuple<double, double, double> getRPYFromQuaternion(
      const geometry_msgs::msg::Quaternion& q);

  // Member variables
  std::shared_ptr<mesh_map::MeshMap> mesh_map_;
  std::shared_ptr<epsilon_mesh_planner::WaveFront> wavefront_;
  std::shared_ptr<epsilon_mesh_planner::TerrainDynamics> terrain_dynamics_;
  std::shared_ptr<epsilon_mesh_planner::DataCollector> data_collector_;

  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;

  std::string name_;
  std::string map_frame_;
  std::atomic<bool> cancel_planning_;

  bool publish_vector_field_;
  bool publish_face_vectors_;
  bool use_simplified_dynamics_;

  // Epsilon-constraint parameters
  int num_candidates_;
  double epsilon_base_;           // Base stability budget (adjusted adaptively based on terrain)
  double alpha_safe_;             // Safe deviation from terrain-normal (radians)
  double alpha_max_;              // Maximum deviation before guaranteed tip-over (radians)
  double noise_stddev_v_;
  double noise_stddev_w_;
  double fmm_bias_strength_;      // How strongly to bias sampling toward FMM (0.0 to 1.0)

  // Adaptive epsilon parameters (NEW)
  double expected_max_roughness_;        // Computed from loaded descriptors (max roughness in terrain)
  double expected_max_inclination_;      // Computed from loaded descriptors (max slope in terrain)
  double roughness_adaptation_strength_; // Weight for roughness adaptation (0.0 to 1.0) - configured
  double slope_adaptation_strength_;     // Weight for slope adaptation (0.0 to 1.0) - configured

  // Control constraints
  double min_velocity_;
  double max_velocity_;
  double min_omega_;
  double max_omega_;

  // Objective F2 parameters (tip-over stability)
  double kappa_;                  // Penalty coefficient for tilt violations
  double s_exp_;                  // Goal proximity scaling exponent
  double max_roll_;               // alpha_safe for roll deviation from terrain-normal
  double max_pitch_;              // alpha_safe for pitch deviation from terrain-normal

  // Execution parameters
  int time_horizon_;
  double control_dt_;
  int max_steps_;
  double goal_threshold_;
  float position_projection_radius_;

  // Warm start
  std::optional<ControlSequence> prev_best_controls_;
  
  // Precomputed vertex descriptors (8D: n_x, n_y, n_z, roughness, K, H, area, reserved)
  std::vector<std::array<double, 8>> vertex_descriptors_;
};

}  // namespace epsilon_mesh_planner

#endif  // EPSILON_MESH_PLANNER_H