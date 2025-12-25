#ifndef NSGA2_MESH_PLANNER_H
#define NSGA2_MESH_PLANNER_H

#include <mbf_mesh_core/mesh_planner.h>
#include <mbf_msgs/action/get_path.hpp>
#include <mesh_map/mesh_map.h>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <lvr2/geometry/Handles.hpp>
#include <vector>
#include <array>
#include <memory>
#include <optional>
#include "nsga2_mesh_planner/data_collector.h"
#include "nsga2_mesh_planner/wavefront.h"
#include "nsga2_mesh_planner/terrain_dynamics.hpp"

namespace nsga2_mesh_planner
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

// Individual in NSGA-II population
struct Individual {
  ControlSequence controls;
  Trajectory trajectory;
  std::vector<TerrainDescriptor> descriptors;
  
  // Objectives
  double f1;  // Path optimality (FMM-based cost)
  double f2;  // Tip-over stability violation
  
  // NSGA-II attributes
  int rank;
  double crowding_distance;
  int domination_count;
  std::vector<int> dominated_solutions;
  
  bool violates_hard_constraints;
  
  Individual() : f1(0), f2(0), rank(0), crowding_distance(0), 
                 domination_count(0), violates_hard_constraints(false) {}
};

class NSGA2MeshPlanner : public mbf_mesh_core::MeshPlanner
{
public:
  typedef std::shared_ptr<nsga2_mesh_planner::NSGA2MeshPlanner> Ptr;

  NSGA2MeshPlanner();
  virtual ~NSGA2MeshPlanner();

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
  // Core NSGA-II algorithm
  std::vector<Individual> nsga2Optimization(
      const State& state_init,
      const State& state_target,
      const std::optional<ControlSequence>& prev_best_controls,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  
  // Population initialization
  std::vector<Individual> initializePopulation(
      const State& current_state,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map,
      const std::optional<ControlSequence>& prev_best_controls);
  
  // Fitness evaluation
  void evaluateIndividual(
      Individual& ind,
      const State& state_init,
      const State& state_target,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  
  // Trajectory simulation
  Trajectory simulateForward(
      const State& state_init,
      const ControlSequence& control_sequence);
  
  std::vector<TerrainDescriptor> extractTerrainDescriptors(
      const Trajectory& trajectory);
  
  TerrainDescriptor getTerrainDescriptor(const mesh_map::Vector& position);
  
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
  
  // NSGA-II operators
  void fastNonDominatedSort(std::vector<Individual>& population);
  void calculateCrowdingDistance(std::vector<Individual>& front);
  bool dominates(const Individual& a, const Individual& b);
  
  // Genetic operators
  std::vector<Individual> tournamentSelection(
      const std::vector<Individual>& population,
      int tournament_size,
      int num_parents);
  
  std::pair<ControlSequence, ControlSequence> uniformCrossover(
      const ControlSequence& parent1,
      const ControlSequence& parent2);
  
  ControlSequence gaussianMutation(
      const ControlSequence& controls,
      double mutation_rate,
      double mutation_stddev);
  
  // Solution selection from Pareto front
  int selectSolutionFromParetoFront(
      const std::vector<Individual>& pareto_front,
      const State& state_target);
  
  // Helper functions
  State dynamics(const State& x, const Control& u, double dt);
  
  Control randomControl();
  
  Control fmmGuidedControl(
      const mesh_map::Vector& position,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  
  boost::optional<mesh_map::Vector> calculateDirectionAtPosition(
      const mesh_map::Vector& position,
      const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  
  float computeDirectionCost(const mesh_map::Vector& current, 
                            const mesh_map::Vector& next,
                            const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  
  // Main planning loop
  Trajectory nsga2Planner(const State& start_point, 
                         const State& goal_point,
                         int time_horizon,
                         const lvr2::DenseVertexMap<mesh_map::Vector>& vector_map);
  
  std::tuple<double, double, double> getRPYFromQuaternion(
      const geometry_msgs::msg::Quaternion& q);

  // Member variables
  mesh_map::MeshMap::Ptr mesh_map_;
  std::shared_ptr<DataCollector> data_collector_;
  std::shared_ptr<WaveFront> wavefront_;
  std::shared_ptr<TerrainDynamics> terrain_dynamics_;
  
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  
  std::string name_;
  std::string map_frame_;
  std::atomic_bool cancel_planning_;
  
  bool publish_vector_field_;
  bool publish_face_vectors_;
  bool use_simplified_dynamics_;
  
  // Control constraints
  double min_velocity_;
  double max_velocity_;
  double min_omega_;
  double max_omega_;
  
  // NSGA-II parameters
  int population_size_;
  int num_generations_;
  int time_horizon_;
  double mutation_rate_;
  int tournament_size_;
  double mutation_stddev_;
  double fmm_bias_strength_;  // Fraction of population initialized with FMM guidance
  
  // Stability parameters (for F2)
  double alpha_safe_;        // Safe deviation from terrain-normal (radians)
  double alpha_max_;         // Maximum deviation before guaranteed tip-over (radians)
  double kappa_;             // Penalty coefficient for tilt violations
  double s_exp_;             // Goal proximity scaling exponent
  
  // Execution parameters
  double control_dt_;
  int max_steps_;
  double goal_threshold_;
  float position_projection_radius_;
  
  // Warm start
  std::optional<ControlSequence> prev_best_controls_;
  
  // Precomputed vertex descriptors (8D: n_x, n_y, n_z, roughness, K, H, area, inclination)
  std::vector<std::array<double, 8>> vertex_descriptors_;
};

}  // namespace nsga2_mesh_planner

#endif  // NSGA2_MESH_PLANNER_H