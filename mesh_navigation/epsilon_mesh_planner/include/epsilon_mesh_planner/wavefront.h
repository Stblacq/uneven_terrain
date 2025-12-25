#ifndef EPSILON_MESH_PLANNER_WAVEFRONT_H
#define EPSILON_MESH_PLANNER_WAVEFRONT_H

#include <lvr2/geometry/Handles.hpp>
#include <lvr2/util/Meap.hpp>
#include <mesh_map/mesh_map.h>
#include <rclcpp/rclcpp.hpp>

namespace epsilon_mesh_planner
{

struct WaveFrontConfig
{
    double cost_limit = 1.0;
    double goal_dist_offset = 0.3;
    
    // Update method selection
    enum class UpdateMethod {
        STANDARD,      
        WITH_S,        
        FMM           
    };
    UpdateMethod update_method = UpdateMethod::STANDARD;
    
    WaveFrontConfig() = default;
};

class WaveFront
{
public:
    explicit WaveFront(const std::shared_ptr<mesh_map::MeshMap>& mesh_map, 
                       const WaveFrontConfig& config = WaveFrontConfig(),
                       rclcpp::Logger logger = rclcpp::get_logger("WaveFront"));
    
    ~WaveFront() = default;
    
    // Main computation method
    bool compute(const mesh_map::Vector& start, 
                 const mesh_map::Vector& goal,
                 std::string& message);
    
    // Getters for results
    const lvr2::DenseVertexMap<mesh_map::Vector>& getVectorMap() const { return vector_map_; }
    const lvr2::DenseVertexMap<float>& getPotential() const { return potential_; }
    const lvr2::DenseVertexMap<lvr2::VertexHandle>& getPredecessors() const { return predecessors_; }
    const lvr2::DenseVertexMap<float>& getDirection() const { return direction_; }

    
    // Cancel computation
    void cancel() { cancel_computation_ = true; }
    
private:
    // Core wavefront propagation
    bool propagate(const mesh_map::Vector& start,
                   const mesh_map::Vector& goal,
                   std::string& message);
    
    bool waveFrontUpdate(lvr2::DenseVertexMap<float>& distances,
                        const lvr2::DenseEdgeMap<float>& edge_weights,
                        const lvr2::VertexHandle& v1,
                        const lvr2::VertexHandle& v2,
                        const lvr2::VertexHandle& v3);
    
    bool waveFrontUpdateWithS(lvr2::DenseVertexMap<float>& distances,
                             const lvr2::DenseEdgeMap<float>& edge_weights,
                             const lvr2::VertexHandle& v1,
                             const lvr2::VertexHandle& v2,
                             const lvr2::VertexHandle& v3);
    
    bool waveFrontUpdateFMM(lvr2::DenseVertexMap<float>& distances,
                           const lvr2::DenseEdgeMap<float>& edge_weights,
                           const lvr2::VertexHandle& v1tmp,
                           const lvr2::VertexHandle& v2tmp,
                           const lvr2::VertexHandle& v3);
    
    // Compute vector field from potential
    void computeVectorMap();
    
    // Member variables
    std::shared_ptr<mesh_map::MeshMap> mesh_map_;
    WaveFrontConfig config_;
    rclcpp::Logger logger_;
    std::atomic_bool cancel_computation_;
    
    // Results
    lvr2::DenseVertexMap<mesh_map::Vector> vector_map_;
    lvr2::DenseVertexMap<float> potential_;
    lvr2::DenseVertexMap<lvr2::VertexHandle> predecessors_;
    lvr2::DenseVertexMap<float> direction_;
    lvr2::DenseVertexMap<lvr2::OptionalFaceHandle> cutting_faces_;
};

} // namespace epsilon_mesh_planner

#endif // EPSILON_MESH_PLANNER_WAVEFRONT_H