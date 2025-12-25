#ifndef NSGA2_MESH_PLANNER_TERRAIN_DYNAMICS_HPP
#define NSGA2_MESH_PLANNER_TERRAIN_DYNAMICS_HPP

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <nanoflann.hpp>

namespace nsga2_mesh_planner {

using Eigen::Vector3d;
using Eigen::MatrixXd;

class TerrainDynamics {
private:
    // Mesh data
    std::vector<Vector3d> vertices_;
    std::vector<std::array<int,3>> faces_;
    std::vector<Vector3d> face_normals_;
    std::vector<Vector3d> tri_centroids_;
    
    // GP corrections precomputed at vertices (6D: x, y, z, roll, pitch, yaw)
    std::vector<std::array<double, 6>> vertex_corrections_;
    
    // Terrain descriptors precomputed at vertices (8D: n_x, n_y, n_z, roughness, K, H, area, reserved)
    std::vector<std::array<double, 8>> vertex_descriptors_;
    
    // Simple data structure for nanoflann
    struct PointCloud {
        std::vector<std::array<double, 3>> pts;
        inline size_t kdtree_get_point_count() const { return pts.size(); }
        inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
            return pts[idx][dim];
        }
        template <class BBOX>
        bool kdtree_get_bbox(BBOX&) const { return false; }
        inline double kdtree_distance(const double *p1, const size_t idx_p2, size_t size) const {
            double dist = 0.0;
            for (size_t i = 0; i < size; ++i) {
                const double d = p1[i] - pts[idx_p2][i];
                dist += d * d;
            }
            return dist;
        }
    };
    
    PointCloud cloud_;
    PointCloud vertex_cloud_;  // KD-tree over vertices for fast correction lookup
    
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloud>,
        PointCloud,
        3,
        size_t
    > KDTree;
    
    std::unique_ptr<KDTree> kdtree_;
    std::unique_ptr<KDTree> vertex_kdtree_;  // For O(log n) nearest vertex lookup
    
    // Parameters
    double dt_;
    double epsilon_;
    
    // Flag to indicate if descriptors are loaded
    bool has_descriptors_;

public:
    TerrainDynamics(const std::string& mesh_file, 
                    const std::string& corrections_file,
                    double dt = 0.1,
                    double epsilon = 0.5);
    
    TerrainDynamics(const std::string& mesh_file, 
                    const std::string& corrections_file,
                    const std::string& descriptors_file,
                    double dt = 0.1,
                    double epsilon = 0.5);
    
    ~TerrainDynamics() = default;
    
    // Full dynamics with GP correction
    std::array<double,6> dynamics(const std::array<double,6>& state,
                                  const std::array<double,2>& control);
    
    // Get terrain descriptor at position (fast O(log n) lookup via KD-tree)
    std::array<double, 8> getDescriptorAtPosition(double x, double y, double z);
    
    // Check if descriptors are available
    bool hasDescriptors() const { return has_descriptors_; }

private:
    void load_obj(const std::string& filename);
    void load_vertex_corrections(const std::string& filename);
    void load_vertex_descriptors(const std::string& filename);
    void compute_face_data();
    void build_kdtree();
    
    double h(double x, double y);
    Vector3d compute_barycentric(const Vector3d& p, 
                                 const Vector3d& a, 
                                 const Vector3d& b, 
                                 const Vector3d& c);
    std::tuple<Vector3d, double, int> nearest_on_surface(const Vector3d& point);
    std::array<double, 6> interpolate_correction(const Vector3d& position);
    std::array<double,6> nominal_dynamics(const std::array<double,6>& state,
                                          const std::array<double,2>& control);
};

} // namespace nsga2_mesh_planner

#endif