#ifndef GAKD_MESH_PLANNER_TERRAIN_DYNAMICS_HPP
#define GAKD_MESH_PLANNER_TERRAIN_DYNAMICS_HPP

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <nanoflann.hpp>
#include <tensorflow/c/c_api.h>

namespace gakd_mesh_planner {

using Eigen::Vector3d;
using Eigen::MatrixXd;

class TerrainDynamics {
private:
    // Mesh data
    std::vector<Vector3d> vertices_;
    std::vector<std::array<int,3>> faces_;
    std::vector<Vector3d> face_normals_;
    std::vector<Vector3d> tri_centroids_;
    std::vector<double> face_areas_;
    std::vector<Vector3d> vertex_normals_;
    
    // Curvature data
    std::vector<double> mean_curv_;
    std::vector<double> gauss_curv_;
    
  // Simple data structure for nanoflann
    struct PointCloud {
        std::vector<std::array<double, 3>> pts;
        
        inline size_t kdtree_get_point_count() const { return pts.size(); }
        
        inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
            return pts[idx][dim];
        }
        
        template <class BBOX>
        bool kdtree_get_bbox(BBOX&) const { return false; }
        
        // Required by some nanoflann versions for L2 distance computation
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
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloud>,
        PointCloud,
        3 /* dimensions */
    >;
    std::unique_ptr<KDTree> kdtree_;

    // TensorFlow C API objects
    TF_Graph* graph_;
    TF_Session* session_;
    TF_Status* status_;
    TF_Output input_op_;
    TF_Output output_op_;
    
    // OPTIMIZATION: Persistent tensor buffers to avoid allocation overhead
    TF_Tensor* input_tensor_;
    double* input_buffer_;
    bool tensors_initialized_;
    
    // Parameters
    double dt_;
    double epsilon_;
    double neighborhood_radius_;
    
public:
    TerrainDynamics(const std::string& mesh_file, 
                   const std::string& curvature_file,
                   const std::string& model_dir,
                   double dt = 0.1,
                   double epsilon = 0.5,
                   double neighborhood_radius = 0.24);
    
    ~TerrainDynamics();
    
    // Project point to mesh
    std::pair<Vector3d, Vector3d> project_to_mesh(const Vector3d& query);
    
    // Height function
    double h(double x, double y);
    
    // Query descriptor at point
    // Returns 8-dimensional descriptor: [nx, ny, nz, sigma_z, K, H, A_total, distance_from_mesh]
    //   - nx, ny, nz: Weighted surface normal
    //   - sigma_z: Height variation (roughness)
    //   - K: Gaussian curvature
    //   - H: Mean curvature
    //   - A_total: Total area of nearby triangles
    //   - distance_from_mesh: Distance from query point to nearest surface point
    std::vector<double> query_descriptor(const Vector3d& point);
    
    // Nominal dynamics
    std::array<double,6> nominal_dynamics(const std::array<double,6>& state,
                                         const std::array<double,2>& control);
    
    // Full dynamics with GP correction
    // Input to TensorFlow model: [state(6), control(2), descriptor(8)] = 16 features
    // Output from TensorFlow model: residual corrections for state (6 values)
    std::array<double,6> dynamics(const std::array<double,6>& state,
                                 const std::array<double,2>& control);
    
private:
    void load_obj(const std::string& filename);
    void load_curvatures(const std::string& filename);
    void compute_face_data();
    void compute_vertex_normals();
    void build_kdtree();
    void load_tensorflow_model(const std::string& model_dir);
    void initialize_tensors();
    void cleanup_tensors();
    
    Vector3d compute_barycentric(const Vector3d& p, 
                                const Vector3d& a, 
                                const Vector3d& b, 
                                const Vector3d& c);
    
    std::tuple<Vector3d, double, int> nearest_on_surface(const Vector3d& point);
};

} // namespace gakd_mesh_planner

#endif