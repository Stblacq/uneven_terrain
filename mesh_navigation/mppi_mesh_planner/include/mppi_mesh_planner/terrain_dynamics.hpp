#ifndef MPPI_MESH_PLANNER_TERRAIN_DYNAMICS_HPP
#define MPPI_MESH_PLANNER_TERRAIN_DYNAMICS_HPP

#include <Eigen/Dense>
#include <vector>
#include <array>
#include <memory>
#include <string>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Sphere_3.h>
#include "mppi_mesh_planner/native_gp.hpp"

namespace mppi_mesh_planner {

using Eigen::Vector3d;
using Eigen::MatrixXd;

// CGAL types
typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point;
typedef K::Triangle_3 Triangle;
typedef K::Sphere_3 Sphere;
typedef std::vector<Triangle>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
typedef CGAL::AABB_traits<K, Primitive> AABB_traits;
typedef CGAL::AABB_tree<AABB_traits> Tree;

class TerrainDynamics {
private:
    std::vector<Vector3d> vertices_;
    std::vector<std::array<int,3>> faces_;
    std::vector<Vector3d> face_normals_;
    std::vector<Vector3d> tri_centroids_;
    std::vector<double> face_areas_;
    std::vector<Vector3d> vertex_normals_;
    
    std::vector<double> mean_curv_;
    std::vector<double> gauss_curv_;
    
    std::vector<Triangle> cgal_triangles_;
    std::unique_ptr<Tree> aabb_tree_;

    NativeGP gp_;
    
    double dt_;
    double epsilon_;
    double neighborhood_radius_;
    
public:
    TerrainDynamics(const std::string& mesh_file, 
                   const std::string& curvature_file,
                   const std::string& gp_params_file,
                   double dt = 0.1,
                   double epsilon = 0.5,
                   double neighborhood_radius = 0.16);
    
    ~TerrainDynamics();
    
    std::pair<Vector3d, Vector3d> project_to_mesh(const Vector3d& query);
    double h(double x, double y);
    std::vector<double> query_descriptor(const Vector3d& point);
    std::array<double,6> nominal_dynamics(const std::array<double,6>& state,
                                         const std::array<double,2>& control,
                                         double dt = 0.1);
    std::array<double,6> dynamics(const std::array<double,6>& state,
                                 const std::array<double,2>& control,
                                 double dt = 0.1);
    
private:
    void load_obj(const std::string& filename);
    void load_curvatures(const std::string& filename);
    void compute_face_data();
    void compute_vertex_normals();
    void build_cgal_tree();
    
    std::tuple<Vector3d, double, int> nearest_on_surface(const Vector3d& point);
    Vector3d compute_barycentric(const Vector3d& p, const Vector3d& a,
                                const Vector3d& b, const Vector3d& c);
};

} // namespace mppi_mesh_planner

#endif