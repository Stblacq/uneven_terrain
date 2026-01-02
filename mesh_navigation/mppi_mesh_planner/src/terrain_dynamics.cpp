#include "mppi_mesh_planner/terrain_dynamics.hpp"
#include <cnpy.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

namespace mppi_mesh_planner {

TerrainDynamics::TerrainDynamics(const std::string& mesh_file,
                                 const std::string& curvature_file,
                                 const std::string& gp_params_file,
                                 double dt,
                                 double epsilon,
                                 double neighborhood_radius)
    : dt_(dt), epsilon_(epsilon), neighborhood_radius_(neighborhood_radius) {
    
    std::cout << "=== Initializing TerrainDynamics with CGAL ===" << std::endl;
    
    std::cout << "\n1. Loading mesh..." << std::endl;
    load_obj(mesh_file);
    std::cout << "   Loaded " << vertices_.size() << " vertices, " 
              << faces_.size() << " faces" << std::endl;
    
    std::cout << "\n2. Computing face data..." << std::endl;
    compute_face_data();
    
    std::cout << "\n3. Computing vertex normals..." << std::endl;
    compute_vertex_normals();
    
    std::cout << "\n4. Loading curvatures..." << std::endl;
    load_curvatures(curvature_file);
    std::cout << "   Loaded " << mean_curv_.size() << " curvature values" << std::endl;
    
    std::cout << "\n5. Building CGAL AABB tree..." << std::endl;
    build_cgal_tree();
    
    std::cout << "\n6. Loading native GP model..." << std::endl;
    gp_.load_from_npz(gp_params_file);
    
    std::cout << "\n✓ TerrainDynamics initialized successfully!" << std::endl;
    std::cout << "=========================================\n" << std::endl;
}

TerrainDynamics::~TerrainDynamics() {
}

void TerrainDynamics::load_obj(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open mesh file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string type;
        ss >> type;
        
        if (type == "v") {
            double x, y, z;
            ss >> x >> y >> z;
            vertices_.push_back(Vector3d(x, y, z));
        } else if (type == "f") {
            std::string v1, v2, v3;
            ss >> v1 >> v2 >> v3;
            
            auto parse_vertex = [](const std::string& s) {
                size_t pos = s.find('/');
                int idx = std::stoi(pos != std::string::npos ? s.substr(0, pos) : s);
                return idx - 1;
            };
            
            int a = parse_vertex(v1);
            int b = parse_vertex(v2);
            int c = parse_vertex(v3);
            faces_.push_back({a, b, c});
        }
    }
}

void TerrainDynamics::load_curvatures(const std::string& filename) {
    cnpy::npz_t npz = cnpy::npz_load(filename);
    
    auto mc_arr = npz["mean_curv"];
    auto gc_arr = npz["gauss_curv"];
    
    double* mc_data = mc_arr.data<double>();
    double* gc_data = gc_arr.data<double>();
    
    size_t n = mc_arr.shape[0];
    mean_curv_.assign(mc_data, mc_data + n);
    gauss_curv_.assign(gc_data, gc_data + n);
}

void TerrainDynamics::compute_face_data() {
    for (const auto& face : faces_) {
        const Vector3d& v0 = vertices_[face[0]];
        const Vector3d& v1 = vertices_[face[1]];
        const Vector3d& v2 = vertices_[face[2]];
        
        Vector3d edge1 = v1 - v0;
        Vector3d edge2 = v2 - v0;
        Vector3d normal = edge1.cross(edge2);
        double area = normal.norm() / 2.0;
        normal.normalize();
        
        face_normals_.push_back(normal);
        face_areas_.push_back(area);
        
        Vector3d centroid = (v0 + v1 + v2) / 3.0;
        tri_centroids_.push_back(centroid);
    }
}

void TerrainDynamics::compute_vertex_normals() {
    vertex_normals_.resize(vertices_.size(), Vector3d::Zero());
    
    for (size_t i = 0; i < faces_.size(); ++i) {
        const auto& face = faces_[i];
        const Vector3d& normal = face_normals_[i];
        double area = face_areas_[i];
        
        for (int j = 0; j < 3; ++j) {
            vertex_normals_[face[j]] += normal * area;
        }
    }
    
    for (auto& n : vertex_normals_) {
        n.normalize();
    }
}

void TerrainDynamics::build_cgal_tree() {
    // Build CGAL triangles
    cgal_triangles_.clear();
    for (const auto& face : faces_) {
        const Vector3d& v0 = vertices_[face[0]];
        const Vector3d& v1 = vertices_[face[1]];
        const Vector3d& v2 = vertices_[face[2]];
        
        Point p0(v0.x(), v0.y(), v0.z());
        Point p1(v1.x(), v1.y(), v1.z());
        Point p2(v2.x(), v2.y(), v2.z());
        
        cgal_triangles_.push_back(Triangle(p0, p1, p2));
    }
    
    // Build AABB tree
    aabb_tree_ = std::make_unique<Tree>(cgal_triangles_.begin(), cgal_triangles_.end());
    aabb_tree_->accelerate_distance_queries();
    
    std::cout << "   Built CGAL AABB tree with " << cgal_triangles_.size() << " triangles" << std::endl;
}

std::tuple<Vector3d, double, int> TerrainDynamics::nearest_on_surface(const Vector3d& point) {
    Point query(point.x(), point.y(), point.z());
    
    // Query closest point
    Point closest_pt = aabb_tree_->closest_point(query);
    
    // Find which triangle it's on
    auto hint = aabb_tree_->closest_point_and_primitive(query);
    int triangle_id = std::distance(cgal_triangles_.begin(), hint.second);
    
    Vector3d cp(closest_pt.x(), closest_pt.y(), closest_pt.z());
    double dist = std::sqrt(CGAL::squared_distance(query, closest_pt));
    
    return {cp, dist, triangle_id};
}

Vector3d TerrainDynamics::compute_barycentric(const Vector3d& p,
                                              const Vector3d& a,
                                              const Vector3d& b,
                                              const Vector3d& c) {
    Vector3d v0 = b - a;
    Vector3d v1 = c - a;
    Vector3d v2 = p - a;
    
    double d00 = v0.dot(v0);
    double d01 = v0.dot(v1);
    double d11 = v1.dot(v1);
    double d20 = v2.dot(v0);
    double d21 = v2.dot(v1);
    
    double denom = d00 * d11 - d01 * d01;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;
    
    return Vector3d(u, v, w);
}

std::pair<Vector3d, Vector3d> TerrainDynamics::project_to_mesh(const Vector3d& query) {
    auto [closest_point, dist, tri_idx] = nearest_on_surface(query);
    Vector3d normal = face_normals_[tri_idx];
    return {closest_point, normal};
}

double TerrainDynamics::h(double x, double y) {
    auto [cp, normal] = project_to_mesh(Vector3d(x, y, 0));
    return cp.z();
}

std::vector<double> TerrainDynamics::query_descriptor(const Vector3d& point) {
    auto [cp, normal] = project_to_mesh(point);
    
    // Use CGAL AABB tree to find nearby triangles efficiently (O(log N))
    Point query_pt(cp.x(), cp.y(), cp.z());
    
    std::vector<int> indices;
    std::vector<double> distances;
    std::vector<double> areas;
    
    // Create sphere for spatial query (note: CGAL uses squared radius)
    Sphere search_sphere(query_pt, neighborhood_radius_ * neighborhood_radius_);
    
    // Find all triangles intersecting the sphere
    std::vector<Tree::Primitive_id> intersected_primitives;
    aabb_tree_->all_intersected_primitives(search_sphere, std::back_inserter(intersected_primitives));
    
    // Process intersected triangles and filter by centroid distance
    for (const auto& prim_id : intersected_primitives) {
        int tri_idx = std::distance(cgal_triangles_.begin(), prim_id);
        
        // Check distance from query point to triangle centroid
        const Vector3d& centroid = tri_centroids_[tri_idx];
        double dist = (centroid - cp).norm();
        
        if (dist <= neighborhood_radius_) {
            indices.push_back(tri_idx);
            distances.push_back(dist);
            areas.push_back(face_areas_[tri_idx]);
        }
    }
    
    if (indices.empty()) {
        return {normal.x(), normal.y(), normal.z(), 0.0, 0.0, 0.0, 0.0, (point - cp).norm()};
    }
    
    // Compute weighted normal (area and distance weighted, matching Python)
    Vector3d weighted_normal = Vector3d::Zero();
    double total_weight = 0.0;
    
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        // Python: weights = nearby_areas / (dists + 1e-6)
        double weight = areas[i] / (distances[i] + 1e-6);
        weighted_normal += face_normals_[idx] * weight;
        total_weight += weight;
    }
    
    if (total_weight > 1e-12) {
        weighted_normal /= total_weight;
        weighted_normal.normalize();
    } else {
        weighted_normal = normal;
    }
    
    // Height variation along weighted normal (matching Python)
    std::vector<double> heights;
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        double h = (tri_centroids_[idx] - cp).dot(weighted_normal);
        heights.push_back(h);
    }
    
    double sigma_z = 0.0;
    if (heights.size() > 1) {
        double mean_h = std::accumulate(heights.begin(), heights.end(), 0.0) / heights.size();
        for (double h : heights) {
            sigma_z += (h - mean_h) * (h - mean_h);
        }
        sigma_z = std::sqrt(sigma_z / heights.size());
    }
    
    // Get curvatures at the closest point using barycentric interpolation
    // Find which triangle we're on
    auto [cp2, dist, tri_idx] = nearest_on_surface(point);
    const auto& face = faces_[tri_idx];
    
    // Get barycentric coordinates (simplified - assume we're at cp already)
    const Vector3d& v0 = vertices_[face[0]];
    const Vector3d& v1 = vertices_[face[1]];
    const Vector3d& v2 = vertices_[face[2]];
    Vector3d bary = compute_barycentric(cp, v0, v1, v2);
    
    // Clamp and normalize barycentric coordinates
    bary = bary.cwiseMax(0.0).cwiseMin(1.0);
    double bary_sum = bary.sum();
    if (bary_sum > 0) {
        bary /= bary_sum;
    } else {
        bary = Vector3d(1.0/3.0, 1.0/3.0, 1.0/3.0);
    }
    
    // Interpolate curvatures using barycentric coordinates
    double K = bary[0] * gauss_curv_[face[0]] + 
               bary[1] * gauss_curv_[face[1]] + 
               bary[2] * gauss_curv_[face[2]];
    
    double H = bary[0] * mean_curv_[face[0]] + 
               bary[1] * mean_curv_[face[1]] + 
               bary[2] * mean_curv_[face[2]];
    
    // Total area in neighborhood
    double A_total = std::accumulate(areas.begin(), areas.end(), 0.0);
    
    // Distance from query to mesh
    double distance = (point - cp).norm();
    
    return {
        weighted_normal.x(), weighted_normal.y(), weighted_normal.z(),
        sigma_z, K, H, A_total, distance
    };
}

std::array<double,6> TerrainDynamics::nominal_dynamics(
    const std::array<double,6>& state,
    const std::array<double,2>& control,
    double dt) {
    
    double x = state[0];
    double y = state[1];
    double z = state[2];
    double phi = state[3];
    double theta = state[4];
    double psi = state[5];
    
    double u0 = control[0];
    double u1 = control[1];
    
    // Position update
    double x1 = x + u0 * std::cos(theta) * std::cos(psi) * dt;
    double y1 = y + u0 * std::cos(theta) * std::sin(psi) * dt;
    double z1 = h(x1, y1);
    
    // Gradient computation
    double h_x_plus = h(x1 + epsilon_, y1);
    double h_x_minus = h(x1 - epsilon_, y1);
    double h_y_plus = h(x1, y1 + epsilon_);
    double h_y_minus = h(x1, y1 - epsilon_);
    
    double delta_x = (h_x_plus - h_x_minus) / (2.0 * epsilon_);
    double delta_y = (h_y_plus - h_y_minus) / (2.0 * epsilon_);
    
    // Orientation update
    double phi1 = std::atan2(-delta_y, std::sqrt(1.0 + delta_x * delta_x));
    double theta1 = std::atan2(delta_x, std::sqrt(1.0 + delta_y * delta_y));
    double psi1 = psi + u1 * dt;
    
    return {x1, y1, z1, phi1, theta1, psi1};
}

std::array<double,6> TerrainDynamics::dynamics(
    const std::array<double,6>& state,
    const std::array<double,2>& control,
    double dt) {
    
    // Get nominal prediction WITH dt
    auto nominal = nominal_dynamics(state, control, dt);
    
    // Get terrain descriptor
    Vector3d position(state[0], state[1], state[2]);
    auto descriptor = query_descriptor(position);
    
    // Build GP input: [state(6), control(2), descriptor(8)] = 16 features
    Eigen::VectorXd gp_input(16);
    for (int i = 0; i < 6; ++i) gp_input[i] = state[i];
    for (int i = 0; i < 2; ++i) gp_input[6 + i] = control[i];
    for (int i = 0; i < 8; ++i) gp_input[8 + i] = descriptor[i];
    
    // Predict residual
    Eigen::VectorXd residual = gp_.predict_mean(gp_input);
    
    // Apply correction
    std::array<double,6> corrected;
    for (int i = 0; i < 6; ++i) {
        corrected[i] = nominal[i] + residual[i];
    }
    
    return corrected;
}

} // namespace mppi_mesh_planner