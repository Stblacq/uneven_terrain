#include "epsilon_mesh_planner/terrain_dynamics.hpp"
#include <cnpy.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

namespace epsilon_mesh_planner {

TerrainDynamics::TerrainDynamics(const std::string& mesh_file,
                                 const std::string& corrections_file,
                                 double dt,
                                 double epsilon)
    : dt_(dt), epsilon_(epsilon), has_descriptors_(false) {
    
    load_obj(mesh_file);
    compute_face_data();
    load_vertex_corrections(corrections_file);
    build_kdtree();
}

TerrainDynamics::TerrainDynamics(const std::string& mesh_file,
                                 const std::string& corrections_file,
                                 const std::string& descriptors_file,
                                 double dt,
                                 double epsilon)
    : dt_(dt), epsilon_(epsilon), has_descriptors_(false) {
    
    load_obj(mesh_file);
    compute_face_data();
    load_vertex_corrections(corrections_file);
    load_vertex_descriptors(descriptors_file);
    build_kdtree();
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
    
    std::cout << "Loaded mesh: " << vertices_.size() << " vertices, " 
              << faces_.size() << " faces" << std::endl;
}

void TerrainDynamics::load_vertex_corrections(const std::string& filename) {
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    
    if (arr.shape.size() != 2 || arr.shape[1] != 6) {
        throw std::runtime_error("Invalid corrections file shape. Expected (N, 6)");
    }
    
    size_t n_vertices = arr.shape[0];
    if (n_vertices != vertices_.size()) {
        throw std::runtime_error("Corrections file has " + std::to_string(n_vertices) + 
                                 " entries but mesh has " + std::to_string(vertices_.size()) + " vertices");
    }
    
    double* data = arr.data<double>();
    vertex_corrections_.resize(n_vertices);
    
    for (size_t i = 0; i < n_vertices; ++i) {
        for (int j = 0; j < 6; ++j) {
            vertex_corrections_[i][j] = data[i * 6 + j];
        }
    }
    
    std::cout << "Loaded " << n_vertices << " vertex corrections" << std::endl;
}

void TerrainDynamics::load_vertex_descriptors(const std::string& filename) {
    try {
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        
        if (arr.shape.size() != 2 || arr.shape[1] != 8) {
            throw std::runtime_error("Invalid descriptors file shape. Expected (N, 8)");
        }
        
        size_t n_vertices = arr.shape[0];
        if (n_vertices != vertices_.size()) {
            throw std::runtime_error("Descriptors file has " + std::to_string(n_vertices) + 
                                     " entries but mesh has " + std::to_string(vertices_.size()) + " vertices");
        }
        
        double* data = arr.data<double>();
        vertex_descriptors_.resize(n_vertices);
        
        for (size_t i = 0; i < n_vertices; ++i) {
            for (int j = 0; j < 8; ++j) {
                vertex_descriptors_[i][j] = data[i * 8 + j];
            }
        }
        
        has_descriptors_ = true;
        std::cout << "Loaded " << n_vertices << " vertex descriptors" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to load vertex descriptors: " << e.what() << std::endl;
        std::cerr << "Descriptors will not be available." << std::endl;
        has_descriptors_ = false;
    }
}

void TerrainDynamics::compute_face_data() {
    for (const auto& face : faces_) {
        Vector3d v0 = vertices_[face[0]];
        Vector3d v1 = vertices_[face[1]];
        Vector3d v2 = vertices_[face[2]];
        
        Vector3d edge1 = v1 - v0;
        Vector3d edge2 = v2 - v0;
        Vector3d normal = edge1.cross(edge2);
        normal.normalize();
        
        face_normals_.push_back(normal);
        tri_centroids_.push_back((v0 + v1 + v2) / 3.0);
    }
}

void TerrainDynamics::build_kdtree() {
    // KD-tree for triangle centroids (used by nearest_on_surface)
    cloud_.pts.resize(tri_centroids_.size());
    for (size_t i = 0; i < tri_centroids_.size(); ++i) {
        cloud_.pts[i][0] = tri_centroids_[i].x();
        cloud_.pts[i][1] = tri_centroids_[i].y();
        cloud_.pts[i][2] = tri_centroids_[i].z();
    }
    
    kdtree_ = std::make_unique<KDTree>(3, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree_->buildIndex();
    
    // KD-tree for vertices (used for fast correction and descriptor lookup)
    vertex_cloud_.pts.resize(vertices_.size());
    for (size_t i = 0; i < vertices_.size(); ++i) {
        vertex_cloud_.pts[i][0] = vertices_[i].x();
        vertex_cloud_.pts[i][1] = vertices_[i].y();
        vertex_cloud_.pts[i][2] = vertices_[i].z();
    }
    
    vertex_kdtree_ = std::make_unique<KDTree>(3, vertex_cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    vertex_kdtree_->buildIndex();
}

std::array<double, 8> TerrainDynamics::getDescriptorAtPosition(double x, double y, double z) {
    if (!has_descriptors_) {
        // Return default descriptor (flat terrain)
        return {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    
    // Fast O(log n) lookup using vertex KD-tree
    double query_pt[3] = {x, y, z};
    size_t nearest_idx;
    double dist_sq;
    vertex_kdtree_->knnSearch(query_pt, 1, &nearest_idx, &dist_sq);
    
    if (nearest_idx >= vertex_descriptors_.size()) {
        // Shouldn't happen, but return default if it does
        return {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    }
    
    return vertex_descriptors_[nearest_idx];
}

std::tuple<Vector3d, double, int> TerrainDynamics::nearest_on_surface(const Vector3d& point) {
    const size_t k = 20;
    std::vector<size_t> indices(k);
    std::vector<double> dists_sq(k);
    
    double query_pt[3] = {point.x(), point.y(), point.z()};
    kdtree_->knnSearch(query_pt, k, indices.data(), dists_sq.data());
    
    double min_dist = std::numeric_limits<double>::max();
    Vector3d closest_point;
    int closest_face = -1;
    
    for (size_t idx = 0; idx < k; ++idx) {
        if (dists_sq[idx] >= std::numeric_limits<double>::infinity()) break;
        
        size_t i = indices[idx];
        if (i >= faces_.size()) continue;
        
        const auto& face = faces_[i];
        const Vector3d& v0 = vertices_[face[0]];
        const Vector3d& v1 = vertices_[face[1]];
        const Vector3d& v2 = vertices_[face[2]];
        
        Vector3d edge0 = v1 - v0;
        Vector3d edge1 = v2 - v0;
        Vector3d edge2 = point - v0;
        
        double a = edge0.dot(edge0);
        double b = edge0.dot(edge1);
        double c = edge1.dot(edge1);
        double d = edge0.dot(edge2);
        double e = edge1.dot(edge2);
        
        double det = a * c - b * b;
        double s = b * e - c * d;
        double t = b * d - a * e;
        
        Vector3d closest;
        
        if (s + t <= det) {
            if (s < 0) {
                if (t < 0) {
                    closest = v0;
                } else {
                    double u = std::clamp(e / c, 0.0, 1.0);
                    closest = v0 + u * edge1;
                }
            } else if (t < 0) {
                double u = std::clamp(d / a, 0.0, 1.0);
                closest = v0 + u * edge0;
            } else {
                double invDet = 1.0 / det;
                s *= invDet;
                t *= invDet;
                closest = v0 + s * edge0 + t * edge1;
            }
        } else {
            if (s < 0) {
                double u = std::clamp((b + e) / (a - 2*b + c), 0.0, 1.0);
                closest = v1 + u * (v2 - v1);
            } else if (t < 0) {
                double u = std::clamp((c + e) / (a - 2*b + c), 0.0, 1.0);
                closest = v0 + u * (v1 - v0);
            } else {
                double u = std::clamp((b + d) / (a - 2*b + c), 0.0, 1.0);
                closest = v1 + u * (v2 - v1);
            }
        }
        
        double dist = (point - closest).norm();
        if (dist < min_dist) {
            min_dist = dist;
            closest_point = closest;
            closest_face = i;
        }
    }
    
    if (closest_face == -1) {
        throw std::runtime_error("Failed to find nearest surface point");
    }
    
    return {closest_point, min_dist, closest_face};
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
    
    Vector3d bary;
    if (std::abs(denom) < 1e-10) {
        bary << 1.0/3.0, 1.0/3.0, 1.0/3.0;
    } else {
        bary[1] = (d11 * d20 - d01 * d21) / denom;
        bary[2] = (d00 * d21 - d01 * d20) / denom;
        bary[0] = 1.0 - bary[1] - bary[2];
    }
    
    bary = bary.cwiseMax(0.0).cwiseMin(1.0);
    double sum = bary.sum();
    if (sum > 0) bary /= sum;
    
    return bary;
}

double TerrainDynamics::h(double x, double y) {
    auto [cp, dist, tri_id] = nearest_on_surface(Vector3d(x, y, 0));
    return cp.z();
}

std::array<double, 6> TerrainDynamics::interpolate_correction(const Vector3d& position) {
    auto [cp, dist, tri_id] = nearest_on_surface(position);
    
    if (tri_id < 0 || tri_id >= static_cast<int>(faces_.size())) {
        return {0, 0, 0, 0, 0, 0};
    }
    
    const auto& face = faces_[tri_id];
    const Vector3d& v0 = vertices_[face[0]];
    const Vector3d& v1 = vertices_[face[1]];
    const Vector3d& v2 = vertices_[face[2]];
    
    Vector3d bary = compute_barycentric(cp, v0, v1, v2);
    
    std::array<double, 6> correction = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 6; ++i) {
        correction[i] = bary[0] * vertex_corrections_[face[0]][i] +
                       bary[1] * vertex_corrections_[face[1]][i] +
                       bary[2] * vertex_corrections_[face[2]][i];
    }
    
    return correction;
}

std::array<double,6> TerrainDynamics::nominal_dynamics(
    const std::array<double,6>& state,
    const std::array<double,2>& control) {
    
    double x = state[0], y = state[1];
    double theta = state[4], psi = state[5];
    double u0 = control[0], u1 = control[1];
    
    double x1 = x + u0 * std::cos(theta) * std::cos(psi) * dt_;
    double y1 = y + u0 * std::cos(theta) * std::sin(psi) * dt_;
    double z1 = h(x1, y1);
    
    double h_x_plus = h(x1 + epsilon_, y1);
    double h_x_minus = h(x1 - epsilon_, y1);
    double h_y_plus = h(x1, y1 + epsilon_);
    double h_y_minus = h(x1, y1 - epsilon_);
    
    double delta_x = (h_x_plus - h_x_minus) / (2.0 * epsilon_);
    double delta_y = (h_y_plus - h_y_minus) / (2.0 * epsilon_);
    
    double phi1 = std::atan2(-delta_y, std::sqrt(1.0 + delta_x * delta_x));
    double theta1 = std::atan2(delta_x, std::sqrt(1.0 + delta_y * delta_y));
    double psi1 = psi + u1 * dt_;
    
    return {x1, y1, z1, phi1, theta1, psi1};
}

std::array<double,6> TerrainDynamics::dynamics(
    const std::array<double,6>& state,
    const std::array<double,2>& control) {
    
    // Nominal prediction
    auto nominal = nominal_dynamics(state, control);
    
    // Fast O(log n) lookup: find nearest vertex and use its precomputed correction
    double query_pt[3] = {state[0], state[1], state[2]};
    size_t nearest_idx;
    double dist_sq;
    vertex_kdtree_->knnSearch(query_pt, 1, &nearest_idx, &dist_sq);
    
    const auto& correction = vertex_corrections_[nearest_idx];
    
    // Apply correction
    std::array<double,6> next_state;
    for (int i = 0; i < 6; ++i) {
        next_state[i] = nominal[i] + correction[i];
    }
    
    return next_state;
}

} // namespace epsilon_mesh_planner