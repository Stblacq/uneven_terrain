#include "gakd_mesh_planner/terrain_dynamics.hpp"
#include <cnpy.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>

namespace gakd_mesh_planner {

TerrainDynamics::TerrainDynamics(const std::string& mesh_file,
                                 const std::string& curvature_file,
                                 const std::string& model_dir,
                                 double dt,
                                 double epsilon,
                                 double neighborhood_radius)
    : dt_(dt), epsilon_(epsilon), neighborhood_radius_(neighborhood_radius),
      graph_(nullptr), session_(nullptr), status_(nullptr),
      input_tensor_(nullptr), input_buffer_(nullptr), tensors_initialized_(false) {
    
    load_obj(mesh_file);
    compute_face_data();
    compute_vertex_normals();
    load_curvatures(curvature_file);
    build_kdtree();
    load_tensorflow_model(model_dir);
    initialize_tensors();  // OPTIMIZATION: Pre-allocate tensors
}

TerrainDynamics::~TerrainDynamics() {
    cleanup_tensors();  // OPTIMIZATION: Clean up persistent tensors
    if (session_ != nullptr) {
        TF_DeleteSession(session_, status_);
    }
    if (graph_ != nullptr) {
        TF_DeleteGraph(graph_);
    }
    if (status_ != nullptr) {
        TF_DeleteStatus(status_);
    }
}

void TerrainDynamics::load_tensorflow_model(const std::string& model_dir) {
    status_ = TF_NewStatus();
    graph_ = TF_NewGraph();
    
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    
    // OPTIMIZATION: Configure session for single-threaded performance
    // For small models with many sequential calls, single-threaded is faster
    uint8_t config[] = {
        0x10, 0x01,  // intra_op_parallelism_threads = 1
        0x28, 0x01,  // inter_op_parallelism_threads = 1  
        0x32, 0x02, 0x20, 0x01  // allow_growth = true
    };
    TF_SetConfig(sess_opts, (void*)config, sizeof(config), status_);
    if (TF_GetCode(status_) != TF_OK) {
        std::cerr << "Warning: Failed to set TF config: " << TF_Message(status_) << std::endl;
        // Continue anyway with default config
        TF_SetStatus(status_, TF_OK, "");
    }
    
    TF_Buffer* run_options = TF_NewBufferFromString("", 0);
    
    const char* tags[] = {"serve"};
    int ntags = 1;
    
    session_ = TF_LoadSessionFromSavedModel(
        sess_opts,
        run_options,
        model_dir.c_str(),
        tags,
        ntags,
        graph_,
        nullptr,
        status_
    );
    
    TF_DeleteBuffer(run_options);
    TF_DeleteSessionOptions(sess_opts);
    
    if (TF_GetCode(status_) != TF_OK) {
        std::string error_msg = "Failed to load TensorFlow model: ";
        error_msg += TF_Message(status_);
        throw std::runtime_error(error_msg);
    }
    
    input_op_.oper = TF_GraphOperationByName(graph_, "serving_default_z");
    input_op_.index = 0;
    
    if (input_op_.oper == nullptr) {
        throw std::runtime_error("Failed to find input operation 'serving_default_z'");
    }
    
    output_op_.oper = TF_GraphOperationByName(graph_, "StatefulPartitionedCall");
    output_op_.index = 0;
    
    if (output_op_.oper == nullptr) {
        throw std::runtime_error("Failed to find output operation 'StatefulPartitionedCall'");
    }
}

// OPTIMIZATION: Initialize persistent tensors once
void TerrainDynamics::initialize_tensors() {
    if (tensors_initialized_) return;
    
    const int64_t dims[] = {1, 16};  // Changed from 15 to 16 (8-feature descriptor)
    const size_t data_size = 16 * sizeof(double);  // Changed from 15 to 16
    
    // Allocate persistent buffer
    input_buffer_ = new double[16];  // Changed from 15 to 16
    
    // Create persistent tensor
    input_tensor_ = TF_AllocateTensor(TF_DOUBLE, dims, 2, data_size);
    
    if (input_tensor_ == nullptr) {
        delete[] input_buffer_;
        throw std::runtime_error("Failed to allocate persistent input tensor");
    }
    
    tensors_initialized_ = true;
}

// OPTIMIZATION: Clean up persistent tensors
void TerrainDynamics::cleanup_tensors() {
    if (input_tensor_ != nullptr) {
        TF_DeleteTensor(input_tensor_);
        input_tensor_ = nullptr;
    }
    if (input_buffer_ != nullptr) {
        delete[] input_buffer_;
        input_buffer_ = nullptr;
    }
    tensors_initialized_ = false;
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
        Vector3d v0 = vertices_[face[0]];
        Vector3d v1 = vertices_[face[1]];
        Vector3d v2 = vertices_[face[2]];
        
        Vector3d edge1 = v1 - v0;
        Vector3d edge2 = v2 - v0;
        Vector3d normal = edge1.cross(edge2);
        double area = 0.5 * normal.norm();
        normal.normalize();
        
        face_normals_.push_back(normal);
        tri_centroids_.push_back((v0 + v1 + v2) / 3.0);
        face_areas_.push_back(area);
    }
}

void TerrainDynamics::compute_vertex_normals() {
    vertex_normals_.resize(vertices_.size(), Vector3d::Zero());
    
    for (size_t i = 0; i < faces_.size(); ++i) {
        const auto& face = faces_[i];
        const Vector3d& normal = face_normals_[i];
        double area = face_areas_[i];
        
        vertex_normals_[face[0]] += normal * area;
        vertex_normals_[face[1]] += normal * area;
        vertex_normals_[face[2]] += normal * area;
    }
    
    for (auto& vn : vertex_normals_) {
        vn.normalize();
    }
}

void TerrainDynamics::build_kdtree() {
    if (tri_centroids_.empty()) {
        throw std::runtime_error("Cannot build KD-tree: no triangle centroids");
    }
    
    cloud_.pts.resize(tri_centroids_.size());
    for (size_t i = 0; i < tri_centroids_.size(); ++i) {
        cloud_.pts[i][0] = tri_centroids_[i].x();
        cloud_.pts[i][1] = tri_centroids_[i].y();
        cloud_.pts[i][2] = tri_centroids_[i].z();
    }
    
    kdtree_ = std::make_unique<KDTree>(3, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    kdtree_->buildIndex();
}


std::tuple<Vector3d, double, int> TerrainDynamics::nearest_on_surface(const Vector3d& point) {
    // Use KD-tree to find nearby candidate triangles
    const size_t k = 20;  // Check 20 nearest triangle centroids for robustness
    std::vector<size_t> indices(k);
    std::vector<double> dists_sq(k);
    
    double query_pt[3] = {point.x(), point.y(), point.z()};
    kdtree_->knnSearch(query_pt, k, indices.data(), dists_sq.data());
    
    // Search through the k nearest triangles
    double min_dist = std::numeric_limits<double>::max();
    Vector3d closest_point;
    int closest_face = -1;
    
    for (size_t idx = 0; idx < k; ++idx) {
        if (dists_sq[idx] >= std::numeric_limits<double>::infinity()) {
            break;  // No more valid results
        }
        
        size_t i = indices[idx];
        if (i >= faces_.size()) continue;
        
        const auto& face = faces_[i];
        const Vector3d& v0 = vertices_[face[0]];
        const Vector3d& v1 = vertices_[face[1]];
        const Vector3d& v2 = vertices_[face[2]];
        
        // Project point onto triangle plane
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

std::pair<Vector3d, Vector3d> TerrainDynamics::project_to_mesh(const Vector3d& query) {
    auto [cp, dist, tri_id] = nearest_on_surface(query);
    
    const auto& face = faces_[tri_id];
    const Vector3d& v0 = vertices_[face[0]];
    const Vector3d& v1 = vertices_[face[1]];
    const Vector3d& v2 = vertices_[face[2]];
    
    Vector3d bary = compute_barycentric(cp, v0, v1, v2);
    
    Vector3d normal = bary[0] * vertex_normals_[face[0]] +
                     bary[1] * vertex_normals_[face[1]] +
                     bary[2] * vertex_normals_[face[2]];
    normal.normalize();
    
    return {cp, normal};
}

double TerrainDynamics::h(double x, double y) {
    auto [cp, normal] = project_to_mesh(Vector3d(x, y, 0));
    return cp.z();
}

std::vector<double> TerrainDynamics::query_descriptor(const Vector3d& point) {
    if (faces_.empty()) {
        std::cerr << "ERROR: No faces loaded!" << std::endl;
        return {0, 0, 1, 0, 0, 0, 0, 0};  // 8 features
    }
    
    auto [cp, dist, tri_id] = nearest_on_surface(point);
    
    if (tri_id < 0 || tri_id >= static_cast<int>(faces_.size())) {
        std::cerr << "ERROR: Invalid triangle ID: " << tri_id << std::endl;
        return {0, 0, 1, 0, 0, 0, 0, 0};  // 8 features
    }
    
    const auto& face = faces_[tri_id];
    
    if (face[0] >= vertices_.size() || face[1] >= vertices_.size() || face[2] >= vertices_.size()) {
        std::cerr << "ERROR: Invalid vertex indices" << std::endl;
        return {0, 0, 1, 0, 0, 0, 0, 0};  // 8 features
    }
    
    const Vector3d& v0 = vertices_[face[0]];
    const Vector3d& v1 = vertices_[face[1]];
    const Vector3d& v2 = vertices_[face[2]];
    
    Vector3d bary = compute_barycentric(cp, v0, v1, v2);
    
    Vector3d normal = bary[0] * vertex_normals_[face[0]] +
                     bary[1] * vertex_normals_[face[1]] +
                     bary[2] * vertex_normals_[face[2]];
    normal.normalize();
    
    if (!kdtree_) {
        std::cerr << "ERROR: KD-tree not initialized!" << std::endl;
        return {normal.x(), normal.y(), normal.z(), 0, 0, 0, 0, 0};  // 8 features
    }
    
    // KD-tree search
    const size_t num_results = 50;
    std::vector<size_t> ret_indices(num_results);
    std::vector<double> out_dists_sqr(num_results);
    
    double query_pt[3] = {cp.x(), cp.y(), cp.z()};
    
    kdtree_->knnSearch(query_pt, num_results, ret_indices.data(), out_dists_sqr.data());
    
    // Filter by radius
    std::vector<size_t> nearby_indices;
    double radius_sq = neighborhood_radius_ * neighborhood_radius_;
    
    for (size_t i = 0; i < num_results; ++i) {
        if (out_dists_sqr[i] < std::numeric_limits<double>::infinity() && 
            out_dists_sqr[i] <= radius_sq) {
            nearby_indices.push_back(ret_indices[i]);
        }
    }
    
    if (nearby_indices.empty()) {
        return {normal.x(), normal.y(), normal.z(), 0.0, 0.0, 0.0, 0.0, dist};  // 8 features, include distance
    }
    
    // Compute weighted statistics
    std::vector<Vector3d> nearby_tris;
    std::vector<Vector3d> nearby_normals;
    std::vector<double> nearby_areas;
    
    for (size_t idx : nearby_indices) {
        nearby_tris.push_back(tri_centroids_[idx]);
        nearby_normals.push_back(face_normals_[idx]);
        nearby_areas.push_back(face_areas_[idx]);
    }
    
    // Compute weights
    std::vector<double> weights(nearby_tris.size());
    double weight_sum = 0.0;
    for (size_t i = 0; i < nearby_tris.size(); ++i) {
        double d = (nearby_tris[i] - cp).norm();
        weights[i] = nearby_areas[i] / (d + 1e-6);
        weight_sum += weights[i];
    }
    for (auto& w : weights) w /= weight_sum;
    
    // Weighted normal
    Vector3d weighted_normal = Vector3d::Zero();
    for (size_t i = 0; i < nearby_normals.size(); ++i) {
        weighted_normal += weights[i] * nearby_normals[i];
    }
    weighted_normal.normalize();
    
    // Height variation
    double mean_height = 0.0;
    for (size_t i = 0; i < nearby_tris.size(); ++i) {
        mean_height += (nearby_tris[i] - cp).dot(weighted_normal);
    }
    mean_height /= nearby_tris.size();
    
    double sigma_z = 0.0;
    for (size_t i = 0; i < nearby_tris.size(); ++i) {
        double h = (nearby_tris[i] - cp).dot(weighted_normal);
        sigma_z += (h - mean_height) * (h - mean_height);
    }
    sigma_z = std::sqrt(sigma_z / nearby_tris.size());
    
    // Interpolate curvatures
    double K = 0.0, H = 0.0;
    if (!gauss_curv_.empty() && !mean_curv_.empty()) {
        if (face[0] < gauss_curv_.size() && face[1] < gauss_curv_.size() && face[2] < gauss_curv_.size()) {
            K = bary[0] * gauss_curv_[face[0]] +
                bary[1] * gauss_curv_[face[1]] +
                bary[2] * gauss_curv_[face[2]];
            
            H = bary[0] * mean_curv_[face[0]] +
                bary[1] * mean_curv_[face[1]] +
                bary[2] * mean_curv_[face[2]];
        }
    }
    
    // Total area
    double A_total = 0.0;
    for (double a : nearby_areas) A_total += a;
    
    // 8th feature: Distance from query point to mesh surface
    double distance_from_mesh = dist;
    
    std::vector<double> descriptor = {weighted_normal.x(), weighted_normal.y(), weighted_normal.z(),
            sigma_z, K, H, A_total, distance_from_mesh};  // Now 8 features
    
    return descriptor;
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
    
    // Verify TensorFlow is initialized
    if (session_ == nullptr || graph_ == nullptr || status_ == nullptr || !tensors_initialized_) {
        std::cerr << "TensorFlow not initialized properly" << std::endl;
        return nominal;
    }
    
    // Get descriptor - NOW RETURNS 8 FEATURES
    Vector3d position(state[0], state[1], state[2]);
    auto descriptor = query_descriptor(position);
    
    // OPTIMIZATION: Reuse persistent buffer instead of allocating new one
    // Input: [state(6), control(2), descriptor(8)] = 16 features
    for (int i = 0; i < 6; ++i) input_buffer_[i] = state[i];
    for (int i = 0; i < 2; ++i) input_buffer_[6 + i] = control[i];
    for (int i = 0; i < 8; ++i) input_buffer_[8 + i] = descriptor[i];  
    
    // OPTIMIZATION: Copy data into pre-allocated tensor instead of creating new one
    memcpy(TF_TensorData(input_tensor_), input_buffer_, 16 * sizeof(double));
    
    // Verify operations are valid
    if (input_op_.oper == nullptr || output_op_.oper == nullptr) {
        std::cerr << "Input or output operation is null" << std::endl;
        return nominal;
    }
    
    // Run inference with persistent tensor
    TF_Tensor* output_tensor = nullptr;
    TF_SessionRun(
        session_,
        nullptr,
        &input_op_, &input_tensor_, 1,  // Use persistent tensor
        &output_op_, &output_tensor, 1,
        nullptr, 0,
        nullptr,
        status_
    );
    
    if (TF_GetCode(status_) != TF_OK) {
        std::cerr << "TensorFlow inference failed: " << TF_Message(status_) << std::endl;
        if (output_tensor != nullptr) {
            TF_DeleteTensor(output_tensor);
        }
        return nominal;
    }
    
    if (output_tensor == nullptr) {
        std::cerr << "Output tensor is null" << std::endl;
        return nominal;
    }
    
    // Extract residual
    double* residual_data = static_cast<double*>(TF_TensorData(output_tensor));
    if (residual_data == nullptr) {
        std::cerr << "Failed to get tensor data" << std::endl;
        TF_DeleteTensor(output_tensor);
        return nominal;
    }
    
    std::array<double,6> next_state;
    for (int i = 0; i < 6; ++i) {
        next_state[i] = nominal[i] + residual_data[i];
    }
    
    TF_DeleteTensor(output_tensor);
    
    return next_state;
}

} // namespace gakd_mesh_planner