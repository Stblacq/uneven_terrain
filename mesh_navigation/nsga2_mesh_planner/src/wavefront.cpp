#include "nsga2_mesh_planner/wavefront.h"
#include <chrono>

namespace nsga2_mesh_planner
{

WaveFront::WaveFront(const std::shared_ptr<mesh_map::MeshMap>& mesh_map,
                     const WaveFrontConfig& config,
                     rclcpp::Logger logger)
    : mesh_map_(mesh_map)
    , config_(config)
    , logger_(logger)
    , cancel_computation_(false)
{
    const auto mesh = mesh_map_->mesh();
    direction_ = lvr2::DenseVertexMap<float>(mesh->nextVertexIndex(), 0);
}

bool WaveFront::compute(const mesh_map::Vector& start,
                        const mesh_map::Vector& goal,
                        std::string& message)
{
    cancel_computation_ = false;
    
    if (!propagate(start, goal, message))
    {
        return false;
    }
    
    computeVectorMap();
    return true;
}

bool WaveFront::propagate(const mesh_map::Vector& original_start,
                         const mesh_map::Vector& original_goal,
                         std::string& message)
{
    const auto mesh = mesh_map_->mesh();
    const auto& vertex_costs = mesh_map_->vertexCosts();
    const auto& edge_weights = mesh_map_->edgeWeights();
    auto& invalid = mesh_map_->invalid;
    
    // IMPORTANT: Wavefront propagates FROM GOAL TO START
    // So we swap start and goal here
    mesh_map::Vector start = original_goal;  // Wavefront starts at goal
    mesh_map::Vector goal = original_start;   // And propagates toward start
    
    // Make mutable copies for getContainingFace
    mesh_map::Vector start_copy = start;
    mesh_map::Vector goal_copy = goal;
    
    // Find containing faces
    const lvr2::OptionalFaceHandle start_opt = mesh_map_->getContainingFace(start_copy, 0.4);
    const lvr2::OptionalFaceHandle goal_opt = mesh_map_->getContainingFace(goal_copy, 0.4);
    
    if (!start_opt)
    {
        message = "Could not find a face close enough to the given start pose";
        RCLCPP_WARN_STREAM(logger_, message);
        return false;
    }
    if (!goal_opt)
    {
        message = "Could not find a face close enough to the given goal pose";
        RCLCPP_WARN_STREAM(logger_, message);
        return false;
    }
    
    const auto& start_face = start_opt.unwrap();
    const auto& goal_face = goal_opt.unwrap();
    
    // Clear and initialize
    potential_.clear();
    predecessors_.clear();
    vector_map_.clear();
    cutting_faces_.clear();
    
    lvr2::DenseVertexMap<bool> fixed(mesh->nextVertexIndex(), false);
    
    // Initialize all vertices
    for (auto const& vH : mesh->vertices())
    {
        potential_.insert(vH, std::numeric_limits<float>::infinity());
        predecessors_.insert(vH, vH);
    }
    
    // Initialize priority queue with start face vertices (at GOAL position)
    lvr2::Meap<lvr2::VertexHandle, float> pq;
    for (auto vH : mesh->getVerticesOfFace(start_face))
    {
        const mesh_map::Vector diff = start - mesh->getVertexPosition(vH);
        const float dist = diff.length();
        potential_[vH] = dist;
        vector_map_.insert(vH, diff);
        cutting_faces_.insert(vH, start_face);
        fixed[vH] = true;
        pq.insert(vH, dist);
    }
    
    std::array<lvr2::VertexHandle, 3> goal_vertices = mesh->getVerticesOfFace(goal_face);
    float goal_dist = std::numeric_limits<float>::infinity();
    
    const auto t_start = std::chrono::steady_clock::now();
    size_t vertices_processed = 0;
    
    // Main wavefront propagation loop
    while (!pq.isEmpty() && !cancel_computation_)
    {
        lvr2::VertexHandle current_vh = pq.popMin().key();
        fixed[current_vh] = true;
        vertices_processed++;
        
        if (potential_[current_vh] > goal_dist)
            continue;
        
        if (vertex_costs[current_vh] >= config_.cost_limit)
            continue;
        
        if (invalid[current_vh])
            continue;
        
        // Check if we reached goal (original start)
        if (current_vh == goal_vertices[0] || current_vh == goal_vertices[1] || current_vh == goal_vertices[2])
        {
            if (goal_dist == std::numeric_limits<float>::infinity() &&
                fixed[goal_vertices[0]] && fixed[goal_vertices[1]] && fixed[goal_vertices[2]])
            {
                RCLCPP_DEBUG_STREAM(logger_, "Wave front reached the goal!");
                goal_dist = potential_[current_vh] + config_.goal_dist_offset;
            }
        }
        
        try
        {
            std::vector<lvr2::FaceHandle> faces;
            mesh->getFacesOfVertex(current_vh, faces);
            
            for (auto fh : faces)
            {
                const auto vertices = mesh->getVerticesOfFace(fh);
                const lvr2::VertexHandle& a = vertices[0];
                const lvr2::VertexHandle& b = vertices[1];
                const lvr2::VertexHandle& c = vertices[2];
                
                if (invalid[a] || invalid[b] || invalid[c])
                    continue;
                
                // Process faces with exactly one free vertex
                if (fixed[a] && fixed[b] && fixed[c])
                {
                    continue; // All fixed
                }
                else if (fixed[a] && fixed[b] && !fixed[c])
                {
                    if (vertex_costs[c] < config_.cost_limit)
                    {
                        bool updated = false;
                        switch (config_.update_method)
                        {
                            case WaveFrontConfig::UpdateMethod::WITH_S:
                                updated = waveFrontUpdateWithS(potential_, edge_weights, a, b, c);
                                break;
                            case WaveFrontConfig::UpdateMethod::FMM:
                                updated = waveFrontUpdateFMM(potential_, edge_weights, a, b, c);
                                break;
                            case WaveFrontConfig::UpdateMethod::STANDARD:
                            default:
                                updated = waveFrontUpdate(potential_, edge_weights, a, b, c);
                                break;
                        }
                        if (updated)
                        {
                            pq.insert(c, potential_[c]);
                        }
                    }
                }
                else if (fixed[a] && !fixed[b] && fixed[c])
                {
                    if (vertex_costs[b] < config_.cost_limit)
                    {
                        bool updated = false;
                        switch (config_.update_method)
                        {
                            case WaveFrontConfig::UpdateMethod::WITH_S:
                                updated = waveFrontUpdateWithS(potential_, edge_weights, c, a, b);
                                break;
                            case WaveFrontConfig::UpdateMethod::FMM:
                                updated = waveFrontUpdateFMM(potential_, edge_weights, c, a, b);
                                break;
                            case WaveFrontConfig::UpdateMethod::STANDARD:
                            default:
                                updated = waveFrontUpdate(potential_, edge_weights, c, a, b);
                                break;
                        }
                        if (updated)
                        {
                            pq.insert(b, potential_[b]);
                        }
                    }
                }
                else if (!fixed[a] && fixed[b] && fixed[c])
                {
                    if (vertex_costs[a] < config_.cost_limit)
                    {
                        bool updated = false;
                        switch (config_.update_method)
                        {
                            case WaveFrontConfig::UpdateMethod::WITH_S:
                                updated = waveFrontUpdateWithS(potential_, edge_weights, b, c, a);
                                break;
                            case WaveFrontConfig::UpdateMethod::FMM:
                                updated = waveFrontUpdateFMM(potential_, edge_weights, b, c, a);
                                break;
                            case WaveFrontConfig::UpdateMethod::STANDARD:
                            default:
                                updated = waveFrontUpdate(potential_, edge_weights, b, c, a);
                                break;
                        }
                        if (updated)
                        {
                            pq.insert(a, potential_[a]);
                        }
                    }
                }
            }
        }
        catch (lvr2::PanicException& exception)
        {
            invalid.insert(current_vh, true);
            RCLCPP_ERROR_STREAM(logger_, "Found invalid vertex!");
            continue;
        }
        catch (lvr2::VertexLoopException& exception)
        {
            invalid.insert(current_vh, true);
            RCLCPP_ERROR_STREAM(logger_, "Found invalid vertex!");
            continue;
        }
    }
    
    const auto t_end = std::chrono::steady_clock::now();
    const auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    
    if (cancel_computation_)
    {
        message = "Computation was canceled";
        RCLCPP_WARN_STREAM(logger_, message);
        return false;
    }
    
    // Check if goal is reachable
    bool goal_reached = false;
    for (auto goal_vertex : goal_vertices)
    {
        if (goal_vertex != predecessors_[goal_vertex])
        {
            goal_reached = true;
            break;
        }
    }
    
    if (!goal_reached && goal_face != start_face)
    {
        message = "Goal is not reachable from start";
        RCLCPP_WARN_STREAM(logger_, message);
        return false;
    }
    
    RCLCPP_INFO_STREAM(logger_, "Wavefront propagation completed. Processed " << vertices_processed 
                       << " vertices in " << duration_ms.count() << " ms");
    
    return true;
}

bool WaveFront::waveFrontUpdate(lvr2::DenseVertexMap<float>& distances,
                                const lvr2::DenseEdgeMap<float>& edge_weights,
                                const lvr2::VertexHandle& v1,
                                const lvr2::VertexHandle& v2,
                                const lvr2::VertexHandle& v3)
{
    const auto mesh = mesh_map_->mesh();
    
    const double u1 = distances[v1];
    const double u2 = distances[v2];
    const double u3 = distances[v3];
    
    const lvr2::OptionalEdgeHandle e12h = mesh->getEdgeBetween(v1, v2);
    const double c = edge_weights[e12h.unwrap()];
    const double c_sq = c * c;
    
    const lvr2::OptionalEdgeHandle e13h = mesh->getEdgeBetween(v1, v3);
    const double b = edge_weights[e13h.unwrap()];
    const double b_sq = b * b;
    
    const lvr2::OptionalEdgeHandle e23h = mesh->getEdgeBetween(v2, v3);
    const double a = edge_weights[e23h.unwrap()];
    const double a_sq = a * a;
    
    const double u1_sq = u1 * u1;
    const double u2_sq = u2 * u2;
    
    // Triangle unfolding computation
    const double sx = (c_sq + u1_sq - u2_sq) / (2 * c);
    const double sy = -sqrt(std::max(u1_sq - sx * sx, 0.0));
    
    const double p = (b_sq + c_sq - a_sq) / (2 * c);
    const double hc = sqrt(std::max(b_sq - p * p, 0.0));
    
    const double dy = hc - sy;
    const double dx = p - sx;
    
    const double u3tmp_sq = dx * dx + dy * dy;
    double u3tmp = sqrt(u3tmp_sq);
    
    if (!std::isfinite(u3tmp))
    {
        return false;
    }
    
    if (u3tmp < u3)
    {
        const double t0a = (a_sq + b_sq - c_sq) / (2 * a * b);
        const double t1a = (u3tmp_sq + b_sq - u1_sq) / (2 * u3tmp * b);
        const double t2a = (a_sq + u3tmp_sq - u2_sq) / (2 * a * u3tmp);
        
        // Corner case: side b + u1 ~= u3
        if (std::fabs(t1a) > 1)
        {
            u3tmp = u1 + b;
            if (u3tmp < u3)
            {
                auto fH = mesh->getFaceBetween(v1, v2, v3).unwrap();
                cutting_faces_.insert(v3, fH);
                predecessors_[v3] = v1;
                distances[v3] = static_cast<float>(u3tmp);
                direction_[v3] = 0;
                return true;
            }
            return false;
        }
        // Corner case: side a + u2 ~= u3
        else if (std::fabs(t2a) > 1)
        {
            u3tmp = u2 + a;
            if (u3tmp < u3)
            {
                auto fH = mesh->getFaceBetween(v1, v2, v3).unwrap();
                cutting_faces_.insert(v3, fH);
                predecessors_[v3] = v2;
                distances[v3] = static_cast<float>(u3tmp);
                direction_[v3] = 0;
                return true;
            }
            return false;
        }
        
        const double theta0 = acos(t0a);
        const double theta1 = acos(t1a);
        const double theta2 = acos(t2a);
        
        // The key logic: choose the update that gives best angle
        if (theta1 < theta0 && theta2 < theta0)
        {
            auto fH = mesh->getFaceBetween(v1, v2, v3).unwrap();
            cutting_faces_.insert(v3, fH);
            distances[v3] = static_cast<float>(u3tmp);
            
            if (theta1 < theta2)
            {
                predecessors_[v3] = v1;
                direction_[v3] = theta1;
            }
            else
            {
                predecessors_[v3] = v2;
                direction_[v3] = -theta2;
            }
            return true;
        }
        else if (theta1 < theta2)
        {
            u3tmp = u1 + b;
            if (u3tmp < u3)
            {
                auto fH = mesh->getFaceBetween(v1, v2, v3).unwrap();
                cutting_faces_.insert(v3, fH);
                predecessors_[v3] = v1;
                distances[v3] = static_cast<float>(u3tmp);
                direction_[v3] = 0;
                return true;
            }
            return false;
        }
        else
        {
            u3tmp = u2 + a;
            if (u3tmp < u3)
            {
                auto fH = mesh->getFaceBetween(v1, v2, v3).unwrap();
                cutting_faces_.insert(v3, fH);
                predecessors_[v3] = v2;
                distances[v3] = static_cast<float>(u3tmp);
                direction_[v3] = 0;
                return true;
            }
            return false;
        }
    }
    return false;
}

void WaveFront::computeVectorMap()
{
    const auto mesh = mesh_map_->mesh();
    const auto& vertex_normals = mesh_map_->vertexNormals();
    const auto& vertex_costs = mesh_map_->vertexCosts();  // Add this
    
    for (auto v3 : mesh->vertices())
    {
        // Skip vertices at or above cost limit
        if (vertex_costs[v3] >= config_.cost_limit) {
            continue;
        }
        
        const lvr2::VertexHandle& v1 = predecessors_[v3];
        
        if (v1 == v3)
            continue;
        
        const auto& optFh = cutting_faces_.get(v3);
        if (!optFh)
            continue;
        
        const auto& vec3 = mesh->getVertexPosition(v3);
        const auto& vec1 = mesh->getVertexPosition(v1);
        
        const auto dirVec = (vec1 - vec3).rotated(vertex_normals[v3], direction_[v3]);
        vector_map_.insert(v3, dirVec.normalized());
    }
    
    mesh_map_->setVectorMap(vector_map_);
}


bool WaveFront::waveFrontUpdateWithS(lvr2::DenseVertexMap<float>& distances,
                                     const lvr2::DenseEdgeMap<float>& edge_weights,
                                     const lvr2::VertexHandle& v1,
                                     const lvr2::VertexHandle& v2,
                                     const lvr2::VertexHandle& v3)
{
    const auto mesh = mesh_map_->mesh();

    const double u1 = distances[v1];
    const double u2 = distances[v2];
    const double u3 = distances[v3];

    const lvr2::OptionalEdgeHandle e12h = mesh->getEdgeBetween(v1, v2);
    const double c = edge_weights[e12h.unwrap()];
    const double c_sq = c * c;

    const lvr2::OptionalEdgeHandle e13h = mesh->getEdgeBetween(v1, v3);
    const double b = edge_weights[e13h.unwrap()];
    const double b_sq = b * b;

    const lvr2::OptionalEdgeHandle e23h = mesh->getEdgeBetween(v2, v3);
    const double a = edge_weights[e23h.unwrap()];
    const double a_sq = a * a;

    const double u1_sq = u1 * u1;
    const double u2_sq = u2 * u2;

    const double A = sqrt(std::max<double>((-u1 + u2 + c) * (u1 - u2 + c) * (u1 + u2 - c) * (u1 + u2 + c), 0));
    const double B = sqrt(std::max<double>((-a + b + c) * (a - b + c) * (a + b - c) * (a + b + c), 0));
    const double sx = (c_sq + u1_sq - u2_sq) / (2 * c);
    const double sy = -A / (2 * c);
    const double p = (-a_sq + b_sq + c_sq) / (2 * c);
    const double hc = B / (2 * c);
    const double dy = hc - sy;
    const double dx = p - sx;

    const double u3tmp_sq = dx * dx + dy * dy;
    double u3tmp = sqrt(u3tmp_sq);
    const auto& vertex_costs = mesh_map_->vertexCosts();

    if (u3tmp < u3)
    {

        float cost_penalty = 0.0f;
        if (vertex_costs[v3] > 0.5) {  // Approaching dangerous area
            cost_penalty = vertex_costs[v3] * 10.0f;  // Scale penalty
        }
        
        u3tmp += cost_penalty;  // Add penalty to distance
        
        if (u3tmp >= u3) {  // If penalty makes it worse, skip
            return false;
        }

        if (distances[v1] < distances[v2])
        {
            const double S = sy * p - sx * hc;
            const double t1cos = (u3tmp_sq + b_sq - u1_sq) / (2 * u3tmp * b);
            if (S <= 0 && std::fabs(t1cos) <= 1)
            {
                const double theta = acos(t1cos);
                predecessors_[v3] = v1;
                direction_[v3] = static_cast<float>(theta);
                distances[v3] = static_cast<float>(u3tmp);
                const lvr2::FaceHandle fh = mesh->getFaceBetween(v1, v2, v3).unwrap();
                cutting_faces_.insert(v3, fh);
                return true;
            }
            else
            {
                u3tmp = u1 + b;
                if (u3tmp < u3)
                {
                    predecessors_[v3] = v1;
                    direction_[v3] = 0;
                    distances[v3] = u3tmp;
                    const lvr2::FaceHandle fh = mesh->getFaceBetween(v1, v2, v3).unwrap();
                    cutting_faces_.insert(v3, fh);
                    return true;
                }
                return false;
            }
        }
        else
        {
            const double S = sx * hc - hc * c + sy * c - sy * p;
            const double t2cos = (a_sq + u3tmp_sq - u2_sq) / (2 * a * u3tmp);
            if (S <= 0 && std::fabs(t2cos) <= 1)
            {
                const lvr2::FaceHandle fh = mesh->getFaceBetween(v1, v2, v3).unwrap();
                const double theta = -acos(t2cos);
                direction_[v3] = static_cast<float>(theta);
                distances[v3] = static_cast<float>(u3tmp);
                predecessors_[v3] = v2;
                cutting_faces_.insert(v3, fh);
                return true;
            }
            else
            {
                u3tmp = u2 + a;
                if (u3tmp < u3)
                {
                    direction_[v3] = 0;
                    distances[v3] = u3tmp;
                    predecessors_[v3] = v2;
                    const lvr2::FaceHandle fh = mesh->getFaceBetween(v1, v2, v3).unwrap();
                    cutting_faces_.insert(v3, fh);
                    return true;
                }
                return false;
            }
        }
    }
    return false;
}

bool WaveFront::waveFrontUpdateFMM(lvr2::DenseVertexMap<float>& distances,
                                   const lvr2::DenseEdgeMap<float>& edge_weights,
                                   const lvr2::VertexHandle& v1tmp,
                                   const lvr2::VertexHandle& v2tmp,
                                   const lvr2::VertexHandle& v3)
{
    const auto mesh = mesh_map_->mesh();

    bool v1_smaller = distances[v1tmp] < distances[v2tmp];
    const lvr2::VertexHandle v1 = v1_smaller ? v1tmp : v2tmp;
    const lvr2::VertexHandle v2 = v1_smaller ? v2tmp : v1tmp;

    const double u1 = distances[v1];
    const double u2 = distances[v2];
    const double u3 = distances[v3];

    const lvr2::OptionalEdgeHandle e12h = mesh->getEdgeBetween(v1, v2);
    const double c = edge_weights[e12h.unwrap()];
    const double c_sq = c * c;

    const lvr2::OptionalEdgeHandle e13h = mesh->getEdgeBetween(v1, v3);
    const double b = edge_weights[e13h.unwrap()];
    const double b_sq = b * b;

    const lvr2::OptionalEdgeHandle e23h = mesh->getEdgeBetween(v2, v3);
    const double a = edge_weights[e23h.unwrap()];
    const double a_sq = a * a;

    const double delta_u = u2 - u1;
    const double cos_theta = (a_sq + b_sq - c_sq) / (2 * a * b);

    const double k0 = a_sq + b_sq - 2 * a * b * cos_theta;
    const double k1 = 2 * b * delta_u * (a * cos_theta - b);
    const double k2 = b_sq * (delta_u * delta_u - a_sq * (1 - cos_theta * cos_theta));

    double t;
    const double r = k1 * k1 - 4 * k0 * k2;
    if (r < 0)
    {
        t = -k1 / (2 * k0);
    }
    else
    {
        t = (-k1 + sqrt(r)) / (2 * k0);
    }

    const double e = b * (t - delta_u) / t;

    if (delta_u < t && (e < a / cos_theta) && (e > a * cos_theta))
    {
        const double u3_tmp = u1 + t;
        if (u3_tmp < u3)
        {
            auto fH = mesh->getFaceBetween(v1, v2, v3).unwrap();
            cutting_faces_.insert(v3, fH);
            predecessors_[v3] = v1;
            distances[v3] = static_cast<float>(u3_tmp);
            const double theta = acos(cos_theta);
            const double phi = asin((e * sin(theta)) / sqrt(a_sq + e * e - 2 * a * e * cos_theta));
            direction_[v3] = theta + phi - M_PI_2;
            return true;
        }
    }
    else
    {
        const double u1t = u1 + b;
        const double u2t = u2 + a;

        if (u1t < u2t)
        {
            if (u1t < u3)
            {
                auto fH = mesh->getFaceBetween(v1, v2, v3).unwrap();
                cutting_faces_.insert(v3, fH);
                predecessors_[v3] = v1;
                distances[v3] = static_cast<float>(u1t);
                direction_[v3] = 0;
                return true;
            }
        }
        else
        {
            if (u2t < u3)
            {
                auto fH = mesh->getFaceBetween(v1, v2, v3).unwrap();
                cutting_faces_.insert(v3, fH);
                predecessors_[v3] = v2;
                distances[v3] = static_cast<float>(u2t);
                direction_[v3] = 0;
                return true;
            }
        }
    }

    return false;
}

} // namespace nsga2_mesh_planner











// The Heat Method
// void WaveFront::computeVectorMap()
// {
//     const auto mesh = mesh_map_->mesh();
//     const auto& face_normals = mesh_map_->faceNormals();
    
//     // Heat method: compute gradient of potential field
//     for (auto fH : mesh->faces())
//     {
//         auto vertices = mesh->getVerticesOfFace(fH);
//         const lvr2::VertexHandle& v0 = vertices[0];
//         const lvr2::VertexHandle& v1 = vertices[1];
//         const lvr2::VertexHandle& v2 = vertices[2];
        
//         // Skip if any vertex unreached
//         if (!std::isfinite(potential_[v0]) || 
//             !std::isfinite(potential_[v1]) || 
//             !std::isfinite(potential_[v2]))
//             continue;
        
//         // Get positions and potentials
//         mesh_map::Vector p0 = mesh->getVertexPosition(v0);
//         mesh_map::Vector p1 = mesh->getVertexPosition(v1);
//         mesh_map::Vector p2 = mesh->getVertexPosition(v2);
        
//         float u0 = potential_[v0];
//         float u1 = potential_[v1];
//         float u2 = potential_[v2];
        
//         // Compute gradient on face (∇u)
//         mesh_map::Vector e1 = p1 - p0;
//         mesh_map::Vector e2 = p2 - p0;
//         mesh_map::Vector n = face_normals[fH];
        
//         // Gradient formula for triangular face
//         mesh_map::Vector grad = (n.cross(e1) * (u2 - u0) + n.cross(e2) * (u0 - u1)) / (2.0f * n.cross(e1).dot(e2));
        
//         // Normalize and negate (gradient points uphill, we want downhill)
//         mesh_map::Vector direction = -grad.normalized();
        
//         // Assign to all three vertices (average if already exists)
//         for (auto v : vertices)
//         {
//             if (vector_map_.containsKey(v))
//             {
//                 vector_map_[v] = (vector_map_[v] + direction).normalized();
//             }
//             else
//             {
//                 vector_map_.insert(v, direction);
//             }
//         }
//     }
    
//     mesh_map_->setVectorMap(vector_map_);
// }

// The heat method corrected
// void WaveFront::computeVectorMap()
// {
//     const auto mesh = mesh_map_->mesh();
//     const auto& face_normals = mesh_map_->faceNormals();
//     const auto& vertex_normals = mesh_map_->vertexNormals();
    
//     vector_map_.clear();
    
//     for (auto fH : mesh->faces())
//     {
//         auto vertices = mesh->getVerticesOfFace(fH);
//         const lvr2::VertexHandle& v0 = vertices[0];
//         const lvr2::VertexHandle& v1 = vertices[1];
//         const lvr2::VertexHandle& v2 = vertices[2];
        
//         if (!std::isfinite(potential_[v0]) || 
//             !std::isfinite(potential_[v1]) || 
//             !std::isfinite(potential_[v2]))
//             continue;
        
//         mesh_map::Vector p0 = mesh->getVertexPosition(v0);
//         mesh_map::Vector p1 = mesh->getVertexPosition(v1);
//         mesh_map::Vector p2 = mesh->getVertexPosition(v2);
        
//         float u0 = potential_[v0];
//         float u1 = potential_[v1];
//         float u2 = potential_[v2];
        
//         mesh_map::Vector e1 = p1 - p0;
//         mesh_map::Vector e2 = p2 - p0;
//         mesh_map::Vector n = face_normals[fH].normalized();
        
//         // Gradient formula
//         mesh_map::Vector grad = (n.cross(e1) * (u2 - u0) + n.cross(e2) * (u0 - u1)) / (2.0f * n.cross(e1).dot(e2));
        
//         // Project gradient onto surface (remove normal component)
//         mesh_map::Vector grad_projected = grad - n * grad.dot(n);
        
//         // Negate (gradient points uphill, we want downhill toward goal)
//         mesh_map::Vector direction = -grad_projected.normalized();
        
//         // Assign to vertices, project onto vertex normal too
//         for (auto v : vertices)
//         {
//             mesh_map::Vector vertex_n = vertex_normals[v].normalized();
//             mesh_map::Vector dir_on_vertex = direction - vertex_n * direction.dot(vertex_n);
//             dir_on_vertex = dir_on_vertex.normalized();
            
//             if (vector_map_.containsKey(v))
//             {
//                 vector_map_[v] = (vector_map_[v] + dir_on_vertex).normalized();
//             }
//             else
//             {
//                 vector_map_.insert(v, dir_on_vertex);
//             }
//         }
//     }
    
//     mesh_map_->setVectorMap(vector_map_);
// }

// void WaveFront::computeVectorMap()
// {
//     auto mesh = mesh_map_->mesh();
//     const auto& face_normals = mesh_map_->faceNormals();
    
//     vector_map_.clear();

//     for (auto fH : mesh->faces())
//     {
//         auto verts = mesh->getVerticesOfFace(fH);
//         auto v0 = verts[0];
//         auto v1 = verts[1];
//         auto v2 = verts[2];

//         // skip faces with unreached vertices
//         if (!std::isfinite(potential_[v0]) ||
//             !std::isfinite(potential_[v1]) ||
//             !std::isfinite(potential_[v2]))
//             continue;

//         // vertex positions
//         mesh_map::Vector p0 = mesh->getVertexPosition(v0);
//         mesh_map::Vector p1 = mesh->getVertexPosition(v1);
//         mesh_map::Vector p2 = mesh->getVertexPosition(v2);

//         // potentials
//         float u0 = potential_[v0];
//         float u1 = potential_[v1];
//         float u2 = potential_[v2];

//         // edges
//         mesh_map::Vector e1 = p1 - p0;
//         mesh_map::Vector e2 = p2 - p0;

//         // face normal (not normalized)
//         mesh_map::Vector N = face_normals[fH];
//         float area2 = N.length();     // 2 * area
//         if (area2 < 1e-12f) continue;

//         // ---- Correct barycentric gradients ----
//         // grad phi0 = (N × (p2 - p1)) / |N|^2
//         // grad phi1 = (N × (p0 - p2)) / |N|^2
//         // grad phi2 = (N × (p1 - p0)) / |N|^2

//         float invNorm2 = 1.0f / (N.dot(N));
//         mesh_map::Vector grad_phi0 = (N.cross(p2 - p1)) * invNorm2;
//         mesh_map::Vector grad_phi1 = (N.cross(p0 - p2)) * invNorm2;
//         mesh_map::Vector grad_phi2 = (N.cross(p1 - p0)) * invNorm2;

//         // gradient of u on face
//         mesh_map::Vector grad =
//             grad_phi0 * u0 +
//             grad_phi1 * u1 +
//             grad_phi2 * u2;

//         mesh_map::Vector grad_proj = grad - N * (grad.dot(N) / N.dot(N));
//         // downhill direction
//         mesh_map::Vector dir = grad_proj * (-1.0f);
//         float nrm = dir.length();
//         if (nrm < 1e-9f) continue;
//         dir = dir / nrm;

//         // assign (average if multiple faces contribute)
//         for (auto v : verts)
//         {
//             if (vector_map_.containsKey(v))
//             {
//                 mesh_map::Vector old = vector_map_[v];
//                 mesh_map::Vector avg = (old + dir);
//                 float l = avg.length();
//                 if (l > 1e-9f) vector_map_[v] = avg / l;
//             }
//             else
//             {
//                 vector_map_.insert(v, dir);
//             }
//         }
//     }

//     mesh_map_->setVectorMap(vector_map_);
// }
