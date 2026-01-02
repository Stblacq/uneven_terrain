#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <iomanip>
#include <cmath>
#include "gakd_mesh_planner/terrain_dynamics.hpp"

struct Transition {
    std::array<double, 6> state;      // [x, y, z, phi, theta, psi]
    std::array<double, 6> next_state;
    std::array<double, 2> control;    // [v, omega]
    double dt;
};

std::vector<Transition> load_transitions(const std::string& filepath) {
    std::vector<Transition> transitions;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        Transition t;
        
        // Actual CSV order: timestamp,position_x,position_y,position_z,orientation_roll,orientation_pitch,orientation_yaw,
        //                   control_linear_x,control_linear_y,control_linear_z,control_angular_x,control_angular_y,control_angular_z,
        //                   dt,next_position_x,next_position_y,next_position_z,next_orientation_roll,next_orientation_pitch,next_orientation_yaw,next_timestamp
        
        char comma;
        double timestamp, linear_y, linear_z, angular_x, angular_y, next_timestamp;
        
        ss >> timestamp >> comma
           >> t.state[0] >> comma  // position_x
           >> t.state[1] >> comma  // position_y
           >> t.state[2] >> comma  // position_z
           >> t.state[3] >> comma  // orientation_roll
           >> t.state[4] >> comma  // orientation_pitch
           >> t.state[5] >> comma  // orientation_yaw
           >> t.control[0] >> comma  // control_linear_x
           >> linear_y >> comma
           >> linear_z >> comma
           >> angular_x >> comma
           >> angular_y >> comma
           >> t.control[1] >> comma  // control_angular_z
           >> t.dt >> comma
           >> t.next_state[0] >> comma  // next_position_x
           >> t.next_state[1] >> comma  // next_position_y
           >> t.next_state[2] >> comma  // next_position_z
           >> t.next_state[3] >> comma  // next_orientation_roll
           >> t.next_state[4] >> comma  // next_orientation_pitch
           >> t.next_state[5] >> comma  // next_orientation_yaw
           >> next_timestamp;
        
        transitions.push_back(t);
    }
    
    std::cout << "Loaded " << transitions.size() << " transitions from " << filepath << std::endl;
    
    if (!transitions.empty()) {
        const auto& t = transitions[0];
        std::cout << "\nFirst transition:" << std::endl;
        std::cout << "  State: [" << t.state[0] << ", " << t.state[1] << ", " << t.state[2] 
                  << ", " << t.state[3] << ", " << t.state[4] << ", " << t.state[5] << "]" << std::endl;
        std::cout << "  Control: [" << t.control[0] << ", " << t.control[1] << "]" << std::endl;
        std::cout << "  dt: " << t.dt << std::endl;
        std::cout << "  Next: [" << t.next_state[0] << ", " << t.next_state[1] << ", " << t.next_state[2] 
                  << ", " << t.next_state[3] << ", " << t.next_state[4] << ", " << t.next_state[5] << "]" << std::endl;
    }
    
    return transitions;
}

struct ValidationMetrics {
    double mean_position_error;
    double rmse_position;
    double max_position_error;
    double mean_orientation_error;
    double rmse_orientation;
    double max_orientation_error;
    std::array<double, 6> mean_errors;
    std::array<double, 6> rmse_errors;
    int num_samples;
};

ValidationMetrics validate_model(
    gakd_mesh_planner::TerrainDynamics& terrain_dynamics,
    const std::vector<Transition>& transitions) {
    
    ValidationMetrics metrics;
    metrics.num_samples = transitions.size();
    
    std::array<double, 6> sum_errors = {0, 0, 0, 0, 0, 0};
    std::array<double, 6> sum_squared_errors = {0, 0, 0, 0, 0, 0};
    double sum_pos_error = 0.0;
    double sum_squared_pos_error = 0.0;
    double max_pos_error = 0.0;
    double sum_orient_error = 0.0;
    double sum_squared_orient_error = 0.0;
    double max_orient_error = 0.0;
    
    // Also track nominal model errors for comparison
    double sum_nominal_pos_error = 0.0;
    double sum_corrected_pos_error = 0.0;
    
    std::cout << "\nValidating " << transitions.size() << " transitions..." << std::endl;
    
    for (size_t i = 0; i < transitions.size(); ++i) {
        const auto& t = transitions[i];
        
        // Get nominal prediction (no GP correction) - USE dt FROM CSV
        auto nominal = terrain_dynamics.nominal_dynamics(t.state, t.control, t.dt);
        
        // Get corrected prediction (with GP) - USE dt FROM CSV
        auto corrected = terrain_dynamics.dynamics(t.state, t.control, t.dt);
        
        // Calculate error exactly like Python - simple L2 norm
        double nominal_error_sq = 0.0;
        double corrected_error_sq = 0.0;
        
        for (int j = 0; j < 6; ++j) {
            double nom_diff = t.next_state[j] - nominal[j];
            double corr_diff = t.next_state[j] - corrected[j];
            
            nominal_error_sq += nom_diff * nom_diff;
            corrected_error_sq += corr_diff * corr_diff;
        }
        
        sum_nominal_pos_error += std::sqrt(nominal_error_sq);
        sum_corrected_pos_error += std::sqrt(corrected_error_sq);
        
        // Still track per-dimension for detailed report
        std::array<double, 6> errors;
        for (int j = 0; j < 6; ++j) {
            errors[j] = t.next_state[j] - corrected[j];
            
            sum_errors[j] += errors[j];
            sum_squared_errors[j] += errors[j] * errors[j];
        }
        
        // Position error for report
        double pos_error = std::sqrt(errors[0]*errors[0] + 
                                    errors[1]*errors[1] + 
                                    errors[2]*errors[2]);
        sum_pos_error += pos_error;
        sum_squared_pos_error += pos_error * pos_error;
        max_pos_error = std::max(max_pos_error, pos_error);
        
        // Orientation error for report  
        double orient_error = std::sqrt(errors[3]*errors[3] + 
                                       errors[4]*errors[4] + 
                                       errors[5]*errors[5]);
        sum_orient_error += orient_error;
        sum_squared_orient_error += orient_error * orient_error;
        max_orient_error = std::max(max_orient_error, orient_error);
        
        // Progress indicator
        if ((i + 1) % 100 == 0) {
            std::cout << "  Processed " << (i + 1) << "/" << transitions.size() 
                      << " (" << (100 * (i + 1) / transitions.size()) << "%)" << std::endl;
        }
    }
    
    // Compute final metrics
    int n = transitions.size();
    metrics.mean_position_error = sum_pos_error / n;
    metrics.rmse_position = std::sqrt(sum_squared_pos_error / n);
    metrics.max_position_error = max_pos_error;
    
    metrics.mean_orientation_error = sum_orient_error / n;
    metrics.rmse_orientation = std::sqrt(sum_squared_orient_error / n);
    metrics.max_orientation_error = max_orient_error;
    
    for (int i = 0; i < 6; ++i) {
        metrics.mean_errors[i] = sum_errors[i] / n;
        metrics.rmse_errors[i] = std::sqrt(sum_squared_errors[i] / n);
    }
    
    // Print comparison
    double avg_nominal = sum_nominal_pos_error / n;
    double avg_corrected = sum_corrected_pos_error / n;
    double improvement = (avg_nominal - avg_corrected) / avg_nominal * 100.0;
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║ Nominal vs Corrected Comparison                          ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "Nominal Error:   " << std::fixed << std::setprecision(6) << avg_nominal << " m" << std::endl;
    std::cout << "Corrected Error: " << avg_corrected << " m" << std::endl;
    std::cout << "Improvement:     " << std::setprecision(2) << improvement << "%" << std::endl;
    
    return metrics;
}

void print_report(const ValidationMetrics& metrics) {
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       Terrain Dynamics Validation Report                 ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    
    std::cout << "\nSamples: " << metrics.num_samples << std::endl;
    
    std::cout << "\n┌─────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ Position Error (meters)                                 │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│  Mean:  " << std::setw(10) << std::fixed << std::setprecision(6) 
              << metrics.mean_position_error << " m                              │" << std::endl;
    std::cout << "│  RMSE:  " << std::setw(10) << metrics.rmse_position 
              << " m                              │" << std::endl;
    std::cout << "│  Max:   " << std::setw(10) << metrics.max_position_error 
              << " m                              │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────┘" << std::endl;
    
    std::cout << "\n┌─────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ Orientation Error (radians)                             │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│  Mean:  " << std::setw(10) << metrics.mean_orientation_error 
              << " rad                            │" << std::endl;
    std::cout << "│  RMSE:  " << std::setw(10) << metrics.rmse_orientation 
              << " rad                            │" << std::endl;
    std::cout << "│  Max:   " << std::setw(10) << metrics.max_orientation_error 
              << " rad                            │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────┘" << std::endl;
    
    std::cout << "\n┌─────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│ Per-Dimension Errors                                    │" << std::endl;
    std::cout << "├──────────┬──────────────────┬─────────────────────────┤" << std::endl;
    std::cout << "│   Dim    │   Mean Error     │     RMSE                │" << std::endl;
    std::cout << "├──────────┼──────────────────┼─────────────────────────┤" << std::endl;
    
    const char* labels[] = {"x (m)", "y (m)", "z (m)", "phi (rad)", "theta (rad)", "psi (rad)"};
    for (int i = 0; i < 6; ++i) {
        std::cout << "│ " << std::setw(8) << std::left << labels[i] << " │ "
                  << std::setw(16) << std::right << std::fixed << std::setprecision(6)
                  << metrics.mean_errors[i] << " │ "
                  << std::setw(23) << metrics.rmse_errors[i] << " │" << std::endl;
    }
    std::cout << "└──────────┴──────────────────┴─────────────────────────┘" << std::endl;
    
    // Quality assessment
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║ Quality Assessment                                        ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    
    std::string pos_quality, orient_quality;
    if (metrics.rmse_position < 0.05) {
        pos_quality = "Excellent (< 5cm)";
    } else if (metrics.rmse_position < 0.10) {
        pos_quality = "Good (< 10cm)";
    } else if (metrics.rmse_position < 0.20) {
        pos_quality = "Acceptable (< 20cm)";
    } else {
        pos_quality = "Poor (> 20cm)";
    }
    
    if (metrics.rmse_orientation < 0.05) {
        orient_quality = "Excellent (< 3°)";
    } else if (metrics.rmse_orientation < 0.10) {
        orient_quality = "Good (< 6°)";
    } else if (metrics.rmse_orientation < 0.20) {
        orient_quality = "Acceptable (< 11°)";
    } else {
        orient_quality = "Poor (> 11°)";
    }
    
    std::cout << "\nPosition prediction: " << pos_quality << std::endl;
    std::cout << "Orientation prediction: " << orient_quality << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // Hardcoded paths - update these if your paths are different
    std::string mesh_file = "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/mesh_export.obj";
    std::string curvature_file = "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/curvatures.npz";
    std::string model_file = "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/gp_matrices.npz";
    std::string data_file = "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/collected_data.csv";
    
    // Allow override via command line if desired
    if (argc >= 2) data_file = argv[1];
    if (argc >= 3) model_file = argv[2];
    if (argc >= 4) mesh_file = argv[3];
    if (argc >= 5) curvature_file = argv[4];
    
    try {
        std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║    Terrain Dynamics Validation Tool                      ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
        
        std::cout << "\nInitializing terrain dynamics..." << std::endl;
        std::cout << "  Mesh: " << mesh_file << std::endl;
        std::cout << "  Curvatures: " << curvature_file << std::endl;
        std::cout << "  Model: " << model_file << std::endl;
        
        gakd_mesh_planner::TerrainDynamics terrain_dynamics(
            mesh_file,
            curvature_file,
            model_file,
            0.1,   // dt
            0.5,   // epsilon
            0.16   // neighborhood_radius
        );
        
        std::cout << "\nLoading transition data..." << std::endl;
        std::cout << "  Data file: " << data_file << std::endl;
        auto transitions = load_transitions(data_file);
        
        if (transitions.empty()) {
            std::cerr << "Error: No transitions loaded!" << std::endl;
            return 1;
        }
        
        auto metrics = validate_model(terrain_dynamics, transitions);
        
        print_report(metrics);
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << std::endl;
        return 1;
    }
}