#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cmath>
#include <functional>
#include "gakd_mesh_planner/terrain_dynamics.hpp"

struct CSVRow {
    double position_x, position_y, position_z;
    double orientation_roll, orientation_pitch, orientation_yaw;
    double control_linear_x, control_angular_z;
    double next_position_x, next_position_y, next_position_z;
    double next_orientation_roll, next_orientation_pitch, next_orientation_yaw;
    double dt;
};

std::vector<CSVRow> load_csv(const std::string& filename) {
    std::vector<CSVRow> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open " << filename << std::endl;
        return data;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        CSVRow row;
        char comma;
        double timestamp, linear_y, linear_z, angular_x, angular_y, next_timestamp;
        
        // Actual CSV order: timestamp,position_x,position_y,position_z,orientation_roll,orientation_pitch,orientation_yaw,
        //                   control_linear_x,control_linear_y,control_linear_z,control_angular_x,control_angular_y,control_angular_z,
        //                   dt,next_position_x,next_position_y,next_position_z,next_orientation_roll,next_orientation_pitch,next_orientation_yaw,next_timestamp
        
        ss >> timestamp >> comma
           >> row.position_x >> comma
           >> row.position_y >> comma
           >> row.position_z >> comma
           >> row.orientation_roll >> comma
           >> row.orientation_pitch >> comma
           >> row.orientation_yaw >> comma
           >> row.control_linear_x >> comma
           >> linear_y >> comma
           >> linear_z >> comma
           >> angular_x >> comma
           >> angular_y >> comma
           >> row.control_angular_z >> comma
           >> row.dt >> comma
           >> row.next_position_x >> comma
           >> row.next_position_y >> comma
           >> row.next_position_z >> comma
           >> row.next_orientation_roll >> comma
           >> row.next_orientation_pitch >> comma
           >> row.next_orientation_yaw >> comma
           >> next_timestamp;
        
        data.push_back(row);
    }
    
    return data;
}

void print_first_row(const std::vector<CSVRow>& data) {
    if (!data.empty()) {
        const auto& r = data[0];
        std::cout << "\nFirst CSV row:" << std::endl;
        std::cout << "  State: [" << r.position_x << ", " << r.position_y << ", " << r.position_z 
                  << ", " << r.orientation_roll << ", " << r.orientation_pitch << ", " << r.orientation_yaw << "]" << std::endl;
        std::cout << "  Control: [" << r.control_linear_x << ", " << r.control_angular_z << "]" << std::endl;
        std::cout << "  dt: " << r.dt << std::endl;
        std::cout << "  Next: [" << r.next_position_x << ", " << r.next_position_y << ", " << r.next_position_z 
                  << ", " << r.next_orientation_roll << ", " << r.next_orientation_pitch << ", " << r.next_orientation_yaw << "]" << std::endl;
    }
}

double evaluate_dynamics(gakd_mesh_planner::TerrainDynamics& terrain,
                        const std::vector<CSVRow>& dataset,
                        std::function<std::array<double,6>(gakd_mesh_planner::TerrainDynamics&, std::array<double,6>, std::array<double,2>, double)> dynamics_function) {
    std::vector<double> errors;
    
    for (const auto& row : dataset) {
        // Current state [x, y, z, roll, pitch, yaw]
        std::array<double, 6> state = {
            row.position_x, row.position_y, row.position_z,
            row.orientation_roll, row.orientation_pitch, row.orientation_yaw
        };
        
        // Control [linear_x, angular_z]
        std::array<double, 2> control = {
            row.control_linear_x, row.control_angular_z
        };
        
        // Predict next state WITH dt FROM ROW
        auto predicted = dynamics_function(terrain, state, control, row.dt);
        
        // True next state
        std::array<double, 6> true_state = {
            row.next_position_x, row.next_position_y, row.next_position_z,
            row.next_orientation_roll, row.next_orientation_pitch, row.next_orientation_yaw
        };
        
        // Calculate error
        double error = 0.0;
        for (int i = 0; i < 6; ++i) {
            double diff = predicted[i] - true_state[i];
            error += diff * diff;
        }
        error = std::sqrt(error);
        errors.push_back(error);
    }
    
    // Return mean error
    double sum = 0.0;
    for (double e : errors) sum += e;
    return sum / errors.size();
}

int main() {
    try {
        std::cout << "=== Native GP Performance Test ===" << std::endl << std::endl;
        
        gakd_mesh_planner::TerrainDynamics terrain(
            "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/mesh_export.obj",
            "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/curvatures.npz",
            "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/gp_matrices.npz",
            0.1, 0.5, 0.16
        );
        
        // Test h() function matches Python
        std::cout << "\n=== Testing h() function ===" << std::endl;
        std::vector<std::pair<double, double>> test_points = {
            {0.0, 0.0},
            {8.67742, -0.776569},
            {10.0, 5.0},
            {5.0, -3.0}
        };
        
        std::cout << "Compare these with Python h() function:" << std::endl;
        for (const auto& [x, y] : test_points) {
            double z = terrain.h(x, y);
            std::cout << "  h(" << std::fixed << std::setprecision(2) << x << ", " << y << ") = " 
                      << std::setprecision(6) << z << std::endl;
        }
        
        // Test descriptor matches Python
        std::cout << "\n=== Testing query_descriptor() ===" << std::endl;
        std::vector<std::array<double, 3>> test_positions = {
            {8.677, -0.777, 0.138},
            {0.0, 0.0, 0.3},
            {10.0, 5.0, 0.28}
        };
        
        std::cout << "Compare these with Python get_descriptor():" << std::endl;
        for (const auto& pos : test_positions) {
            Eigen::Vector3d point(pos[0], pos[1], pos[2]);
            auto desc = terrain.query_descriptor(point);
            std::cout << "\nDescriptor at (" << pos[0] << ", " << pos[1] << ", " << pos[2] << "):" << std::endl;
            std::cout << "  [" << std::fixed << std::setprecision(6);
            for (size_t i = 0; i < desc.size(); ++i) {
                std::cout << desc[i];
                if (i < desc.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Test nominal_dynamics matches Python
        std::cout << "\n=== Testing nominal_dynamics() ===" << std::endl;
        struct TestCase {
            std::array<double, 6> state;
            std::array<double, 2> control;
            double dt;
        };
        
        std::vector<TestCase> test_cases = {
            {{8.677, -0.777, 0.138, 0.0, 0.0, 0.5}, {1.0, 0.1}, 0.1},
            {{0.0, 0.0, 0.3, 0.0, 0.0, 0.0}, {0.5, 0.0}, 0.1},
            {{10.0, 5.0, 0.28, 0.1, -0.05, 1.2}, {0.8, -0.2}, 0.1}
        };
        
        std::cout << "Compare these with Python nominal_dynamics():" << std::endl;
        for (size_t i = 0; i < test_cases.size(); ++i) {
            const auto& tc = test_cases[i];
            auto result = terrain.nominal_dynamics(tc.state, tc.control);
            
            std::cout << "\nTest " << (i+1) << ":" << std::endl;
            std::cout << "  State:   [" << std::setprecision(3);
            for (int j = 0; j < 6; ++j) {
                std::cout << tc.state[j];
                if (j < 5) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  Control: [" << tc.control[0] << ", " << tc.control[1] << "]" << std::endl;
            std::cout << "  dt:      " << tc.dt << std::endl;
            std::cout << "  Result:  [" << std::setprecision(6);
            for (int j = 0; j < 6; ++j) {
                std::cout << result[j];
                if (j < 5) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        
        // Test GP residual prediction matches Python
        std::cout << "\n=== Testing GP Residual Prediction ===" << std::endl;
        std::array<double, 6> test_state = {-0.038495, -1.032783, 0.290499, -0.004520, 0.008242, -0.109493};
        std::array<double, 2> test_control = {0.171007, -0.417102};
        
        // Get descriptor
        Eigen::Vector3d pos(test_state[0], test_state[1], test_state[2]);
        auto descriptor = terrain.query_descriptor(pos);
        
        // Build GP input: [state(6), control(2), descriptor(8)] = 16
        Eigen::VectorXd gp_input(16);
        for (int i = 0; i < 6; ++i) gp_input[i] = test_state[i];
        for (int i = 0; i < 2; ++i) gp_input[6 + i] = test_control[i];
        for (int i = 0; i < 8; ++i) gp_input[8 + i] = descriptor[i];
        
        std::cout << "GP input: [" << std::fixed << std::setprecision(6);
        for (int i = 0; i < 16; ++i) {
            std::cout << gp_input[i];
            if (i < 15) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Get residual prediction (this is what gets added to nominal)
        auto nominal_pred = terrain.nominal_dynamics(test_state, test_control, 0.06);
        auto corrected_pred = terrain.dynamics(test_state, test_control);
        
        std::cout << "\nNominal prediction: [";
        for (int i = 0; i < 6; ++i) {
            std::cout << nominal_pred[i];
            if (i < 5) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "Corrected prediction: [";
        for (int i = 0; i < 6; ++i) {
            std::cout << corrected_pred[i];
            if (i < 5) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::cout << "GP residual (corrected - nominal): [";
        for (int i = 0; i < 6; ++i) {
            std::cout << (corrected_pred[i] - nominal_pred[i]);
            if (i < 5) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Python GP residual:                [0.00243024, -0.00498812, -0.00269332, -0.01160899, -0.00464564, 0.0195931]" << std::endl;
        
        // Evaluate nominal dynamics on CSV
        std::cout << "\n=== Evaluating Nominal Dynamics ===" << std::endl;
        std::string csv_file = "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/collected_data.csv";
        auto dataset = load_csv(csv_file);
        
        if (!dataset.empty()) {
            std::cout << "Loaded " << dataset.size() << " samples" << std::endl;
            print_first_row(dataset);
            
            auto nominal_func = [](gakd_mesh_planner::TerrainDynamics& t, std::array<double,6> state, std::array<double,2> control, double dt) {
                return t.nominal_dynamics(state, control, dt);
            };
            
            double mean_error = evaluate_dynamics(terrain, dataset, nominal_func);
            std::cout << "Mean error: " << std::fixed << std::setprecision(6) << mean_error << std::endl;
        }
        
        std::cout << "\n=== Performance Benchmark ===" << std::endl;
        
        std::array<double,6> state = {0, 0, 0, 0, 0, 0};
        std::array<double,2> control = {1.0, 0.1};
        
        // Warm up
        std::cout << "Warming up..." << std::endl;
        for (int i = 0; i < 20; ++i) {
            terrain.dynamics(state, control);
        }
        
        const int iterations = 1000;
        std::cout << "Running " << iterations << " iterations...\n" << std::endl;
        
        // TEST 1: Full dynamics
        long long total_dynamics = 0;
        std::array<double,6> current_state = state;
        for (int i = 0; i < iterations; ++i) {
            current_state[0] = i * 0.15;
            current_state[1] = i * 0.15;
            
            auto start = std::chrono::high_resolution_clock::now();
            current_state = terrain.dynamics(current_state, control);
            auto end = std::chrono::high_resolution_clock::now();
            total_dynamics += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        double avg_dynamics_ms = total_dynamics / (iterations * 1000000.0);
        
        // TEST 2: Just nominal_dynamics
        long long total_nominal = 0;
        current_state = state;
        for (int i = 0; i < iterations; ++i) {
            current_state[0] = i * 0.15;
            current_state[1] = i * 0.15;
            
            auto start = std::chrono::high_resolution_clock::now();
            auto result = terrain.nominal_dynamics(current_state, control);
            auto end = std::chrono::high_resolution_clock::now();
            total_nominal += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        double avg_nominal_ms = total_nominal / (iterations * 1000000.0);
        
        // TEST 3: Just query_descriptor
        long long total_descriptor = 0;
        for (int i = 0; i < iterations; ++i) {
            gakd_mesh_planner::Vector3d pos(i * 0.15, i * 0.15, 0.0);
            
            auto start = std::chrono::high_resolution_clock::now();
            auto desc = terrain.query_descriptor(pos);
            auto end = std::chrono::high_resolution_clock::now();
            total_descriptor += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        }
        double avg_descriptor_ms = total_descriptor / (iterations * 1000000.0);
        
        // CALCULATE BREAKDOWN
        // dynamics() = nominal + descriptor + GP + overhead
        double gp_plus_overhead = avg_dynamics_ms - avg_nominal_ms - avg_descriptor_ms;
        
        std::cout << "=== MEASURED BREAKDOWN ===" << std::endl;
        std::cout << "Average over " << iterations << " iterations:" << std::endl;
        std::cout << "┌─────────────────────────────────────┬──────────────┬────────────┐" << std::endl;
        std::cout << "│ Component                           │ Time (ms)    │ Percentage │" << std::endl;
        std::cout << "├─────────────────────────────────────┼──────────────┼────────────┤" << std::endl;
        
        auto print_row = [&](const std::string& name, double time_ms, double total) {
            std::cout << "│ " << std::setw(35) << std::left << name << " │ "
                      << std::setw(12) << std::right << std::fixed << std::setprecision(4) << time_ms
                      << " │ " << std::setw(9) << std::right << std::fixed << std::setprecision(1)
                      << (time_ms / total * 100.0) << "% │" << std::endl;
        };
        
        print_row("nominal_dynamics()", avg_nominal_ms, avg_dynamics_ms);
        print_row("query_descriptor()", avg_descriptor_ms, avg_dynamics_ms);
        print_row("Native GP + overhead", gp_plus_overhead, avg_dynamics_ms);
        
        std::cout << "├─────────────────────────────────────┼──────────────┼────────────┤" << std::endl;
        print_row("TOTAL dynamics()", avg_dynamics_ms, avg_dynamics_ms);
        std::cout << "└─────────────────────────────────────┴──────────────┴────────────┘" << std::endl;
        
        std::cout << "\n=== DETAILED BREAKDOWN ===" << std::endl;
        std::cout << "Full dynamics():                     " << std::fixed << std::setprecision(4)
                  << avg_dynamics_ms << " ms" << std::endl;
        std::cout << "  - nominal_dynamics():              " << std::fixed << std::setprecision(4)
                  << avg_nominal_ms << " ms" << std::endl;
        std::cout << "  - query_descriptor():              " << std::fixed << std::setprecision(4)
                  << avg_descriptor_ms << " ms" << std::endl;
        std::cout << "  - Native GP + overhead:            " << std::fixed << std::setprecision(4)
                  << gp_plus_overhead << " ms" << std::endl;
        
        // Scaled and Hz for TOTAL only
        const double scale_factor = 50000.0;
        double total_scaled = avg_dynamics_ms * scale_factor;
        double total_freq = 1.0 / (total_scaled / 1000.0);
        
        std::cout << "\n=== TRAJECTORY METRICS (50000 steps) ===" << std::endl;
        std::cout << "Total time for 50000 dynamics calls: " << std::fixed << std::setprecision(2)
                  << total_scaled << " ms (" << std::setprecision(2) << (total_scaled/1000.0) 
                  << " seconds)" << std::endl;
        std::cout << "Frequency:                           " << std::fixed << std::setprecision(4)
                  << total_freq << " Hz" << std::endl;
        
        // Compare with TensorFlow baseline
        double tf_baseline = 0.0483;  // Your original TF time
        double speedup = tf_baseline / avg_dynamics_ms;
        
        std::cout << "\n=== COMPARISON WITH TENSORFLOW ===" << std::endl;
        std::cout << "TensorFlow baseline:  " << std::fixed << std::setprecision(4) 
                  << tf_baseline << " ms" << std::endl;
        std::cout << "Native GP:            " << std::fixed << std::setprecision(4) 
                  << avg_dynamics_ms << " ms" << std::endl;
        std::cout << "Speedup:              " << std::fixed << std::setprecision(1) 
                  << speedup << "x faster" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}