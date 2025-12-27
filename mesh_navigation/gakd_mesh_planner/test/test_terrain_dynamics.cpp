#include <iostream>
#include <chrono>
#include <iomanip>
#include "gakd_mesh_planner/terrain_dynamics.hpp"

int main() {
    try {
        std::cout << "=== Component Breakdown Test ===" << std::endl << std::endl;
        
        gakd_mesh_planner::TerrainDynamics terrain(
            "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/mesh_export.obj",
            "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/curvatures.npz",
            "/home/otto/workspaces/RL_ws/src/mesh_navigation/gakd_mesh_planner/data/gp_residual_export",
            0.1, 0.5, 0.16
        );
        
        std::array<double,6> state = {0, 0, 0, 0, 0, 0};
        std::array<double,2> control = {1.0, 0.1};
        
        // Warm up
        for (int i = 0; i < 20; ++i) {
            terrain.dynamics(state, control);
        }
        
        const int iterations = 1000;
        
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
        // dynamics() = nominal + descriptor + TF + overhead
        // TF + overhead = dynamics - nominal - descriptor
        
        double tf_plus_overhead = avg_dynamics_ms - avg_nominal_ms - avg_descriptor_ms;
        
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
        print_row("TensorFlow + overhead", tf_plus_overhead, avg_dynamics_ms);
        
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
        std::cout << "  - TensorFlow inference + overhead: " << std::fixed << std::setprecision(4)
                  << tf_plus_overhead << " ms" << std::endl;
        
        // Scaled and Hz for TOTAL only
        const double scale_factor = 50000.0;
        double total_scaled = avg_dynamics_ms * scale_factor;
        double total_freq = 1.0 / (total_scaled / 1000.0);  // 1 / seconds for 50000 calls
        
        std::cout << "\n=== TRAJECTORY METRICS (50000 steps) ===" << std::endl;
        std::cout << "Total time for 50000 dynamics calls: " << std::fixed << std::setprecision(2)
                  << total_scaled << " ms (" << std::setprecision(2) << (total_scaled/1000.0) 
                  << " seconds)" << std::endl;
        std::cout << "Frequency:                           " << std::fixed << std::setprecision(4)
                  << total_freq << " Hz" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}