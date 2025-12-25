#include <iostream>
#include <chrono>
#include <iomanip>
#include "epsilon_mesh_planner/terrain_dynamics.hpp"

void print_timing(const std::string& label, const std::chrono::microseconds& duration) {
    std::cout << std::setw(40) << std::left << label << ": " 
              << std::setw(10) << std::right << std::fixed << std::setprecision(3) 
              << duration.count() / 1000.0 << " ms" << std::endl;
}

int main() {
    try {
        std::cout << "=== TerrainDynamics Performance Test ===" << std::endl;
        std::cout << std::endl;
        
        // ===== INITIALIZATION =====
        std::cout << "=== Step 1: Initializing TerrainDynamics ===" << std::endl;
        auto init_start = std::chrono::high_resolution_clock::now();
        
        epsilon_mesh_planner::TerrainDynamics terrain(
            "/home/otto/workspaces/RL_ws/src/mesh_navigation/epsilon_mesh_planner/data/mesh_export.obj",
            "/home/otto/workspaces/RL_ws/src/mesh_navigation/epsilon_mesh_planner/data/vertex_corrections.npy",
            0.1,   // dt
            0.5    // epsilon
        );
        
        auto init_end = std::chrono::high_resolution_clock::now();
        auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(init_end - init_start);
        print_timing("Total initialization time", std::chrono::duration_cast<std::chrono::microseconds>(init_duration));
        std::cout << std::endl;
        
        // ===== TEST DYNAMICS (SINGLE CALL) =====
        std::cout << "=== Step 2: Testing dynamics() (single call) ===" << std::endl;
        std::array<double,6> state = {0, 0, 0, 0, 0, 0};
        std::array<double,2> control = {1.0, 0.1};
        
        auto dynamics_start = std::chrono::high_resolution_clock::now();
        auto next_state = terrain.dynamics(state, control);
        auto dynamics_end = std::chrono::high_resolution_clock::now();
        auto dynamics_duration = std::chrono::duration_cast<std::chrono::microseconds>(dynamics_end - dynamics_start);
        
        std::cout << "Next state: ";
        for (auto v : next_state) std::cout << v << " ";
        std::cout << std::endl;
        print_timing("dynamics() execution time", dynamics_duration);
        std::cout << std::endl;
        
        // ===== BENCHMARK: MULTIPLE CALLS =====
        std::cout << "=== Step 3: Benchmarking multiple dynamics() calls ===" << std::endl;
        const int num_iterations = 1000;
        
        auto benchmark_start = std::chrono::high_resolution_clock::now();
        
        std::array<double,6> current_state = state;
        for (int i = 0; i < num_iterations; ++i) {
            current_state = terrain.dynamics(current_state, control);
        }
        
        auto benchmark_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(benchmark_end - benchmark_start);
        auto avg_duration = total_duration.count() / num_iterations;
        
        std::cout << "Ran " << num_iterations << " iterations" << std::endl;
        print_timing("Total time", total_duration);
        std::cout << std::setw(40) << std::left << "Average per call" << ": " 
                  << std::setw(10) << std::right << std::fixed << std::setprecision(3) 
                  << avg_duration / 1000.0 << " ms" << std::endl;
        std::cout << std::setw(40) << std::left << "Calls per second" << ": " 
                  << std::setw(10) << std::right << std::fixed << std::setprecision(1) 
                  << 1000000.0 / avg_duration << " Hz" << std::endl;
        std::cout << std::endl;
        
        // ===== PLANNING PROJECTION =====
        std::cout << "=== Step 4: Planning Performance Projection ===" << std::endl;
        int pop_size = 100;
        int generations = 50;
        int horizon = 10;
        long long total_calls = (long long)pop_size * generations * horizon;
        double estimated_time = (total_calls * avg_duration) / 1000000.0; // seconds
        
        std::cout << "For Genetic Algorithm with:" << std::endl;
        std::cout << "  - Population: " << pop_size << std::endl;
        std::cout << "  - Generations: " << generations << std::endl;
        std::cout << "  - Horizon: " << horizon << std::endl;
        std::cout << "  - Total dynamics() calls: " << total_calls << std::endl;
        std::cout << "  - Estimated time: " << std::fixed << std::setprecision(2) 
                  << estimated_time << " seconds (" 
                  << estimated_time / 60.0 << " minutes)" << std::endl;
        std::cout << std::endl;
        
        // ===== SUMMARY TABLE =====
        std::cout << "=== Performance Summary ===" << std::endl;
        std::cout << "┌────────────────────────────────────────┬──────────────┐" << std::endl;
        std::cout << "│ Metric                                 │ Value        │" << std::endl;
        std::cout << "├────────────────────────────────────────┼──────────────┤" << std::endl;
        std::cout << "│ Single dynamics() call                 │ " 
                  << std::setw(9) << std::right << std::fixed << std::setprecision(3) 
                  << dynamics_duration.count() / 1000.0 << " ms │" << std::endl;
        std::cout << "│ Average (1000 calls)                   │ " 
                  << std::setw(9) << std::right << std::fixed << std::setprecision(3) 
                  << avg_duration / 1000.0 << " ms │" << std::endl;
        std::cout << "│ Throughput                             │ " 
                  << std::setw(7) << std::right << std::fixed << std::setprecision(0) 
                  << 1000000.0 / avg_duration << " Hz │" << std::endl;
        std::cout << "└────────────────────────────────────────┴──────────────┘" << std::endl;
        std::cout << std::endl;
        
        std::cout << "=== All tests passed! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "=== Fatal Error ===" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}