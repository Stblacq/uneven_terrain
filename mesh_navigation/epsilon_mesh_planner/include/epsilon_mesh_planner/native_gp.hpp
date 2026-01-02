#ifndef EPSILON_MESH_PLANNER_NATIVE_GP_HPP
#define EPSILON_MESH_PLANNER_NATIVE_GP_HPP

#include <Eigen/Dense>
#include <string>
#include <cmath>

namespace epsilon_mesh_planner {

class NativeGP {
private:
    // GP matrices
    Eigen::MatrixXd Z_;              // Inducing points (n_inducing × input_dim)
    Eigen::MatrixXd alpha_;          // Precomputed coefficients (n_inducing × output_dim)
    Eigen::VectorXd lengthscales_;   // ARD lengthscales (input_dim,)
    double kernel_variance_;         // Kernel variance (scalar)
    
    // Scalers
    Eigen::VectorXd scaler_X_mean_;  // Input mean (input_dim,)
    Eigen::VectorXd scaler_X_scale_; // Input scale (input_dim,)
    Eigen::VectorXd scaler_y_mean_;  // Output mean (output_dim,)
    Eigen::VectorXd scaler_y_scale_; // Output scale (output_dim,)
    
    bool is_loaded_;
    
public:
    NativeGP();
    
    /**
     * Load GP parameters from NPZ file.
     * Expected arrays:
     *   - Z: inducing points (n_inducing × input_dim)
     *   - alpha: prediction coefficients (n_inducing × output_dim)
     *   - kernel_lengthscales: ARD lengthscales (input_dim,)
     *   - kernel_variance: kernel variance (1,)
     *   - scaler_X_mean: input mean (input_dim,)
     *   - scaler_X_scale: input scale (input_dim,)
     *   - scaler_y_mean: output mean (output_dim,)
     *   - scaler_y_scale: output scale (output_dim,)
     */
    void load_from_npz(const std::string& filename);
    
    /**
     * Predict mean using matrix-based GP prediction.
     * Implements: y* = K*Z @ alpha with proper standardization
     * 
     * @param x Input vector (input_dim,)
     * @return Predicted output (output_dim,)
     */
    Eigen::VectorXd predict_mean(const Eigen::VectorXd& x);
    
    bool is_loaded() const { return is_loaded_; }
    
private:
    /**
     * Compute Matérn 5/2 kernel with ARD.
     * k(x, z) = σ² * (1 + √5*r + 5*r²/3) * exp(-√5*r)
     * where r² = sum((x - z)² / lengthscales²)
     */
    Eigen::VectorXd compute_kernel(const Eigen::VectorXd& x);
};

} // namespace epsilon_mesh_planner

#endif