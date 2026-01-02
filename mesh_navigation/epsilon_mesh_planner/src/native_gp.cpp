#include "epsilon_mesh_planner/native_gp.hpp"
#include <cnpy.h>
#include <iostream>
#include <iomanip>
#include <stdexcept>

namespace epsilon_mesh_planner {

NativeGP::NativeGP() : is_loaded_(false), kernel_variance_(1.0) {
}

void NativeGP::load_from_npz(const std::string& filename) {
    std::cout << "Loading GP matrices from: " << filename << std::endl;
    
    try {
        cnpy::npz_t npz = cnpy::npz_load(filename);
        
        // Load inducing points Z
        // NumPy arrays are row-major (C order), Eigen is column-major by default
        // Use RowMajor map to load correctly
        auto Z_arr = npz["Z"];
        double* Z_data = Z_arr.data<double>();
        size_t n_inducing = Z_arr.shape[0];
        size_t input_dim = Z_arr.shape[1];
        Z_ = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            Z_data, n_inducing, input_dim);
        
        // Load alpha coefficients
        auto alpha_arr = npz["alpha"];
        double* alpha_data = alpha_arr.data<double>();
        size_t output_dim = alpha_arr.shape[1];
        alpha_ = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
            alpha_data, n_inducing, output_dim);
        
        // Load kernel parameters
        auto ls_arr = npz["kernel_lengthscales"];
        double* ls_data = ls_arr.data<double>();
        lengthscales_ = Eigen::Map<Eigen::VectorXd>(ls_data, input_dim);
        
        auto kv_arr = npz["kernel_variance"];
        kernel_variance_ = kv_arr.data<double>()[0];
        
        // Load input scalers
        auto xmean_arr = npz["scaler_X_mean"];
        double* xmean_data = xmean_arr.data<double>();
        scaler_X_mean_ = Eigen::Map<Eigen::VectorXd>(xmean_data, input_dim);
        
        auto xscale_arr = npz["scaler_X_scale"];
        double* xscale_data = xscale_arr.data<double>();
        scaler_X_scale_ = Eigen::Map<Eigen::VectorXd>(xscale_data, input_dim);
        
        // Load output scalers
        auto ymean_arr = npz["scaler_y_mean"];
        double* ymean_data = ymean_arr.data<double>();
        scaler_y_mean_ = Eigen::Map<Eigen::VectorXd>(ymean_data, output_dim);
        
        auto yscale_arr = npz["scaler_y_scale"];
        double* yscale_data = yscale_arr.data<double>();
        scaler_y_scale_ = Eigen::Map<Eigen::VectorXd>(yscale_data, output_dim);
        
        is_loaded_ = true;
        
        std::cout << "✓ GP matrices loaded successfully" << std::endl;
        std::cout << "  Inducing points: " << n_inducing << " × " << input_dim << std::endl;
        std::cout << "  Output dimension: " << output_dim << std::endl;
        std::cout << "  Kernel variance: " << kernel_variance_ << std::endl;
        
        // Verify first inducing point matches Python
        std::cout << "\n  Verification - First inducing point Z[0]:" << std::endl;
        std::cout << "    Expected: [-0.37985728, 1.07639685, 0.03098423, -0.02622385, ...]" << std::endl;
        std::cout << "    Got:      [" << std::fixed << std::setprecision(8);
        for (int i = 0; i < std::min(4, (int)input_dim); ++i) {
            std::cout << Z_(0, i);
            if (i < 3) std::cout << ", ";
        }
        std::cout << ", ...]" << std::endl;
        
        std::cout << "  First alpha values:" << std::endl;
        std::cout << "    Expected: [1.654292, 4.81522549, 11.83777409, ...]" << std::endl;
        std::cout << "    Got:      [";
        for (int i = 0; i < std::min(3, (int)output_dim); ++i) {
            std::cout << alpha_(0, i);
            if (i < 2) std::cout << ", ";
        }
        std::cout << ", ...]" << std::endl;
        
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load GP matrices: " + std::string(e.what()));
    }
}

Eigen::VectorXd NativeGP::compute_kernel(const Eigen::VectorXd& x_scaled) {
    size_t n_inducing = Z_.rows();
    Eigen::VectorXd K_xZ(n_inducing);
    
    // Compute kernel between x and each inducing point
    for (size_t i = 0; i < n_inducing; ++i) {
        // Compute scaled difference: (x - Z[i]) / lengthscales
        Eigen::VectorXd diff = x_scaled - Z_.row(i).transpose();
        Eigen::VectorXd scaled_diff = diff.array() / lengthscales_.array();
        
        // Squared distance: r² = sum(scaled_diff²)
        double r2 = scaled_diff.squaredNorm();
        
        // Matérn 5/2 kernel: k(r) = σ² * (1 + √5*r + 5*r²/3) * exp(-√5*r)
        double r = std::sqrt(r2 + 1e-12);  // Add small epsilon to avoid division by zero
        double sqrt5_r = std::sqrt(5.0) * r;
        
        K_xZ(i) = kernel_variance_ * (1.0 + sqrt5_r + 5.0 * r2 / 3.0) * std::exp(-sqrt5_r);
    }
    
    return K_xZ;
}

Eigen::VectorXd NativeGP::predict_mean(const Eigen::VectorXd& x) {
    if (!is_loaded_) {
        throw std::runtime_error("GP model not loaded. Call load_from_npz() first.");
    }
    
    // Step 1: Standardize input
    // x_scaled = (x - mean) / scale
    Eigen::VectorXd x_scaled = (x.array() - scaler_X_mean_.array()) / scaler_X_scale_.array();
    
    // Step 2: Compute kernel vector K*Z
    Eigen::VectorXd K_xZ = compute_kernel(x_scaled);
    
    // Step 3: Predict in scaled space
    // y_scaled = K*Z @ alpha
    Eigen::VectorXd y_scaled = K_xZ.transpose() * alpha_;
    
    // Step 4: Destandardize output
    // y = y_scaled * scale + mean
    Eigen::VectorXd y = y_scaled.array() * scaler_y_scale_.array() + scaler_y_mean_.array();
    
    return y;
}

} // namespace epsilon_mesh_planner