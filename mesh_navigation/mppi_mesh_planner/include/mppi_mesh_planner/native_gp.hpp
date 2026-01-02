#ifndef MPPI_MESH_PLANNER_NATIVE_GP_HPP
#define MPPI_MESH_PLANNER_NATIVE_GP_HPP

#include <Eigen/Dense>
#include <string>
#include <cmath>

namespace mppi_mesh_planner {

class NativeGP {
private:
    Eigen::MatrixXd Z_;
    Eigen::MatrixXd alpha_;
    Eigen::VectorXd lengthscales_;
    double kernel_variance_;
    
    Eigen::VectorXd scaler_X_mean_;
    Eigen::VectorXd scaler_X_scale_;
    Eigen::VectorXd scaler_y_mean_;
    Eigen::VectorXd scaler_y_scale_;
    
    bool is_loaded_;
    
public:
    NativeGP();
    void load_from_npz(const std::string& filename);
    Eigen::VectorXd predict_mean(const Eigen::VectorXd& x);
    bool is_loaded() const { return is_loaded_; }
    
private:
    Eigen::VectorXd compute_kernel(const Eigen::VectorXd& x);
};

} // namespace mppi_mesh_planner

#endif