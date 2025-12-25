#ifndef MPPI_MESH_PLANNER_DATA_COLLECTOR_H
#define MPPI_MESH_PLANNER_DATA_COLLECTOR_H

#include <string>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

namespace mppi_mesh_planner
{

struct DataPoint {
    double timestamp;
    double position_x, position_y, position_z;
    double orientation_roll, orientation_pitch, orientation_yaw;
    double control_linear_x, control_linear_y, control_linear_z;
    double control_angular_x, control_angular_y, control_angular_z;
    double dt;
    double next_position_x, next_position_y, next_position_z;
    double next_orientation_roll, next_orientation_pitch, next_orientation_yaw;
    double next_timestamp;
};

class DataCollector {
private:
    std::string filepath_;
    std::ofstream file_;
    bool header_written_;
    bool enabled_;
    std::mutex mutex_;
    rclcpp::Logger logger_;
    
    void writeHeader();
    
public:
    explicit DataCollector(const std::string& filepath = "", 
                          rclcpp::Logger logger = rclcpp::get_logger("DataCollector"));
    ~DataCollector();
    
    // Enable/disable data collection
    void setEnabled(bool enabled);
    bool isEnabled() const;
    
    // Write a data point (only if enabled)
    void writeDataPoint(const DataPoint& data);
    
    // Collect data from pose states and control
    void collectTransition(
        const geometry_msgs::msg::PoseStamped& previous_state,
        const geometry_msgs::msg::PoseStamped& current_state,
        double control_linear_x,
        double control_angular_z);
    
    // Get file path
    std::string getFilePath() const;
    
    // Close and reopen file (useful for manual flushing)
    void flush();
};

} // namespace mppi_mesh_planner

#endif // MPPI_MESH_PLANNER_DATA_COLLECTOR_H