#include "nsga2_mesh_planner/data_collector.h"

namespace nsga2_mesh_planner
{

DataCollector::DataCollector(const std::string& filepath, rclcpp::Logger logger)
    : header_written_(false), enabled_(false), logger_(logger)
{
    // Determine file path
    if (filepath.empty()) {
        std::string home_dir = std::getenv("HOME");
        if (home_dir.empty()) {
            home_dir = "/tmp";
        }
        filepath_ = home_dir + "/robot_dataset.csv";
    } else {
        filepath_ = filepath;
    }
    
    // Check if file exists to determine if we need to write header
    header_written_ = std::filesystem::exists(filepath_);
    
    RCLCPP_INFO(logger_, "DataCollector initialized with file: %s (enabled: %s)", 
                filepath_.c_str(), enabled_ ? "true" : "false");
}

DataCollector::~DataCollector()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.close();
        RCLCPP_INFO(logger_, "DataCollector closed file: %s", filepath_.c_str());
    }
}

void DataCollector::writeHeader()
{
    file_ << "timestamp,position_x,position_y,position_z,"
          << "orientation_roll,orientation_pitch,orientation_yaw,"
          << "control_linear_x,control_linear_y,control_linear_z,"
          << "control_angular_x,control_angular_y,control_angular_z,"
          << "dt,next_position_x,next_position_y,next_position_z,"
          << "next_orientation_roll,next_orientation_pitch,next_orientation_yaw,"
          << "next_timestamp\n";
    file_.flush();
    header_written_ = true;
}

void DataCollector::setEnabled(bool enabled)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (enabled == enabled_) {
        return; // No change
    }
    
    enabled_ = enabled;
    
    if (enabled_) {
        // Open file in append mode
        file_.open(filepath_, std::ios::app);
        if (!file_.is_open()) {
            RCLCPP_ERROR(logger_, "Failed to open data file: %s", filepath_.c_str());
            enabled_ = false;
            return;
        }
        
        // Write header if this is a new file
        if (!header_written_) {
            writeHeader();
        }
        
        RCLCPP_INFO(logger_, "Data collection ENABLED - writing to: %s", filepath_.c_str());
    } else {
        // Close file
        if (file_.is_open()) {
            file_.close();
        }
        RCLCPP_INFO(logger_, "Data collection DISABLED");
    }
}

bool DataCollector::isEnabled() const
{
    return enabled_;
}

void DataCollector::writeDataPoint(const DataPoint& data)
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!enabled_ || !file_.is_open()) {
        return;
    }
    
    file_ << std::fixed << std::setprecision(6)
          << data.timestamp << ","
          << data.position_x << "," << data.position_y << "," << data.position_z << ","
          << data.orientation_roll << "," << data.orientation_pitch << "," << data.orientation_yaw << ","
          << data.control_linear_x << "," << data.control_linear_y << "," << data.control_linear_z << ","
          << data.control_angular_x << "," << data.control_angular_y << "," << data.control_angular_z << ","
          << data.dt << ","
          << data.next_position_x << "," << data.next_position_y << "," << data.next_position_z << ","
          << data.next_orientation_roll << "," << data.next_orientation_pitch << "," << data.next_orientation_yaw << ","
          << data.next_timestamp << "\n";
    file_.flush();
}

void DataCollector::collectTransition(
    const geometry_msgs::msg::PoseStamped& previous_state,
    const geometry_msgs::msg::PoseStamped& current_state,
    double control_linear_x,
    double control_angular_z)
{
    if (!enabled_) {
        return;
    }
    
    DataPoint data_point;
    
    // Previous state
    data_point.timestamp = rclcpp::Time(previous_state.header.stamp).seconds();
    data_point.position_x = previous_state.pose.position.x;
    data_point.position_y = previous_state.pose.position.y;
    data_point.position_z = previous_state.pose.position.z;
    
    // Convert previous quaternion to Euler
    tf2::Quaternion prev_quat(
        previous_state.pose.orientation.x,
        previous_state.pose.orientation.y,
        previous_state.pose.orientation.z,
        previous_state.pose.orientation.w
    );
    double prev_roll, prev_pitch, prev_yaw;
    tf2::Matrix3x3(prev_quat).getRPY(prev_roll, prev_pitch, prev_yaw);
    data_point.orientation_roll = prev_roll;
    data_point.orientation_pitch = prev_pitch;
    data_point.orientation_yaw = prev_yaw;
    
    // Control inputs
    data_point.control_linear_x = control_linear_x;
    data_point.control_linear_y = 0.0;
    data_point.control_linear_z = 0.0;
    data_point.control_angular_x = 0.0;
    data_point.control_angular_y = 0.0;
    data_point.control_angular_z = control_angular_z;
    
    // Time difference
    data_point.dt = rclcpp::Time(current_state.header.stamp).seconds() - 
                   rclcpp::Time(previous_state.header.stamp).seconds();
    
    // Current state (becomes next state)
    data_point.next_timestamp = rclcpp::Time(current_state.header.stamp).seconds();
    data_point.next_position_x = current_state.pose.position.x;
    data_point.next_position_y = current_state.pose.position.y;
    data_point.next_position_z = current_state.pose.position.z;
    
    // Convert current quaternion to Euler
    tf2::Quaternion curr_quat(
        current_state.pose.orientation.x,
        current_state.pose.orientation.y,
        current_state.pose.orientation.z,
        current_state.pose.orientation.w
    );
    double curr_roll, curr_pitch, curr_yaw;
    tf2::Matrix3x3(curr_quat).getRPY(curr_roll, curr_pitch, curr_yaw);
    data_point.next_orientation_roll = curr_roll;
    data_point.next_orientation_pitch = curr_pitch;
    data_point.next_orientation_yaw = curr_yaw;
    
    // Write data point
    try {
        writeDataPoint(data_point);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(logger_, "Failed to write data: %s", e.what());
    }
}

std::string DataCollector::getFilePath() const
{
    return filepath_;
}

void DataCollector::flush()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.flush();
    }
}

} // namespace nsga2_mesh_planner