# standard ros header with timestamp
# Two-integer timestamp that is expressed as:
# * header.stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * header.stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# other parameters - like seq and frame_id of header are not used
std_msgs/Header header

# unique id to each sensor frame, this is resetted after a certain value
int64 frame_id

# description as follows
# Example 1 "RSU_4__LIDAR_RIGHT"
# Example 2 "RSU_4__LIDAR_LEFT"
string rsu_sensor_id

# count of total points in the frame
int64 total_points

# additional information (optional)
string info

# Point Cloud generated for each frame.
LidarPoint[] lidar_point_cloud
