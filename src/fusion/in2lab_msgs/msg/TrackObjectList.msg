# standard ros header with timestamp
# Two-integer timestamp that is expressed as:
# * header.stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * header.stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# other parameters - like seq and frame_id of header are not used

std_msgs/Header header

# unique id to each sensor frame, this is resetted after a certain value
int64 frame_id

# description as follows
# Example 1 "RSU_1__RADAR_LEFT"
# Example 2 "RSU_4__LIDAR_RIGHT"
# Example 3 "RSU_7__CAMERA_MIDDLE"
string rsu_sensor_id

# count of total objects in the frame
int64 total_objects

# additional information (optional)
string info

# List of objects tracked per frame from one sensor. For Object type, refer Object msg file
TrackObject[] object_list
