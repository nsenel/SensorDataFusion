# Multi-Sensor Data Fusion for Real-Time Multi-Object Tracking

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Welcome to our multi-sensor fusion framework for environmental perception in smart traffic applications! Our framework fuses data on the object list level from distributed automotive sensors, including cameras, radar, and LiDAR. By combining data from multiple sensors, the accuracy and reliability of environmental perception are increased.

Our modular, real-time capable architecture is adaptable and does not rely on the number or types of sensors. The framework consists of several modules, including a coordinate transformation module, an object association module (Hungarian algorithm), an object tracking module (unscented Kalman filter), and a movement compensation module.


## Usage
To use our multi-sensor fusion framework, follow these steps:

Install the required dependencies ROS, Python(Matplotlib, numpy, opencv).
Running with lgsvl simulator:
Run the fusion framework main ros launch: roslaunch unscented_kf_v4 lgsvl_bag_tracker.launch
  - Change the bag file location or disable the command simulation running in parallel
Runing with custom sensor set:
  - Change subcribe topic names in ukf_ros_node_lgsvl.py as custom topic names.
  - Adjust the detection callback functions to parse custom detection message.
  - Change sensor_parameters_config.py regarding custom sensor calibration.
Customize the fusion framework parameters as needed in tracker_parameters_config.py.

Note: This fusion framework has been used in multiple projects and hasn't been cleaned up completely yet, therefore you may encounter unused ROS messages and/or project-specific comments in the code.

## TODO
- [ ] Code clean up
- [ ] Update ukf_class_diagram

## Citation
Full paper: https://www.mdpi.com/2227-9717/11/2/501

MDPI and ACS Style
Senel, N.; Kefferpütz, K.; Doycheva, K.; Elger, G. Multi-Sensor Data Fusion for Real-Time Multi-Object Tracking. Processes 2023, 11, 501.

## License
This code is released under the MIT License. Feel free to use, modify, and distribute this code, as long as you include the original license and attribution.
