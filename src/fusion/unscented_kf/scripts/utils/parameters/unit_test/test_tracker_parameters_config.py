import numpy as np
from utils.kalman_filter.concrete_classes.unscented_kf import UnscentedKalmanFilter
from utils.kalman_filter.concrete_classes.unscented_kf_bridge import UnscentedKalmanFilterBridge
from utils.parameters.concrete_classes.tracker_parameters import TrackerParameters
from utils.parameters.concrete_classes.unscented_kf_parameters import UKFFilterParameters
from utils.track_object.concrete_classes.basic_track import BasicTrack
from utils.track_object.concrete_classes.track_decorators import AddOnGroundTruth, AddOnTrace

state_dimention = 5
cnt_process_noise_params = 2
augmented_state_dimension = state_dimention + cnt_process_noise_params
std_acceleration = 12
std_yawdd = 0.5

test_ukf_parameters ={
    "state_dimention":state_dimention, ##State dimensions [x(Sensor unit(meter)),y(Sensor unit(meter)),v(Sensor unit(meter)/second),yaw(Target car orientation rad),yawrate(rad/s)]
    "convariance_matrix": np.diag([1,1,1,1,1]),
    "process_noise_parameters":{
        "std_acceleration":std_acceleration, # Standard deviation longitudinal acceleration in m/s^2
        "std_yaw_rate_acceleration":std_yawdd, # Standard deviation yaw acceleration in rad/s^2
        "R":np.array([[std_acceleration**2,0],[0,std_yawdd**2]])
    },
    "augmented_state_dimension": augmented_state_dimension, # 5 from state dimention and 2 from precess noise
    "lambda": 3 - state_dimention, # Sigma point spreading parameter
    "sigma_points_dimension": 2 * augmented_state_dimension + 1 # +1 for mean and split sigma point differante direction
}

filter_settings_person = UKFFilterParameters(ukf_parameters=test_ukf_parameters)
filter_settings_car = UKFFilterParameters(ukf_parameters=test_ukf_parameters)

tracker_parameters_person = {"kalman_filter":UnscentedKalmanFilter,
                             "kf_bridge":UnscentedKalmanFilterBridge,
                             "kf_settings":filter_settings_person,
                             "tracking_strategy":BasicTrack,
                             "add_on_classes_dict":{AddOnGroundTruth:{},
                                                    AddOnTrace:{"max_trace_length":15}}
                            }
tracker_parameters_car = {"kalman_filter":UnscentedKalmanFilter,
                          "kf_bridge":UnscentedKalmanFilterBridge,
                          "kf_settings":filter_settings_car,
                          "tracking_strategy":BasicTrack,
                          "add_on_classes_dict":{AddOnGroundTruth:{},
                                                 AddOnTrace:{"max_trace_length":15}}
                         }
object_based_tracker_parameters = {"person":TrackerParameters(tracker_parameters_person),
                                   "car":TrackerParameters(tracker_parameters_car)}


tracker_parameters_person = {"kalman_filter":UnscentedKalmanFilter,
                             "kf_bridge":UnscentedKalmanFilterBridge,
                             "kf_settings":filter_settings_person,
                             "tracking_strategy":BasicTrack,
                             "add_on_classes_dict":{AddOnTrace:{"max_trace_length":15}}
                            }
tracker_parameters_car = {"kalman_filter":UnscentedKalmanFilter,
                          "kf_bridge":UnscentedKalmanFilterBridge,
                          "kf_settings":filter_settings_car,
                          "tracking_strategy":BasicTrack,
                          "add_on_classes_dict":{AddOnTrace:{"max_trace_length":15}}
                         }
object_based_tracker_parameters_without_GT = {"person":TrackerParameters(tracker_parameters_person),
                                   "car":TrackerParameters(tracker_parameters_car)}