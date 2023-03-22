import numpy as np
from utils.common.enums import ObjectTypes, SensorTypes
from utils.kalman_filter.concrete_classes.unscented_kf import UnscentedKalmanFilter
from utils.kalman_filter.concrete_classes.unscented_kf_bridge import UnscentedKalmanFilterBridge
from utils.movement_compensation.concrete_classes.movement_compensation_with_global_change import MVWithGlobalPostionChange
from utils.movement_compensation.concrete_classes.movement_compensation_with_speed import MVWithSpeed
from utils.object_factories.concrete_classes.track_object_factory import TrackKFOObjectTypeBasedFactory, TrackKFObjectFactory
from utils.parameters.concrete_classes.tracker_parameters import TrackerParameters
from utils.parameters.concrete_classes.unscented_kf_parameters import UKFFilterParameters
from utils.track_object.concrete_classes.basic_track import BasicTrack
from utils.track_object.concrete_classes.track_decorators import AddOnCalibration, AddOnGroundTruth, AddOnMovementCompensation, AddOnSensorDetectInfo, AddOnTrace, AddOnTrackVisualization, AddOnMap
from utils.tracker.concrete_classes.basic_tracker import BasicTracker
from utils.tracker.concrete_classes.tracker_prevent_gost_detection import DoubleTrackTracker
from utils.tracker.concrete_classes.tracker_with_map import DoubleTrackTrackerMap
from utils.tracker.concrete_classes.cost_calculator import NormCostCalculator, NormCostCalculatorSeperateByObjectExeptUnknown, NormCostCalculatorSeperateByObjectType, NormCostCalculatorSeperateByObjectExeptUnknownMAP
from utils.map_generator.basic_map import BasicMap
from utils.track_object.concrete_classes.auto_calibration import Calibrator
object_map = BasicMap(map_l_min=0, map_l_max=20, map_h_min=-134, map_h_max=134, cell_l=3, cell_h=8)
target_sensor_to_calibrate = Calibrator(target_sensor_id=65)
state_dimention = 5
cnt_process_noise_params = 2
augmented_state_dimension = state_dimention + cnt_process_noise_params
std_acceleration = 7
std_yawdd = 0.05

ukf_parameters ={
    "state_dimention":state_dimention, ##State dimensions [x(Sensor unit(meter)),y(Sensor unit(meter)),v(Sensor unit(meter)/second),yaw(Target car orientation rad),yawrate(rad/s)]
    "convariance_matrix": np.diag([10,10,100,10,10]), ### 3th element convariance for speed it is very big since car in long distance doesnt have good accuracity of location by camera be carefull if you change it!!
    "process_noise_parameters":{
        "std_acceleration":std_acceleration, # Standard deviation longitudinal acceleration in m/s^2
        "std_yaw_rate_acceleration":std_yawdd, # Standard deviation yaw acceleration in rad/s^2
        "R":np.array([[std_acceleration**2,0],[0,std_yawdd**2]])
    },
    "augmented_state_dimension": augmented_state_dimension, # 5 from state dimention and 2 from precess noise
    "lambda": 3 - state_dimention, # Sigma point spreading parameter
    "sigma_points_dimension": 2 * augmented_state_dimension + 1 # +1 for mean and split sigma point differante direction
}

ukf_parameters_person ={
    "state_dimention":state_dimention, ##State dimensions [x(Sensor unit(meter)),y(Sensor unit(meter)),v(Sensor unit(meter)/second),yaw(Target car orientation rad),yawrate(rad/s)]
    "convariance_matrix": np.diag([20,20,20,10,10]),
    "process_noise_parameters":{
        "std_acceleration":std_acceleration/10, # Standard deviation longitudinal acceleration in m/s^2
        "std_yaw_rate_acceleration":std_yawdd*3, # Standard deviation yaw acceleration in rad/s^2
        "R":np.array([[std_acceleration**2,0],[0,std_yawdd**2]])
    },
    "augmented_state_dimension": augmented_state_dimension, # 5 from state dimention and 2 from precess noise
    "lambda": 3 - state_dimention, # Sigma point spreading parameter
    "sigma_points_dimension": 2 * augmented_state_dimension + 1 # +1 for mean and split sigma point differante direction
}
movement_compensetor = MVWithSpeed() ### single object shared between all track objects and tracker updates it state!!
# movement_compensetor = MVWithGlobalPostionChange() ### single object shared between all track objects and tracker updates it state!!
filter_settings_person = UKFFilterParameters(ukf_parameters=ukf_parameters_person)
filter_settings_car = UKFFilterParameters(ukf_parameters=ukf_parameters)

tracker_parameters_person = {"kalman_filter":UnscentedKalmanFilter,
                             "kf_bridge":UnscentedKalmanFilterBridge,
                             "kf_settings":filter_settings_person,
                             "tracking_strategy":BasicTrack,
                             "add_on_classes_dict":{
                                                    AddOnGroundTruth:{},
                                                    AddOnTrace:{"max_trace_length":5},
                                                    AddOnTrackVisualization:{},
                                                    #AddOnCalibration:{"calibrator":target_sensor_to_calibrate, "target_sensor": SensorTypes.radar_cartesian, "sensor_in_track_cor": SensorTypes.camera},
                                                    AddOnMovementCompensation:{"movement_compensation_method":movement_compensetor},
                                                    AddOnSensorDetectInfo:{},
                                                    #AddOnMap:{"object_map":object_map}
                                                    }
                            }
tracker_parameters_car = {"kalman_filter":UnscentedKalmanFilter,
                          "kf_bridge":UnscentedKalmanFilterBridge,
                          "kf_settings":filter_settings_car,
                          "tracking_strategy":BasicTrack,
                          "add_on_classes_dict":{
                                                AddOnGroundTruth:{},
                                                AddOnTrace:{"max_trace_length":5},
                                                #AddOnCalibration:{"calibrator":target_sensor_to_calibrate, "target_sensor": SensorTypes.radar_cartesian, "sensor_in_track_cor": SensorTypes.camera},
                                                AddOnTrackVisualization:{"live_plot":False, "plot_data_on_delete":False},  #### DONT USE BOTH WITH ROS it is for offline reading
                                                AddOnMovementCompensation:{"movement_compensation_method":movement_compensetor},
                                                AddOnSensorDetectInfo:{},
                                                #AddOnMap:{"object_map":object_map}
                                                }
                         }

object_based_tracker_parameters = {ObjectTypes.person:TrackerParameters(tracker_parameters_person),
                                   ObjectTypes.car:TrackerParameters(tracker_parameters_car)}

def generate_traker():
    track_object_factory = TrackKFOObjectTypeBasedFactory(object_type_based_trackKF_settings = object_based_tracker_parameters,
                                                            generate_id_from_dict=None)
    cost_calculator = NormCostCalculatorSeperateByObjectExeptUnknown()
    tracker_obj =  DoubleTrackTracker(track_generator=track_object_factory,
                                      cost_calculator=cost_calculator,dist_thresh=5,
                                      time_treshhold_to_remove=1.5, ## seconds
                                      waiting_list_time_treshhold_to_remove=0.5, #seconds
                                      waiting_list_max_hit=5)
    return tracker_obj



#remove visualization and grount truth since we dont have it
tracker_parameters_person["add_on_classes_dict"] = tracker_parameters_person["add_on_classes_dict"].copy()
tracker_parameters_car["add_on_classes_dict"] = tracker_parameters_car["add_on_classes_dict"].copy()
del tracker_parameters_person["add_on_classes_dict"][AddOnGroundTruth]
del tracker_parameters_person["add_on_classes_dict"][AddOnTrackVisualization]
del tracker_parameters_car["add_on_classes_dict"][AddOnGroundTruth]
del tracker_parameters_car["add_on_classes_dict"][AddOnTrackVisualization]
object_based_tracker_parameters2 = {ObjectTypes.person:TrackerParameters(tracker_parameters_person),
                                    ObjectTypes.car:TrackerParameters(tracker_parameters_car)}
def generate_traker_twizy():
    track_object_factory = TrackKFObjectFactory(track_kf_object_factory_settings_default = TrackerParameters(tracker_parameters_car),
                                                generate_id_from_dict=None, object_type_based_trackKF_settings=object_based_tracker_parameters2)
    cost_calculator = NormCostCalculatorSeperateByObjectExeptUnknown() ##### Lidar doesnt provide object class therefore you can't use NormCostCalculatorSeperateByObjectType and TrackKFOObjectTypeBasedFactory !!!
    tracker_obj =  DoubleTrackTracker(track_generator=track_object_factory,
                                      cost_calculator=cost_calculator,dist_thresh=3,
                                      max_frames_to_skip=30,
                                      waiting_list_max_skip=5,
                                      waiting_list_max_hit=10)
    return tracker_obj

# generate_id_from_dict={"car":{"min":100, "max":999}, "person":{"min":1000, "max":1999}} car ids will start from 100 up to 999 and person will start 1000 up to what ever said in max number
# generate_id_from_dict=False will generate increasing ids from 1 to 255 and than returns back to 1.
# generate_id_from_dict=None will make object based id assigment. Ex: car will start from 1 also person will start from one. so there might be same id for both object in sceen in same time.
def generate_traker_in2lab():
    track_object_factory = TrackKFObjectFactory(track_kf_object_factory_settings_default = TrackerParameters(tracker_parameters_car),
                                                generate_id_from_dict=False, object_type_based_trackKF_settings=object_based_tracker_parameters2)
    
    cost_calculator = NormCostCalculatorSeperateByObjectExeptUnknownMAP(object_map) ##### Lidar doesnt provide object class therefore you can't use NormCostCalculatorSeperateByObjectType and TrackKFOObjectTypeBasedFactory !!!
    # cost_calculator = NormCostCalculatorSeperateByObjectExeptUnknown()
    tracker_obj =  DoubleTrackTrackerMap(track_generator=track_object_factory,
                                         cost_calculator=cost_calculator,dist_thresh=5,
                                         time_treshhold_to_remove=1.5, ## seconds
                                         waiting_list_time_treshhold_to_remove=0.5, #seconds
                                         waiting_list_max_hit=5,
                                         object_map=object_map)
    # tracker_obj =  DoubleTrackTracker(track_generator=track_object_factory,
    #                                   cost_calculator=cost_calculator,dist_thresh=5,
    #                                   time_treshhold_to_remove=1.5, ## seconds
    #                                   waiting_list_time_treshhold_to_remove=0.5, #seconds
    #                                   waiting_list_max_hit=5,
    #                                   object_map=object_map)
    return tracker_obj
