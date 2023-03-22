
#Typing imports
from utils.track_object.abstract_classes.abstract_tracked_object import AbstractTrack
from utils.kalman_filter.abstract_classes.abstract_kf import AbstractKalmanFilterImplementer
from utils.kalman_filter.abstract_classes.abstract_kf_bridge import AbstractKalmanFilterBridge
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.object_factories.abstract_classes.abstract_kf_factory import AbstractKalmanFilterObjectFactory
from utils.parameters.concrete_classes.unscented_kf_parameters import UKFFilterParameters
import numpy as np


class KalmanFilterObjectFactory(AbstractKalmanFilterObjectFactory):
    def __init__(self, kalman_filter: AbstractKalmanFilterImplementer,
                       kalman_filter_bridge: AbstractKalmanFilterBridge,
                       filter_settings: UKFFilterParameters) -> None:
        super().__init__(kalman_filter, kalman_filter_bridge, filter_settings)
        ##Filter setting etc. object type based as default since this is created for number of object type defined in tracker_parameters_config
    
    def reset_kf_obj_bridge(self, tracked_obj: AbstractTrack) -> AbstractKalmanFilterBridge:
        kf_obj: AbstractKalmanFilterImplementer = self.kf(self.filter_settings)
        kf_obj.extensive_init(initial_state_vector=tracked_obj.KF._x, intial_p=tracked_obj.KF._P, initial_measurment_time=tracked_obj.KF.fist_detection_time,
                              initial_measutment_pos=tracked_obj.KF.initial_object_location, register_time=tracked_obj.KF._last_update_time)
        return self.kf_bridge(kf_obj)
    
    def generate_kf_obj_bridge(self, initial_measurment: AbstractMeasurment, filter_settings: UKFFilterParameters=None) -> AbstractKalmanFilterBridge:
        filter_settings = filter_settings if filter_settings else self.filter_settings
        kf_obj: AbstractKalmanFilterImplementer = self.kf(filter_settings)
        kf_obj.init_measurment(initial_measurment)
        return self.kf_bridge(kf_obj)