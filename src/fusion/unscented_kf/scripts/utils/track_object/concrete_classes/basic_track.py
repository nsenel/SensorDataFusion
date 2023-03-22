from utils.track_object.abstract_classes.abstract_tracked_object import AbstractTrack
#Typing imports
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.kalman_filter.abstract_classes.abstract_kf import AbstractKalmanFilterImplementer
from utils.kalman_filter.abstract_classes.abstract_kf_bridge import AbstractKalmanFilterBridge
from utils.common.enums import SensorTypes
import numpy as np

class BasicTrack(AbstractTrack):
    def __init__(self, track_id: int, kalman_filter_bridge: AbstractKalmanFilterBridge, track_object_type: str, initial_measurment: AbstractMeasurment= None):
        super().__init__(track_id, kalman_filter_bridge, initial_measurment)
        self._tracked_obj_name = track_object_type
    @property
    def track_id(self) ->int: return super().track_id
    @property
    def kalman_filter_bridge(self) ->AbstractKalmanFilterBridge: return super().kalman_filter_bridge
    @property
    def KF(self) ->AbstractKalmanFilterImplementer: return super().KF
    @property
    def skipped_frames(self) ->int: return super().skipped_frames
    @property
    def hits(self) ->int: return super().hits
    @property
    def last_measurment_time(self) ->int: return super().last_measurment_time
    @property
    def track_object_type(self) ->str: return super().track_object_type
    @property
    def prediction(self) ->np.ndarray: return super().prediction
    @property
    def detected_by_sensors(self) ->dict: return super().detected_by_sensors
    @property
    def remove_me(self): return super().remove_me
    @property
    def valid_object(self): return super().valid_object
    @property
    def additional_time_before_remove(self): return super().additional_time_before_remove
    def set_skipped_frames(self, skipped_frames: int) ->None: super().set_skipped_frames(skipped_frames)
    def set_last_meaurment_time(self,measurment_time: int) ->None: super().set_last_meaurment_time(measurment_time)
    def set_object_type(self, object_type: str) ->None: super().set_object_type(object_type)
    def set_number_of_hits(self,hits: int) ->None: super().set_number_of_hits(hits)
    def set_valid_obj_state(self,is_valid: bool) ->None : super().set_valid_obj_state(is_valid)
    def set_remove_state(self,self_remove: bool) ->None : super().set_remove_state(self_remove)
    def set_additional_time_before_remove(self, aditional_time_valid_until:int, aditional_time: int) ->None : super().set_additional_time_before_remove(aditional_time_valid_until, aditional_time)
    def predict_next_state(self,measurment_time: int, sensor_type:SensorTypes=None) ->None: super().predict_next_state(measurment_time,sensor_type)
    def correct_prediction(self,measurment: AbstractMeasurment) ->None: super().correct_prediction(measurment)