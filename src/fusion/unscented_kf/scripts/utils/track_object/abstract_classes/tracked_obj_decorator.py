"""
    @author: Numan Senel
    @email: Numan.Senel@thi.de
"""

from utils.track_object.abstract_classes.abstract_tracked_object import AbstractTrack
#Typing imports
from utils.kalman_filter.abstract_classes.abstract_kf_bridge import AbstractKalmanFilterBridge

class TrackDecorator(AbstractTrack):
    def __init__(self, track_object: AbstractTrack):
        super().__init__(track_object.track_id, track_object.kalman_filter_bridge)
        self._track = track_object
    @property
    def track_id(self) -> int: return self._track.track_id
    @property
    def kalman_filter_bridge(self) -> AbstractKalmanFilterBridge: return self._track.kalman_filter_bridge
    @property
    def KF(self): return self._track.KF
    @property
    def skipped_frames(self): return self._track.skipped_frames
    @property
    def hits(self): return self._track.hits
    @property
    def last_measurment_time(self): return self._track.last_measurment_time
    @property
    def track_object_type(self): return self._track.track_object_type
    @property
    def prediction(self): return self._track.prediction
    @property
    def detected_by_sensors(self): return self._track.detected_by_sensors
    @property
    def remove_me(self): return self._track.remove_me
    @property
    def valid_object(self): return self._track.valid_object
    @property
    def additional_time_before_remove(self): return self._track.additional_time_before_remove
    def set_skipped_frames(self, skipped_frames: int) ->None: self._track.set_skipped_frames(skipped_frames)
    def set_last_meaurment_time(self,measurment_time): self._track.set_last_meaurment_time(measurment_time)
    def set_object_type(self, object_type: str): self._track.set_object_type(object_type)
    def set_number_of_hits(self,hits: int) : self._track.set_number_of_hits(hits)
    def set_valid_obj_state(self,is_valid: bool) : self._track.set_valid_obj_state(is_valid)
    def set_remove_state(self,self_remove: bool) : self._track.set_remove_state(self_remove)
    def set_additional_time_before_remove(self, aditional_time_valid_until:int, aditional_time: int) : self._track.set_additional_time_before_remove(aditional_time_valid_until, aditional_time)
    def predict_next_state(self,measurment_time, sensor_type=None): self._track.predict_next_state(measurment_time, sensor_type)
    def correct_prediction(self,measurment): self._track.correct_prediction(measurment)
