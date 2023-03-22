from utils.sensor.abstract_classes.abstract_sensor import AbstractSensor

## Typing imports
from utils.measurment.concrete_classes.incoming_measurments import TrackedObject
import numpy as np
import enum

class AbstractMeasurment(AbstractSensor):
    """ Defines required methods properties of valid sensor measurment
        For each detection this class will be generated and removed after kf update.
        It is like a bridge class to use sensor class implementation with new detection values.
    """
    __slots__  = ("_sensor_obj","_measurment")
    def __init__(self, sensor_obj: AbstractSensor, measurment) -> None: ### measurment type is one of the measurment object in incoming_measurments.py
        self._sensor_obj = sensor_obj
        self._measurment = measurment

    @property
    def sensor_obj(self) -> AbstractSensor:
        return self._sensor_obj

    @property
    def measurment(self): ### measurment type is one of the measurment object in incoming_measurments.py
        return self._measurment

    @property
    def obj_name(self) -> str:
        return self.measurment.obj_name

    @property
    def measurment_time(self) -> int:
        return self.measurment.measurment_time

    @property
    def sensor_type(self) -> enum: return self.sensor_obj.sensor_type

    @property
    def sensor_id(self) -> int: return self.sensor_obj.sensor_id

    @property
    def z_dim(self) -> int: return self.sensor_obj.z_dim

    @property
    def R(self) -> np.ndarray: return self.sensor_obj.R

    @property
    def measurment_matrix(self) -> np.ndarray: 
        return self.sensor_obj.measurment_matrix(self.measurment)

    @property
    def measurment_matrix_in_track_cor_system(self) -> np.ndarray: 
        return self.sensor_obj.measurment_matrix_in_track_cor_system(self.measurment)

    @property
    def measurment_matrix_in_cost_function_cor_system(self) -> np.ndarray: 
        return self.sensor_obj.measurment_matrix_in_cost_function_cor_system(self.measurment)
    
    @property
    def sensor_field_of_view(self) -> np.ndarray: 
        return self.sensor_obj.sensor_field_of_view

    def convert_track_cor_to_sensor_cor_system(self, track_obj_position: TrackedObject, object_height: int=0):
        return self.sensor_obj.convert_track_cor_to_sensor_cor_system(track_obj_position, object_height)
    
    def convert_tracks_to_sensor_cor_system(self, predicted_sigma_points: np.ndarray, object_height: int=0) -> np.ndarray:
        return self.sensor_obj.convert_tracks_to_sensor_cor_system(predicted_sigma_points, object_height)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self,key)}' for key in self.__slots__])})"

