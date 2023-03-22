from abc import ABC, abstractmethod, abstractproperty
#Typing exports
import numpy as np
from utils.sensor.abstract_classes.abstract_sensor_properties import AbstractSensorProperties
from utils.measurment.concrete_classes.incoming_measurments import TrackedObject
import enum

class AbstractSensor(ABC):
    """ Defines required methods properties of valid Sensor class """
    def __init__(self, sensor_properties_obj: AbstractSensorProperties) -> None:
        self._sensor_properties=sensor_properties_obj
    
    @property
    def sensor_properties(self) -> AbstractSensorProperties:
        return self._sensor_properties

    @property
    def sensor_type(self) -> enum:
        return self.sensor_properties.sensor_type
    
    @property
    def sensor_id(self) -> int:
        return self.sensor_properties.sensor_id

    @property
    def z_dim(self) -> int:
        return self.sensor_properties.measurment_dimention

    @property
    def R(self) -> np.ndarray:
        return self.sensor_properties.sensor_noise_matric
    
    @property
    def sensor_field_of_view(self) -> np.ndarray:
        return self.sensor_properties.sensor_field_of_view

    @abstractmethod
    def measurment_matrix(self, incoming_measurment_from_sensor) -> np.ndarray: 
        pass

    @abstractmethod
    def measurment_matrix_in_track_cor_system(self, incoming_measurment_from_sensor) -> np.ndarray: ### incoming_measurment_from_sensor is not AbstractMeasurment it is AbstractMeasurment.measurment so it is raw measurment from some sensor
        pass

    @abstractmethod
    def measurment_matrix_in_cost_function_cor_system(self, incoming_measurment_from_sensor) -> np.ndarray: ### incoming_measurment_from_sensor is not AbstractMeasurment it is AbstractMeasurment.measurment so it is raw measurment from some sensor
        pass

    @abstractmethod
    def convert_track_cor_to_sensor_cor_system(self, track_obj_position:TrackedObject) -> np.ndarray:
        pass

    @abstractmethod
    def convert_tracks_to_sensor_cor_system(self, predicted_sigma_points: np.ndarray) -> np.ndarray:
        pass
    
    def is_object_in_FOV(self, target:float):
        return self.sensor_properties.is_object_in_FOV(target)
    
    def __str__(self):
        if self.sensor_id != None:
            info_dict = {"sensor_id":   self.sensor_id,
                         "sensor_type": self.sensor_type} ## TODO add sensor properties
            return "AbstractSensor()"+",".join("{}={}".format(*i) for i in info_dict.items())
        return None