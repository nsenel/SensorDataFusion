from utils.sensor.abstract_classes.abstract_sensor_properties import AbstractSensorProperties
#Typing exports
import numpy as np
import enum

class RadarProperties(AbstractSensorProperties):
    def __init__(self, sensor_config: dict) -> None:
        super().__init__(sensor_config)

    @property
    def sensor_type(self) -> enum:
        return self.get_sensor_property("sensor_type")

    @property
    def sensor_id(self) -> int:
        return self.get_sensor_property("sensor_id")
    
    @property
    def measurment_dimention(self) -> int:
        return self.get_sensor_property("measurment_dimention")
    
    @property
    def sensor_noise_matric(self) -> np.ndarray:
        return self.get_sensor_property("sensor_noise_matric")
    
    @property
    def RT_mtx_sensor_to_track(self) -> np.ndarray:
        return self.get_sensor_property("RT_mtx_sensor_to_track")
    
    @property
    def RT_mtx_track_to_sensor(self) -> np.ndarray:
        return self.get_sensor_property("RT_mtx_track_to_sensor")
    
    @property
    def sensor_field_of_view(self)->tuple: #((x_min,x_max), (y_min,y_max))
        return self.get_sensor_property("sensor_field_of_view")