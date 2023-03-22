from abc import ABC, abstractproperty

#Typing exports
import numpy as np
import enum

class AbstractSensorProperties(ABC):
    """ Defines required methods properties of valid sensor properties class """
    def __init__(self, sensor_config: dict) -> None:
        self._sensor_config = sensor_config
    
    @property
    def sensor_config(self):
        return self._sensor_config

    @abstractproperty
    def sensor_type(self) -> enum:
        pass

    @abstractproperty
    def sensor_id(self) -> int:
        pass

    @abstractproperty
    def measurment_dimention(self) -> int:
        pass
    
    @abstractproperty
    def sensor_noise_matric(self) -> np.ndarray:
        pass
    
    @abstractproperty
    def RT_mtx_sensor_to_track(self) -> np.ndarray:
        pass

    @abstractproperty
    def RT_mtx_track_to_sensor(self) -> np.ndarray:
        pass

    @abstractproperty
    def sensor_field_of_view(self)->tuple: #((x_min,x_max), (y_min,y_max))
        pass

    def get_sensor_property(self, property: str) -> any:
        if property in self.sensor_config.keys():
            return self.sensor_config[property]
        return False
    
    def is_object_in_FOV(self, target:float):
        """ Compere detection field of view of the sensor and the track object
            Args:
            Target is object view angle in terms of track cordinate system such;
            is_object_in_FOV(np.degrees(np.arctan2(track.prediction[1],track.prediction[0])))
        """
        start, end = self.sensor_field_of_view[0], self.sensor_field_of_view[1]
        end = end - start + 360.0 if (end - start) < 0.0 else end - start  
        target = target - start + 360.0 if (target - start) < 0.0 else  target - start 
        return (target < end)
    