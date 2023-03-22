from abc import ABC, abstractmethod, abstractproperty
import numpy as np

## Typing imports
from utils.measurment.concrete_classes.incoming_measurments import EgoCarOdomMessage

class AbstractMovmentCompensation(ABC):

    @abstractmethod
    def get_movement_compensation(self, track_x, track_y, measurment_time, last_update_time) -> tuple:
        """ Returns x, y change due to car movment and oriantation"""
        pass

    @abstractmethod
    def update_state(self, incoming_ego_data:EgoCarOdomMessage) ->None:
        """ Update car state with latest ego data"""
        pass

    def trasform_point_with_car_orientation(self, track_x: float, track_y: float,
                                                  moved_in_x: float, moved_in_y: float,
                                                  total_yaw_change: float):
        transformation = np.array([[np.cos(total_yaw_change), -np.sin(total_yaw_change), moved_in_x],
                                   [np.sin(total_yaw_change), np.cos(total_yaw_change), moved_in_y]])
        return np.matmul(transformation,np.array([track_x,track_y,1]))