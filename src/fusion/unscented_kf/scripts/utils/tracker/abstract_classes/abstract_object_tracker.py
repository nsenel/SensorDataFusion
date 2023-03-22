from abc import ABC, abstractmethod


#Typing imports
from utils.object_factories.abstract_classes.abstract_track_factory import AbstractTrackFactory
from utils.sensor.abstract_classes.abstract_sensor_properties import AbstractSensorProperties
from utils.tracker.abstract_classes.abstrack_cost_calculator import AbstractCostCalculator
from utils.common.enums import SensorTypes
from typing import List
import numpy as np

class AbstractTracker(ABC):
    """ Defines required methods properties of valid tracker class """
    def __init__(self, track_generator: AbstractTrackFactory, cost_calculator: AbstractCostCalculator):
        self._track_generator = track_generator
        self._cost_calculator = cost_calculator
        self._tracked_objs = []
    
    @property
    def track_generator(self) ->AbstractTrackFactory:
        """ Getter for track generator
        Args:
            None
        Return:
            AbstractTrackFactory 
        """
        return self._track_generator
    
    @property
    def cost_calculator(self) ->AbstractCostCalculator: 
        """ Getter for cost_calculator
        Args:
            None
        Return:
            AbstractCostCalculator 
        """
        return self._cost_calculator

    @property
    def tracked_object_list(self) -> List:
        """ Getter for trackt objects
        Args:
            None
        Return:
            tracked_objs -> list[AbstrackTrack]
        """
        return self._tracked_objs
    
    def remove_track_obj_by_index(self, index:int, target_list: list=None) ->None:
        try:
            if target_list != None:
                if target_list is self._tracked_objs:
                    print(self._tracked_objs[index].track_id, "is going to be deleted")
                del target_list[index]
            else:
                print(self._tracked_objs[index].track_id, "is going to be deleted")
                del self._tracked_objs[index]
        except Exception as ex:
                print("remove_track_obj_by_index() \n", ex)
                raise ex

    @abstractmethod
    def add_new_tracks(self, un_assigned_detects_measurments: list, measurment_sensor_properties:AbstractSensorProperties) ->None:
        pass

    @abstractmethod
    def calculate_cost(self, detections: list, measurment_time, *args) ->np.ndarray:
        pass
    
    @abstractmethod
    def assign_detections_to_trackers(self,cost_matrix: np.ndarray, *args):
        pass
    
    @abstractmethod
    def update_tracks_filter(self, assigned_tracks: list, un_assigned_tracks: list, detections, *args) ->None:
        pass

    @abstractmethod
    def predict_new_state(self, measurment_time: int, measurment_sensor_properties:AbstractSensorProperties) ->None:
        pass

    @abstractmethod
    def update_tracks(self, detections: list, measurment_time: int, measurment_sensor_properties:AbstractSensorProperties) ->None:
        pass


