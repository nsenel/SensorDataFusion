from abc import ABC, abstractmethod

#Typing imports
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.parameters.concrete_classes.unscented_kf_parameters import UKFFilterParameters
from utils.kalman_filter.abstract_classes.abstract_kf import AbstractKalmanFilterImplementer
from utils.kalman_filter.abstract_classes.abstract_kf_bridge import AbstractKalmanFilterBridge
import numpy as np


class AbstractKalmanFilterObjectFactory(ABC):
    """ Generates Kalman filter object """
    def __init__(self, kalman_filter: AbstractKalmanFilterImplementer,
                       kalman_filter_bridge: AbstractKalmanFilterBridge,
                       filter_settings: UKFFilterParameters) -> None:
        self._kf = kalman_filter
        self._kf_bridge = kalman_filter_bridge
        self._filter_settings = filter_settings
    
    @property
    def kf(self):
        return self._kf
    
    @property
    def kf_bridge(self):
        return self._kf_bridge
    
    @property
    def filter_settings(self):
        return self._filter_settings

    @abstractmethod
    def reset_kf_obj_bridge(self, state_vektor: np.ndarray, filter_settings: UKFFilterParameters) -> AbstractKalmanFilterBridge:
        pass

    @abstractmethod
    def generate_kf_obj_bridge(self, initial_measurment: AbstractMeasurment, filter_settings: UKFFilterParameters=None) -> AbstractKalmanFilterBridge:
        pass