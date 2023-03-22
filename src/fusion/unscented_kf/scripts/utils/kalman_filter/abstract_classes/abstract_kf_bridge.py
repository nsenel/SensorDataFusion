from abc import ABC, abstractmethod, abstractproperty

#Typing exports
from utils.kalman_filter.abstract_classes.abstract_kf import AbstractKalmanFilterImplementer
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment

class AbstractKalmanFilterBridge(ABC):
    """Bridge class: Updating, prediction new object state with differant kalman filter approachs
    i.e. linear kalman, ukf, extended kalman, ..
    """
    def __init__(self, kalman_filter_obj: AbstractKalmanFilterImplementer) -> None:
        self._KF = kalman_filter_obj
    
    @property
    def KF(self) -> AbstractKalmanFilterImplementer:
        return self._KF
    
    @abstractmethod
    def predict_with_KF(self, measurment_time: int) -> None:
        """ Predict new object state from previous state with using time differance """
        pass
    @abstractmethod
    def update_KF(self, measurment: AbstractMeasurment) -> None:
        """ Update predicted object state with using new measurment """
        pass