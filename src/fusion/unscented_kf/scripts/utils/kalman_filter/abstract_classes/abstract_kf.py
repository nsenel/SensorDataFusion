from abc import ABC, abstractmethod, abstractproperty


#Typing exports
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
import numpy as np

class AbstractKalmanFilterImplementer(ABC):
    """ Defines required methods properties of valid Kalman class """
    @property
    def x_postion(self)-> float:
        pass
    
    @x_postion.setter
    def x_postion(self, values: tuple)-> None: #tuple(new_postion: float, uncertainty: float=0)
        pass

    @property
    def y_postion(self)->float:
        pass

    @y_postion.setter
    def y_postion(self, values: tuple)-> None: #tuple(new_postion: float, uncertainty: float=0)
        pass
    
    @abstractproperty
    def state_dimension(self) -> int:
        pass
    @abstractproperty
    def x(self) -> np.ndarray:
        pass
    @abstractproperty
    def P(self) -> np.ndarray:
        pass
    @abstractproperty
    def last_update_time(self) ->int:
        pass
    ## Abstract class methods
    @abstractmethod
    def init_measurment(self, initial_measurment: AbstractMeasurment) -> None:
        pass
    @abstractmethod
    def extensive_init(self, initial_state_vector: np.ndarray, intial_p:np.ndarray, initial_measurment_time: int, initial_measutment_pos: np.ndarray, register_time: np.ndarray) -> None:
        pass
    @abstractmethod
    def predict(self, measurment_time: int) -> None:
        pass
    @abstractmethod
    def update(self, measurment: AbstractMeasurment) -> None:
        pass