from abc import ABC, abstractproperty

#Typing exports
import numpy as np

class AbstractFilterParameters(ABC):
    
    @abstractproperty
    def state_dim(self) -> int:
        pass

    @abstractproperty
    def convariance_matrix(self) -> np.ndarray:
        pass
    
    @abstractproperty
    def process_noise_parameters(self) -> np.ndarray:
        pass