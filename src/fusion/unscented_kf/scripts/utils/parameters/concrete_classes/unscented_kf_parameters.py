from utils.parameters.abstract_classes.abstract_filter_parameters import AbstractFilterParameters

#Typing exports
import numpy as np

class UKFFilterParameters(AbstractFilterParameters):
    def __init__(self, ukf_parameters: dict) -> None:
        self._state_dim=ukf_parameters["state_dimention"]
        self._P = ukf_parameters["convariance_matrix"]
        self._R = ukf_parameters["process_noise_parameters"]["R"]
        self._n_aug = ukf_parameters["augmented_state_dimension"]
        self._lambda=ukf_parameters["lambda"]
        self._n_sig = ukf_parameters["sigma_points_dimension"]

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def convariance_matrix(self) -> np.ndarray:
        return self._P.copy()
    
    @property
    def process_noise_parameters(self) -> np.ndarray:
        return self._R.copy()
    
    @property
    def augmented_state_dimension(self) -> int:
        return self._n_aug
    
    @property
    def sigma_point_spreading_ratio(self) -> int:
        return self._lambda
    
    @property
    def sigma_points_dimension(self) -> int:
        return self._n_sig