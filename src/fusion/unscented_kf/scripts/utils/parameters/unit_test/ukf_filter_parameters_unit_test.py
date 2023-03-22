import unittest
import numpy as np
from utils.parameters.unit_test.test_tracker_parameters_config import test_ukf_parameters
from utils.parameters.abstract_classes.abstract_filter_parameters import AbstractFilterParameters
from utils.parameters.concrete_classes.unscented_kf_parameters import UKFFilterParameters

class TestAbstractFilterParameters(AbstractFilterParameters, unittest.TestCase):
    pass

class TestUKFFilterParameters(unittest.TestCase):
    
    def setUp(self):
        self.filter_parameters = UKFFilterParameters(ukf_parameters=test_ukf_parameters)

    def test_state_dim(self):
        """Reading and setter test for state dimention"""
        self.assertEqual(self.filter_parameters.state_dim, test_ukf_parameters["state_dimention"])

    def test_convariance_matrix(self):
        """Reading and setter test for convariance matrix"""
        self.assertEqual(self.filter_parameters.convariance_matrix.tolist(), test_ukf_parameters["convariance_matrix"].tolist())
    
    def test_process_noise_parameters(self):
        """Reading and setter test for process noise parameters"""
        self.assertEqual(self.filter_parameters.process_noise_parameters.tolist(), test_ukf_parameters["process_noise_parameters"]["R"].tolist())
    
    def test_augmented_state_dimension(self):
        """Reading and setter test for augmented state dimension"""
        self.assertEqual(self.filter_parameters.augmented_state_dimension, test_ukf_parameters["augmented_state_dimension"])
    
    def test_sigma_point_spreading_ratio(self):
        """Reading and setter test for lambda"""
        self.assertEqual(self.filter_parameters.sigma_point_spreading_ratio, test_ukf_parameters["lambda"])
    
    def test_sigma_points_dimension(self):
        """Reading and setter test for sigma points dimension"""
        self.assertEqual(self.filter_parameters.sigma_points_dimension, test_ukf_parameters["sigma_points_dimension"])
