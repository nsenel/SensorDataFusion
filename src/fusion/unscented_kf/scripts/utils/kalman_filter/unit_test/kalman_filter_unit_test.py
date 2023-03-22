import unittest
import numpy as np

from utils.parameters.unit_test.test_sensor_parameters_config import test_sensor_properties
from utils.parameters.unit_test.test_tracker_parameters_config import test_ukf_parameters
from utils.parameters.concrete_classes.unscented_kf_parameters import UKFFilterParameters
from utils.kalman_filter.concrete_classes.unscented_kf import UnscentedKalmanFilter
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.measurment.concrete_classes.incoming_measurments import InitialLidarMeasurment
from utils.sensor.concrete_classes.lidar_sensor import LidarSensor
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties



class TestUnscentedKalmanFilter(unittest.TestCase):
    
    def setUp(self):
        filter_parameters = UKFFilterParameters(ukf_parameters=test_ukf_parameters)
        cam_properties   = LidarProperties(test_sensor_properties["Lidar"])
        self.lidar = LidarSensor(cam_properties)
        self.initial_measurment = InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)
        measurment = AbstractMeasurment(sensor_obj=self.lidar, measurment=self.initial_measurment)
        self.ukf = UnscentedKalmanFilter(filter_parameters=filter_parameters)
        self.ukf.init_measurment(initial_measurment=measurment)

    def test_state_dim(self):
        """Reading and setter test for state dimention"""
        self.assertEqual(self.ukf.state_dimension, test_ukf_parameters["state_dimention"])

    def test_x(self):
        """Getter test for state matrix"""
        x = np.zeros(test_ukf_parameters["state_dimention"])
        x[0:2] = [self.initial_measurment.x, self.initial_measurment.y]
        np.testing.assert_almost_equal(self.ukf.x, x)

    def test_P(self):
        """Reading and setter test for convariance matrix"""
        self.assertEqual(self.ukf.P.tolist(), test_ukf_parameters["convariance_matrix"].tolist())
    
    def test_last_update_time(self):
        """Getter setter test for last update time"""
        self.assertEqual(self.ukf.last_update_time, self.initial_measurment.measurment_time)
    
    def test_predict(self):
        """Test ukf predict"""
        x = np.zeros(test_ukf_parameters["state_dimention"])
        x[0:2] = [self.initial_measurment.x, self.initial_measurment.y]
        previous_P = self.ukf.P.diagonal().copy()
        self.assertEqual(self.ukf.predict(2), None)
        np.testing.assert_almost_equal(self.ukf.x.tolist(), x.tolist())
        ## Uncertainty about object location should be increased due to applying predict
        for p_predict,p_before_predict in zip(self.ukf.P.diagonal(), previous_P):
            self.assertGreaterEqual(p_predict,p_before_predict)
        
    def test_update(self):
        """Test ukf predict"""
        second_incoming_measurment = InitialLidarMeasurment(x=8.5, y=0.5, obj_name="car", measurment_time=2)
        second_measurment = AbstractMeasurment(sensor_obj=self.lidar, measurment=second_incoming_measurment)
        self.ukf.predict(2)
        previous_x = self.ukf.x.copy()[0:2]
        previous_P = self.ukf.P.diagonal().copy()
        self.ukf.update(second_measurment)
        for p_predict,p_before_predict in zip(self.ukf.P.diagonal(), previous_P):
            self.assertLessEqual(p_predict,p_before_predict)
        for updated_state,previous_state in zip(self.ukf.x[0:2], previous_x):
            self.assertGreaterEqual(updated_state,previous_state)
        for updated_state,meaurment_state in zip(self.ukf.x[0:2], [second_incoming_measurment.x,second_incoming_measurment.y]):
            self.assertLessEqual(updated_state,meaurment_state)

