import unittest
import numpy as np

from utils.parameters.unit_test.test_sensor_parameters_config import test_sensor_properties
from utils.parameters.unit_test.test_tracker_parameters_config import test_ukf_parameters
from utils.kalman_filter.concrete_classes.unscented_kf_bridge import UnscentedKalmanFilterBridge
from utils.parameters.concrete_classes.unscented_kf_parameters import UKFFilterParameters
from utils.kalman_filter.concrete_classes.unscented_kf import UnscentedKalmanFilter
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.measurment.concrete_classes.incoming_measurments import InitialLidarMeasurment
from utils.sensor.concrete_classes.lidar_sensor import LidarSensor
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties



class TestUnscentedKalmanFilterBridge(unittest.TestCase):
    
    def setUp(self):
        filter_parameters = UKFFilterParameters(ukf_parameters=test_ukf_parameters)
        cam_properties   = LidarProperties(test_sensor_properties["Lidar"])
        self.lidar = LidarSensor(cam_properties)
        self.initial_measurment = InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)
        measurment = AbstractMeasurment(sensor_obj=self.lidar, measurment=self.initial_measurment)
        ukf = UnscentedKalmanFilter(filter_parameters=filter_parameters)
        ukf.init_measurment(initial_measurment=measurment)
        self.ukf_bridge = UnscentedKalmanFilterBridge(kalman_filter_obj=ukf)

    def test_predict_with_KF(self):
        """Test ukf predict over kf bridge"""
        x = np.zeros(test_ukf_parameters["state_dimention"])
        x[0:2] = [self.initial_measurment.x, self.initial_measurment.y]
        previous_P = self.ukf_bridge.KF.P.diagonal().copy()
        self.assertEqual(self.ukf_bridge.predict_with_KF(2), None)
        np.testing.assert_almost_equal(self.ukf_bridge.KF.x, x)
        ## Uncertainty about object location should be increased due to applying predict
        for p_predict,p_before_predict in zip(self.ukf_bridge.KF.P.diagonal(), previous_P):
            self.assertGreaterEqual(p_predict,p_before_predict)
        
    def test_update_KF(self):
        """Test ukf predict over kf bridge"""
        second_incoming_measurment = InitialLidarMeasurment(x=8.5, y=0.5, obj_name="car", measurment_time=2)
        second_measurment = AbstractMeasurment(sensor_obj=self.lidar, measurment=second_incoming_measurment)
        self.ukf_bridge.predict_with_KF(2)
        previous_x = self.ukf_bridge.KF.x.copy()[0:2]
        previous_P = self.ukf_bridge.KF.P.diagonal().copy()
        self.ukf_bridge.update_KF(second_measurment)
        for p_predict,p_before_predict in zip(self.ukf_bridge.KF.P.diagonal(), previous_P):
            self.assertLessEqual(p_predict,p_before_predict)
        for updated_state,previous_state in zip(self.ukf_bridge.KF.x[0:2], previous_x):
            self.assertGreaterEqual(updated_state,previous_state)
        for updated_state,meaurment_state in zip(self.ukf_bridge.KF.x[0:2], [second_incoming_measurment.x,second_incoming_measurment.y]):
            self.assertLessEqual(updated_state,meaurment_state)

