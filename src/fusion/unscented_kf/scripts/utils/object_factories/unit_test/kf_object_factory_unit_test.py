import unittest

import numpy as np
from utils.kalman_filter.abstract_classes.abstract_kf_bridge import AbstractKalmanFilterBridge
from utils.measurment.concrete_classes.incoming_measurments import InitialLidarMeasurment

from utils.object_factories.concrete_classes.kf_object_factory import KalmanFilterObjectFactory
from utils.parameters.concrete_classes.unscented_kf_parameters import UKFFilterParameters
from utils.parameters.unit_test.test_tracker_parameters_config import tracker_parameters_person
from utils.parameters.unit_test.test_sensor_parameters_config import test_sensor_properties
from utils.parameters.unit_test.test_tracker_parameters_config import test_ukf_parameters
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.sensor.concrete_classes.lidar_sensor import LidarSensor
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties


class TestKalmanFilterObjectFactory(unittest.TestCase):
    
    def setUp(self):
        cam_properties   = LidarProperties(test_sensor_properties["Lidar"])
        lidar = LidarSensor(cam_properties)
        initial_measurment = InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)
        self.measurment = AbstractMeasurment(sensor_obj=lidar, measurment=initial_measurment)
        self.kf_object_factory = KalmanFilterObjectFactory(kalman_filter = tracker_parameters_person["kalman_filter"],
                                                           kalman_filter_bridge = tracker_parameters_person["kf_bridge"],
                                                           filter_settings = tracker_parameters_person["kf_settings"])

    def test_generate_kf_obj_bridge_without_settings(self):
        """Testing for generation of kf bridge object without settings"""
        self.assertTrue(isinstance(self.kf_object_factory.generate_kf_obj_bridge(self.measurment), AbstractKalmanFilterBridge))
    
    def test_generate_kf_obj_bridge_with_settings(self):
        """Testing for generation of kf bridge object with settings"""
        filter_settings = test_ukf_parameters
        filter_settings["convariance_matrix"] = np.diag([1,1,1,1,2])
        filter_settings = UKFFilterParameters(filter_settings)
        
        kf_bridge = self.kf_object_factory.generate_kf_obj_bridge(self.measurment, filter_settings)
        self.assertTrue(isinstance(kf_bridge, AbstractKalmanFilterBridge))
        self.assertNotEqual(tracker_parameters_person["kf_settings"].convariance_matrix.tolist(), kf_bridge.KF.P.tolist())
