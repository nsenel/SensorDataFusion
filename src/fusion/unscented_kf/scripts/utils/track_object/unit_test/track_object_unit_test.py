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
from utils.track_object.concrete_classes.basic_track import BasicTrack

class TestBasicTrack(unittest.TestCase):
    
    def setUp(self):
        filter_parameters = UKFFilterParameters(ukf_parameters=test_ukf_parameters)
        cam_properties   = LidarProperties(test_sensor_properties["Lidar"])
        self.lidar = LidarSensor(cam_properties)
        self.initial_measurment = InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)
        measurment = AbstractMeasurment(sensor_obj=self.lidar, measurment=self.initial_measurment)
        self.ukf = UnscentedKalmanFilter(filter_parameters=filter_parameters)
        self.ukf.init_measurment(initial_measurment=measurment)
        self.ukf_bridge = UnscentedKalmanFilterBridge(kalman_filter_obj=self.ukf)
        self.track_id = 1
        self.basic_traker = BasicTrack(track_id=self.track_id, kalman_filter_bridge=self.ukf_bridge, track_object_type=self.initial_measurment.obj_name)

    def test_track_id(self):
        """Getter test for track id"""
        self.assertEqual(self.basic_traker.track_id, self.track_id)
    
    def test_kalman_filter_bridge(self):
        """Getter test for kalman filter bridge"""
        self.assertTrue(self.basic_traker.kalman_filter_bridge is self.ukf_bridge)
    
    def test_KF(self):
        """Getter test for KF object"""
        self.assertTrue(self.basic_traker.KF is self.ukf)

    def test_skipped_frames(self):
        """Getter test for skipped frames"""
        self.assertEqual(self.basic_traker.skipped_frames, 0)
    
    def test_hits(self):
        """Getter test for hits"""
        self.assertEqual(self.basic_traker.hits, 1)
    
    def test_track_object_type(self):
        """Getter test for track object type"""
        self.assertEqual(self.basic_traker.track_object_type, self.initial_measurment.obj_name)
    
    def test_track_prediction(self):
        """Getter test for track prediction"""
        self.assertEqual(self.basic_traker.prediction.tolist(), [self.initial_measurment.x, self.initial_measurment.y])
    
    def test_predict_next_state(self):
        """Test ukf predict over track obj"""
        x= [self.initial_measurment.x, self.initial_measurment.y]
        previous_P = self.basic_traker.kalman_filter_bridge.KF.P.diagonal().copy()
        self.assertEqual(self.basic_traker.predict_next_state(2), None)
        self.assertEqual(self.basic_traker.prediction.tolist(), x)
        ## Uncertainty about object location should be increased due to applying predict
        for p_predict,p_before_predict in zip(self.basic_traker.kalman_filter_bridge.KF.P.diagonal(), previous_P):
            self.assertGreaterEqual(p_predict,p_before_predict)
    
    def test_correct_prediction(self):
        """Test ukf predict over kf bridge"""
        second_incoming_measurment = InitialLidarMeasurment(x=8.5, y=0.5, obj_name="car", measurment_time=2)
        second_measurment = AbstractMeasurment(sensor_obj=self.lidar, measurment=second_incoming_measurment)
        self.basic_traker.predict_next_state(2)
        previous_x = self.basic_traker.kalman_filter_bridge.KF.x.copy()[0:2]
        previous_P = self.basic_traker.kalman_filter_bridge.KF.P.diagonal().copy()
        self.basic_traker.correct_prediction(second_measurment)
        for p_predict,p_before_predict in zip(self.basic_traker.kalman_filter_bridge.KF.P.diagonal(), previous_P):
            self.assertLessEqual(p_predict,p_before_predict)
        for updated_state,previous_state in zip(self.basic_traker.prediction, previous_x):
            self.assertGreaterEqual(updated_state,previous_state)
        for updated_state,meaurment_state in zip(self.basic_traker.prediction, [second_incoming_measurment.x,second_incoming_measurment.y]):
            self.assertLessEqual(updated_state,meaurment_state)
