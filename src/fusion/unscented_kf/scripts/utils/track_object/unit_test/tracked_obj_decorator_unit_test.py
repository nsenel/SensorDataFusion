from collections import deque
import unittest
import numpy as np
from utils.movement_compensation.concrete_classes.movement_compensation_with_global_change import MVWithGlobalPostionChange

from utils.parameters.unit_test.test_sensor_parameters_config import test_sensor_properties
from utils.parameters.unit_test.test_tracker_parameters_config import test_ukf_parameters
from utils.kalman_filter.concrete_classes.unscented_kf_bridge import UnscentedKalmanFilterBridge
from utils.parameters.concrete_classes.unscented_kf_parameters import UKFFilterParameters
from utils.kalman_filter.concrete_classes.unscented_kf import UnscentedKalmanFilter
from utils.measurment.concrete_classes.parsed_measurment import ParsedMeasurmentWithGT
from utils.measurment.concrete_classes.incoming_measurments import EgoCarOdomMessage, InitialLidarMeasurment, LidarMeasurmentwithGT
from utils.sensor.concrete_classes.lidar_sensor import LidarSensor
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties
from utils.track_object.abstract_classes.tracked_obj_decorator import TrackDecorator
from utils.track_object.concrete_classes.basic_track import BasicTrack
from utils.track_object.concrete_classes.track_decorators import AddOnGroundTruth, AddOnMovementCompensation, AddOnTrace

class TestTrackDecorator(unittest.TestCase):
    
    def setUp(self):
        filter_parameters = UKFFilterParameters(ukf_parameters=test_ukf_parameters)
        cam_properties   = LidarProperties(test_sensor_properties["Lidar"])
        self.lidar = LidarSensor(cam_properties)
        
        self.initial_measurment = LidarMeasurmentwithGT(x=8.0, y=0.0, obj_name="car", measurment_time=1, x_gt=7.95, y_gt=0.1, vx_gt=1, vy_gt=0.1, yaw_gt=0, yawrate_gt=0, gt_id=5)
        measurment = ParsedMeasurmentWithGT(sensor_obj=self.lidar, measurment=self.initial_measurment)
        self.ukf = UnscentedKalmanFilter(filter_parameters=filter_parameters)
        self.ukf.init_measurment(initial_measurment=measurment)
        self.ukf_bridge = UnscentedKalmanFilterBridge(kalman_filter_obj=self.ukf)
        self.track_id = 1
        self.basic_traker = BasicTrack(track_id=self.track_id, kalman_filter_bridge=self.ukf_bridge, track_object_type=self.initial_measurment.obj_name)
        self.track_decorator = TrackDecorator(track_object=self.basic_traker)
    
    def test_add_on_ground_truth(self):
        """AddOnGroundTruth decorator behaviour"""
        ### Force to detect not matching gt_id measurment with inital gt_id
        add_on_ground_truth_track = AddOnGroundTruth(self.basic_traker)
        add_on_ground_truth_track.predict_next_state(2)
        incoming_measurment2 = LidarMeasurmentwithGT(x=8.5, y=0.0, obj_name="car", measurment_time=2, x_gt=8.25, y_gt=0.1, vx_gt=1, vy_gt=0.1, yaw_gt=0, yawrate_gt=0, gt_id=5)
        measurment2 = ParsedMeasurmentWithGT(sensor_obj=self.lidar, measurment=incoming_measurment2)
        add_on_ground_truth_track.correct_prediction(measurment2)
        add_on_ground_truth_track.predict_next_state(3)
        
        incoming_measurment3 = LidarMeasurmentwithGT(x=10.5, y=0.5, obj_name="car", measurment_time=3, x_gt=10.25, y_gt=0.4, vx_gt=1, vy_gt=0.1, yaw_gt=0, yawrate_gt=0, gt_id=7)
        measurment3 = ParsedMeasurmentWithGT(sensor_obj=self.lidar, measurment=incoming_measurment3)
        add_on_ground_truth_track.correct_prediction(measurment3)
        #### Decorator over rides previous gt_id and gives console out put stating that gt_id not match
        self.assertEqual(incoming_measurment3.gt_id, add_on_ground_truth_track._ground_truth_id)
        
    def test_add_on_trace(self):
        """AddOnTrace decorator behaviour"""
        ### Force to detect not matching gt_id measurment with inital gt_id
        add_on_trace_track = AddOnTrace(self.basic_traker,2)

        add_on_trace_track.predict_next_state(2)
        incoming_measurment2 = InitialLidarMeasurment(x=8.5, y=0.0, obj_name="car", measurment_time=2)
        measurment2 = ParsedMeasurmentWithGT(sensor_obj=self.lidar, measurment=incoming_measurment2)
        add_on_trace_track.correct_prediction(measurment2)
        self.assertEqual(1, len(add_on_trace_track._trace))

        add_on_trace_track.predict_next_state(3)
        incoming_measurment3 = InitialLidarMeasurment(x=10.5, y=0.5, obj_name="car", measurment_time=3)
        measurment3 = ParsedMeasurmentWithGT(sensor_obj=self.lidar, measurment=incoming_measurment3)
        add_on_trace_track.correct_prediction(measurment3)
        self.assertEqual(2, len(add_on_trace_track._trace))

        add_on_trace_track.predict_next_state(4)
        incoming_measurment4 = InitialLidarMeasurment(x=11.5, y=0.5, obj_name="car", measurment_time=4)
        measurment4 = ParsedMeasurmentWithGT(sensor_obj=self.lidar, measurment=incoming_measurment4)
        add_on_trace_track.correct_prediction(measurment4)
        self.assertEqual(2, len(add_on_trace_track._trace))

        self.assertTrue(type(add_on_trace_track.trace) is deque)
    
    def test_add_on_movement_compensation(self):
        """AddOnMovementCompensation decorator behaviour"""
        movement_compensation_method = MVWithGlobalPostionChange()
        movement_compensation_method.update_state(EgoCarOdomMessage(x=9,y=1,timestamp=1,yaw=0,vx=2,vy=1))
        movement_compensation_method.update_state(EgoCarOdomMessage(x=10,y=2,timestamp=2,yaw=0,vx=2,vy=1))
        add_on_mv_track = AddOnMovementCompensation(self.basic_traker, movement_compensation_method)
        add_on_mv_track.predict_next_state(2)
        temp_x = self.basic_traker.kalman_filter_bridge.KF.x.copy()
        
        basic_traker2 = BasicTrack(track_id=self.track_id, kalman_filter_bridge=self.ukf_bridge, track_object_type=self.initial_measurment.obj_name)
        add_on_ground_truth_track = AddOnGroundTruth(basic_traker2)
        add_on_ground_truth_track.predict_next_state(2)
        self.assertNotEqual(basic_traker2.kalman_filter_bridge.KF.x.tolist(), temp_x.tolist())

    def test_track_id(self):
        """Getter test for track id"""
        self.assertEqual(self.track_decorator.track_id, self.track_id)
    
    def test_kalman_filter_bridge(self):
        """Getter test for kalman filter bridge"""
        self.assertTrue(self.track_decorator.kalman_filter_bridge is self.ukf_bridge)
    
    def test_KF(self):
        """Getter test for KF object"""
        self.assertTrue(self.track_decorator.KF is self.ukf)

    def test_skipped_frames(self):
        """Getter test for skipped frames"""
        self.assertEqual(self.track_decorator.skipped_frames, 0)
    
    def test_hits(self):
        """Getter test for hits"""
        self.assertEqual(self.track_decorator.hits, 1)
    
    def test_track_object_type(self):
        """Getter test for track object type"""
        self.assertEqual(self.track_decorator.track_object_type, self.initial_measurment.obj_name)
    
    def test_track_prediction(self):
        """Getter test for track prediction"""
        self.assertEqual(self.track_decorator.prediction.tolist(), [self.initial_measurment.x, self.initial_measurment.y])
