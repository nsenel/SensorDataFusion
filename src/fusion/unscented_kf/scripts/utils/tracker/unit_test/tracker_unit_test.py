import unittest

import numpy as np
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.measurment.concrete_classes.incoming_measurments import InitialLidarMeasurment
from utils.measurment.concrete_classes.parsed_measurment import ParsedMeasurment
from utils.object_factories.concrete_classes.track_object_factory import TrackKFOObjectTypeBasedFactory, TrackKFObjectFactory
from utils.parameters.concrete_classes.tracker_parameters import TrackerParameters
from utils.sensor.concrete_classes.lidar_sensor import LidarSensor
from utils.tracker.concrete_classes.basic_tracker import BasicTracker
from utils.tracker.concrete_classes.cost_calculator import NormCostCalculatorSeperateByObjectType, NormCostCalculator
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties
from utils.parameters.unit_test.test_tracker_parameters_config import tracker_parameters_car, object_based_tracker_parameters_without_GT
from utils.parameters.unit_test.test_sensor_parameters_config import test_sensor_properties


class TestBasicTracker(unittest.TestCase):
    
    def setUp(self):
        lidar_properties   = LidarProperties(test_sensor_properties["Lidar"])
        lidar = LidarSensor(lidar_properties)
        
        self.detection1 = [ParsedMeasurment(sensor_obj=lidar, measurment=InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)),
                           ParsedMeasurment(sensor_obj=lidar, measurment=InitialLidarMeasurment(x=15, y=2, obj_name="car", measurment_time=1))]

        self.detection2 = [ParsedMeasurment(sensor_obj=lidar, measurment=InitialLidarMeasurment(x=9, y=1, obj_name="car", measurment_time=2)),
                           ParsedMeasurment(sensor_obj=lidar, measurment=InitialLidarMeasurment(x=20, y=4, obj_name="car", measurment_time=2)),
                           ParsedMeasurment(sensor_obj=lidar, measurment=InitialLidarMeasurment(x=50, y=4, obj_name="person", measurment_time=2))]

        self.detection3 = [ParsedMeasurment(sensor_obj=lidar, measurment=InitialLidarMeasurment(x=9.5, y=1, obj_name="car", measurment_time=2))]
        self.detection4 = [ParsedMeasurment(sensor_obj=lidar, measurment=InitialLidarMeasurment(x=10.5, y=1, obj_name="car", measurment_time=2))]
        self.detection5 = [ParsedMeasurment(sensor_obj=lidar, measurment=InitialLidarMeasurment(x=50, y=15, obj_name="car", measurment_time=2))]
        track_object_factory = TrackKFOObjectTypeBasedFactory( object_type_based_trackKF_settings = object_based_tracker_parameters_without_GT,
                                                                generate_id_from_dict=None)
        
        cost_calculator = NormCostCalculatorSeperateByObjectType()
        self.test_tracker =  BasicTracker(track_generator=track_object_factory,
                                          cost_calculator=cost_calculator,dist_thresh=20,
                                          max_frames_to_skip=2)

    def test_update_tracks_add_new_track(self):
        """Test update tracks adding new track fuction"""
        self.test_tracker.update_tracks(self.detection1, self.detection1[0].measurment_time)
        self.assertEqual(len(self.test_tracker.tracked_object_list), len(self.detection1))

    def test_update_tracks_assigment_function(self):
        """Test update tracks assigment function"""
        self.test_tracker.update_tracks(self.detection1, self.detection1[0].measurment_time)
        self.test_tracker.update_tracks(self.detection2, self.detection2[0].measurment_time)
        self.assertEqual(len(self.test_tracker.tracked_object_list), 3)
        self.assertEqual(self.test_tracker.tracked_object_list[0].hits, 2)
        self.assertEqual(self.test_tracker.tracked_object_list[1].hits, 2)
        self.assertEqual(self.test_tracker.tracked_object_list[2].hits, 1)
    
    def test_update_tracks_remove_track_function(self):
        """Test update tracks remove function"""
        self.test_tracker.update_tracks(self.detection1, self.detection1[0].measurment_time)
        self.test_tracker.update_tracks(self.detection2, self.detection2[0].measurment_time)
        self.test_tracker.update_tracks(self.detection3, self.detection3[0].measurment_time)
        self.assertEqual(len(self.test_tracker.tracked_object_list), 3)
        self.assertEqual(self.test_tracker.tracked_object_list[0].hits, 3)
        self.assertEqual(self.test_tracker.tracked_object_list[1].hits, 2)
        self.assertEqual(self.test_tracker.tracked_object_list[2].hits, 1)

        self.test_tracker.update_tracks(self.detection4, self.detection4[0].measurment_time)
        self.assertEqual(len(self.test_tracker.tracked_object_list), 3)
        self.assertEqual(self.test_tracker.tracked_object_list[0].hits, 4)
        self.assertEqual(self.test_tracker.tracked_object_list[0].skipped_frames, 0)
        self.assertEqual(self.test_tracker.tracked_object_list[1].skipped_frames, 2)
        self.assertEqual(self.test_tracker.tracked_object_list[2].skipped_frames, 2)

        self.test_tracker.update_tracks(self.detection5, self.detection5[0].measurment_time)
        self.assertEqual(len(self.test_tracker.tracked_object_list), 2)

        self.test_tracker.update_tracks([], self.detection5[0].measurment_time)
        self.assertEqual(len(self.test_tracker.tracked_object_list), 2)


