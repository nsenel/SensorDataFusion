import unittest

import numpy as np
from utils.measurment.concrete_classes.incoming_measurments import InitialLidarMeasurment

from utils.object_factories.concrete_classes.track_object_factory import TrackKFObjectFactory, TrackKFOObjectTypeBasedFactory
from utils.parameters.concrete_classes.tracker_parameters import TrackerParameters
from utils.parameters.unit_test.test_tracker_parameters_config import tracker_parameters_car, object_based_tracker_parameters
from utils.parameters.unit_test.test_sensor_parameters_config import test_sensor_properties
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.sensor.concrete_classes.lidar_sensor import LidarSensor
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties
from utils.track_object.abstract_classes.abstract_tracked_object import AbstractTrack


class TestTrackKFObjectFactory(unittest.TestCase):
    def setUp(self):
        cam_properties   = LidarProperties(test_sensor_properties["Lidar"])
        lidar = LidarSensor(cam_properties)
        initial_measurment = InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)
        self.measurment = AbstractMeasurment(sensor_obj=lidar, measurment=initial_measurment)

    def test_generate_track_obj_without_id(self):
        """Testing for generation of track object without specified id"""
        track_object_factory = TrackKFObjectFactory(track_kf_object_factory_settings_default=TrackerParameters(tracker_parameters_car))
        track_obj = track_object_factory.generate_track_obj(self.measurment)
        self.assertEqual(track_obj.track_id, 1)
        self.assertTrue(isinstance(track_obj, AbstractTrack))

    def test_generate_track_obj_with_id(self):
        """Testing for generation of track object with specified id"""
        track_object_factory = TrackKFObjectFactory(track_kf_object_factory_settings_default=TrackerParameters(tracker_parameters_car),
                                                          generate_id_from_dict={"car":{"min":100, "max":999}})
        track_obj = track_object_factory.generate_track_obj(self.measurment)
        self.assertEqual(track_obj.track_id, 101)

class TestTrackKFOObjectTypeBasedFactory(unittest.TestCase):
    def setUp(self):
        cam_properties   = LidarProperties(test_sensor_properties["Lidar"])
        lidar = LidarSensor(cam_properties)
        initial_measurment = InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)
        self.measurment = AbstractMeasurment(sensor_obj=lidar, measurment=initial_measurment)

    def test_generate_track_obj_without_id(self):
        """Testing for generation of track object without specified id"""
        track_object_factory = TrackKFOObjectTypeBasedFactory(object_based_tracker_parameters)
        track_obj = track_object_factory.generate_track_obj(self.measurment)
        self.assertEqual(track_obj.track_id, 1)
        self.assertTrue(isinstance(track_obj, AbstractTrack))

    def test_generate_track_obj_with_id(self):
        """Testing for generation of track object with specified id"""
        kf_object_factory = TrackKFOObjectTypeBasedFactory(object_type_based_trackKF_settings=object_based_tracker_parameters,
                                                          generate_id_from_dict={"car":{"min":100, "max":999}})
        track_obj = kf_object_factory.generate_track_obj(self.measurment)
        self.assertEqual(track_obj.track_id, 101)
