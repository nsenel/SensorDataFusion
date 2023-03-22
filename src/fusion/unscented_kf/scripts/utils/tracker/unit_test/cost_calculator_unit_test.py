import unittest

import numpy as np
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.measurment.concrete_classes.incoming_measurments import InitialLidarMeasurment
from utils.object_factories.concrete_classes.track_object_factory import TrackKFObjectFactory
from utils.parameters.concrete_classes.tracker_parameters import TrackerParameters
from utils.sensor.concrete_classes.lidar_sensor import LidarSensor
from utils.tracker.concrete_classes.cost_calculator import NormCostCalculatorSeperateByObjectExeptUnknown, NormCostCalculatorSeperateByObjectType, NormCostCalculator
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties
from utils.parameters.unit_test.test_tracker_parameters_config import tracker_parameters_car, object_based_tracker_parameters
from utils.parameters.unit_test.test_sensor_parameters_config import test_sensor_properties


class TestNormCostCalculator(unittest.TestCase):
    
    def setUp(self):
        lidar_properties = LidarProperties(test_sensor_properties["Lidar"])
        lidar = LidarSensor(lidar_properties)
        initial_measurment = InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)
        initial_measurment2 = InitialLidarMeasurment(x=15, y=2, obj_name="car", measurment_time=1)
        second_measurment = InitialLidarMeasurment(x=9, y=1, obj_name="car", measurment_time=2)
        second_measurment2 = InitialLidarMeasurment(x=20, y=4, obj_name="car", measurment_time=2)
        track_object_factory = TrackKFObjectFactory(TrackerParameters(tracker_parameters_car))
        track_obj_1 = track_object_factory.generate_track_obj(AbstractMeasurment(sensor_obj=lidar, measurment=initial_measurment))
        track_obj_2 = track_object_factory.generate_track_obj(AbstractMeasurment(sensor_obj=lidar, measurment=initial_measurment2))
        self.tracks = (track_obj_1, track_obj_2)
        self.detections = (AbstractMeasurment(sensor_obj=lidar, measurment=second_measurment),
                           AbstractMeasurment(sensor_obj=lidar, measurment=second_measurment2))

    def test_calculate_cost(self):
        """Test cost calculator"""
        norm_cost_calculator = NormCostCalculator()
        #Test empty track
        cost_mtx = norm_cost_calculator.calculate_cost(tracks=self.tracks,detections=[], measurment_time=self.detections[0].measurment_time)
        self.assertEqual(cost_mtx.tolist(), [[], []])
        cost_mtx = norm_cost_calculator.calculate_cost(tracks=self.tracks,detections=self.detections, measurment_time=self.detections[0].measurment_time)
        self.assertEqual(cost_mtx.shape, (len(self.detections),len(self.tracks)))
        np.testing.assert_almost_equal(cost_mtx, np.array([[ 1.41421356, 12.64911064],[ 6.0827625,  5.3851648]]))

class TestNormCostCalculatorSeperateByObjectType(unittest.TestCase):
    
    def setUp(self):
        lidar_properties = LidarProperties(test_sensor_properties["Lidar"])
        lidar = LidarSensor(lidar_properties)
        initial_measurment = InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)
        initial_measurment2 = InitialLidarMeasurment(x=15, y=2, obj_name="car", measurment_time=1)
        second_measurment = InitialLidarMeasurment(x=9, y=1, obj_name="car", measurment_time=2)
        second_measurment2 = InitialLidarMeasurment(x=20, y=4, obj_name="person", measurment_time=2)
        track_object_factory = TrackKFObjectFactory(TrackerParameters(tracker_parameters_car))
        track_obj_1 = track_object_factory.generate_track_obj(AbstractMeasurment(sensor_obj=lidar, measurment=initial_measurment))
        track_obj_2 = track_object_factory.generate_track_obj(AbstractMeasurment(sensor_obj=lidar, measurment=initial_measurment2))
        self.tracks = (track_obj_1, track_obj_2)
        self.detections = (AbstractMeasurment(sensor_obj=lidar, measurment=second_measurment),
                           AbstractMeasurment(sensor_obj=lidar, measurment=second_measurment2))

    def test_calculate_cost(self):
        """Test cost calculator"""
        norm_cost_calculator = NormCostCalculatorSeperateByObjectType()
        #Test empty track
        cost_mtx = norm_cost_calculator.calculate_cost(tracks=self.tracks,detections=[], measurment_time=self.detections[0].measurment_time)
        self.assertEqual(cost_mtx.tolist(), [[], []])
        cost_mtx = norm_cost_calculator.calculate_cost(tracks=self.tracks,detections=self.detections, measurment_time=self.detections[0].measurment_time)
        self.assertEqual(cost_mtx.shape, (len(self.detections),len(self.tracks)))
        np.testing.assert_almost_equal(cost_mtx, np.array([[ 1.41421356, 10000],[ 6.0827625, 10000]]))

class TestNormCostCalculatorSeperateByObjectExeptUnknown(unittest.TestCase):
    
    def setUp(self):
        lidar_properties = LidarProperties(test_sensor_properties["Lidar"])
        lidar = LidarSensor(lidar_properties)
        initial_measurment = InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)
        initial_measurment2 = InitialLidarMeasurment(x=15, y=2, obj_name="car", measurment_time=1)
        second_measurment = InitialLidarMeasurment(x=9, y=1, obj_name="car", measurment_time=2)
        second_measurment2 = InitialLidarMeasurment(x=20, y=4, obj_name="person", measurment_time=2)
        track_object_factory = TrackKFObjectFactory(TrackerParameters(tracker_parameters_car))
        track_obj_1 = track_object_factory.generate_track_obj(AbstractMeasurment(sensor_obj=lidar, measurment=initial_measurment))
        track_obj_2 = track_object_factory.generate_track_obj(AbstractMeasurment(sensor_obj=lidar, measurment=initial_measurment2))
        self.tracks = (track_obj_1, track_obj_2)
        self.detections = (AbstractMeasurment(sensor_obj=lidar, measurment=second_measurment),
                           AbstractMeasurment(sensor_obj=lidar, measurment=second_measurment2))

    def test_calculate_cost(self):
        """Test cost calculator"""
        norm_cost_calculator = NormCostCalculatorSeperateByObjectExeptUnknown()
        #Test empty track
        cost_mtx = norm_cost_calculator.calculate_cost(tracks=self.tracks,detections=[])
        self.assertEqual(cost_mtx.tolist(), [[], []])
        cost_mtx = norm_cost_calculator.calculate_cost(tracks=self.tracks,detections=self.detections)
        self.assertEqual(cost_mtx.shape, (len(self.detections),len(self.tracks)))
        np.testing.assert_almost_equal(cost_mtx, np.array([[ 1.41421356, 10000],[ 6.08276253,  10000]]))
