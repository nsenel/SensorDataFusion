import unittest
import numpy as np
from utils.kalman_filter.concrete_classes.unscented_kf import UnscentedKalmanFilter
from utils.kalman_filter.concrete_classes.unscented_kf_bridge import UnscentedKalmanFilterBridge
from utils.object_factories.concrete_classes.kf_object_factory import KalmanFilterObjectFactory
from utils.parameters.concrete_classes.tracker_parameters import TrackerParameters
from utils.parameters.unit_test.test_tracker_parameters_config import tracker_parameters_person
from utils.parameters.abstract_classes.abstract_filter_parameters import AbstractFilterParameters
from utils.track_object.concrete_classes.basic_track import BasicTrack
from utils.track_object.concrete_classes.track_decorators import AddOnGroundTruth, AddOnTrace

class TestAbstractFilterParameters(AbstractFilterParameters, unittest.TestCase):
    pass

class TestTrackerParameters(unittest.TestCase):
    
    def setUp(self):
        self.tracking_parameters_obj = TrackerParameters(tracker_parameters_person)

    def test_tracker_parameters(self):
        """Reading and setter test for tracker parameters"""
        self.assertEqual(self.tracking_parameters_obj.tracking_strategy, tracker_parameters_person["tracking_strategy"])

    def test_track_add_ons(self):
        """Reading and setter test for track add ons"""
        self.assertEqual(self.tracking_parameters_obj.track_add_ons, tracker_parameters_person["add_on_classes_dict"])
    
    def test_kf_bridge(self):
        """Reading and setter test for kf bridge"""
        self.assertTrue(isinstance(self.tracking_parameters_obj.kf_bridge, KalmanFilterObjectFactory))
    
