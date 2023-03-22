import unittest

import numpy as np
from utils.parameters.unit_test.test_sensor_parameters_config import test_sensor_properties
from utils.sensor.abstract_classes.abstract_sensor_properties import AbstractSensorProperties
from utils.sensor.concrete_classes.camera_sensor_properties import CameraProperties
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties
from utils.sensor.concrete_classes.radar_sensor_properties import RadarProperties

class TestAbstractSensorProperties(AbstractSensorProperties, unittest.TestCase):
    pass

class TestCameraProperties(unittest.TestCase):
    
    def setUp(self):
        self.camera_properties = CameraProperties(sensor_config= test_sensor_properties["Camera"])

    def test_camera_intrinsic(self):
        """Reading and setter test for camera intrinsic"""
        self.assertEqual(self.camera_properties.camera_intrinsic.tolist(), test_sensor_properties["Camera"]["camera_intrinsic"].tolist())

    def test_projection_matrix(self):
        """Reading and setter test for projection matrix"""
        self.assertEqual(self.camera_properties._projection_matrix_track_to_camera.tolist(),
                         np.matmul(test_sensor_properties["Camera"]["camera_intrinsic"],
                                   np.matmul(np.vstack((test_sensor_properties["Camera"]["extrinsic_cam_to_world"], np.array([0,0,0,1]))),
                                             np.vstack((test_sensor_properties["Camera"]["RT_mtx_track_to_sensor"], np.array([0,0,0,1]))))[:3,:]).tolist())
    
    def test_camera_height(self):
        """Reading and setter test for camera height"""
        self.assertEqual(self.camera_properties.camera_height, test_sensor_properties["Camera"]["camera_height"])
    
    def test_sensor_type(self):
        """Reading and setter test for sensor type"""
        self.assertEqual(self.camera_properties.sensor_type, test_sensor_properties["Camera"]["sensor_type"])
    
    def test_sensor_id(self):
        """Reading and setter test for sensor id"""
        self.assertEqual(self.camera_properties.sensor_id, test_sensor_properties["Camera"]["sensor_id"])
    
    def test_measurment_dimention(self):
        """Reading and setter test for measurment dimention"""
        self.assertEqual(self.camera_properties.measurment_dimention, test_sensor_properties["Camera"]["measurment_dimention"])

    def test_sensor_noise_matric(self):
        """Reading and setter test for sensor noise matric"""
        self.assertEqual(self.camera_properties.sensor_noise_matric.tolist(), test_sensor_properties["Camera"]["sensor_noise_matric"].tolist())
    
    def test_RT_mtx_sensor_to_track(self):
        """Reading and setter test for rotation traslation mtx from sensor to track"""
        self.assertEqual(self.camera_properties.RT_mtx_sensor_to_track.tolist(), test_sensor_properties["Camera"]["RT_mtx_sensor_to_track"].tolist())
    
    def test_RT_mtx_track_to_sensor(self):
        """Reading and setter test for rotation traslation mtx from track to sensor"""
        self.assertEqual(self.camera_properties.RT_mtx_track_to_sensor.tolist(), test_sensor_properties["Camera"]["RT_mtx_track_to_sensor"].tolist())


class TestLidarProperties(unittest.TestCase):
    
    def setUp(self):
        self.lidar_properties = LidarProperties(sensor_config= test_sensor_properties["Lidar"])

    
    def test_sensor_type(self):
        """Reading and setter test for sensor type"""
        self.assertEqual(self.lidar_properties.sensor_type, test_sensor_properties["Lidar"]["sensor_type"])
    
    def test_sensor_id(self):
        """Reading and setter test for sensor id"""
        self.assertEqual(self.lidar_properties.sensor_id, test_sensor_properties["Lidar"]["sensor_id"])
    
    def test_measurment_dimention(self):
        """Reading and setter test for measurment dimention"""
        self.assertEqual(self.lidar_properties.measurment_dimention, test_sensor_properties["Lidar"]["measurment_dimention"])

    def test_sensor_noise_matric(self):
        """Reading and setter test for sensor noise matric"""
        self.assertEqual(self.lidar_properties.sensor_noise_matric.tolist(), test_sensor_properties["Lidar"]["sensor_noise_matric"].tolist())
    
    def test_RT_mtx_sensor_to_track(self):
        """Reading and setter test for rotation traslation mtx from sensor to track"""
        self.assertEqual(self.lidar_properties.RT_mtx_sensor_to_track.tolist(), test_sensor_properties["Lidar"]["RT_mtx_sensor_to_track"].tolist())
    
    def test_RT_mtx_track_to_sensor(self):
        """Reading and setter test for rotation traslation mtx from track to sensor"""
        self.assertEqual(self.lidar_properties.RT_mtx_track_to_sensor.tolist(), test_sensor_properties["Lidar"]["RT_mtx_track_to_sensor"].tolist())

class TestRadarProperties(unittest.TestCase):
    
    def setUp(self):
        self.radar_properties = RadarProperties(sensor_config= test_sensor_properties["Radar"])
    
    def test_sensor_type(self):
        """Reading and setter test for sensor type"""
        self.assertEqual(self.radar_properties.sensor_type, test_sensor_properties["Radar"]["sensor_type"])
    
    def test_sensor_id(self):
        """Reading and setter test for sensor id"""
        self.assertEqual(self.radar_properties.sensor_id, test_sensor_properties["Radar"]["sensor_id"])
    
    def test_measurment_dimention(self):
        """Reading and setter test for measurment dimention"""
        self.assertEqual(self.radar_properties.measurment_dimention, test_sensor_properties["Radar"]["measurment_dimention"])

    def test_sensor_noise_matric(self):
        """Reading and setter test for sensor noise matric"""
        self.assertEqual(self.radar_properties.sensor_noise_matric.tolist(), test_sensor_properties["Radar"]["sensor_noise_matric"].tolist())
    
    def test_RT_mtx_sensor_to_track(self):
        """Reading and setter test for rotation traslation mtx from sensor to track"""
        self.assertEqual(self.radar_properties.RT_mtx_sensor_to_track.tolist(), test_sensor_properties["Radar"]["RT_mtx_sensor_to_track"].tolist())
    
    def test_RT_mtx_track_to_sensor(self):
        """Reading and setter test for rotation traslation mtx from track to sensor"""
        self.assertEqual(self.radar_properties.RT_mtx_track_to_sensor.tolist(), test_sensor_properties["Radar"]["RT_mtx_track_to_sensor"].tolist())
