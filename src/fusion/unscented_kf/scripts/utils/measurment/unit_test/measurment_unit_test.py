import unittest

import numpy as np
from utils.parameters.unit_test.test_sensor_parameters_config import test_sensor_properties
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.measurment.concrete_classes.incoming_measurments import InitialLidarMeasurment
from utils.sensor.concrete_classes.lidar_sensor import LidarSensor

from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties





class TestAbstractMeasurment(unittest.TestCase):
    
    def setUp(self):
        cam_properties   = LidarProperties(test_sensor_properties["Lidar"])
        self.lidar = LidarSensor(cam_properties)
        self.initial_measurment = InitialLidarMeasurment(x=8, y=0, obj_name="car", measurment_time=1)
        self.abstract_measurment = AbstractMeasurment(sensor_obj=self.lidar, measurment=self.initial_measurment)

    def test_sensor_obj(self):
        """Getter test for sensor obj"""
        self.assertTrue(self.abstract_measurment.sensor_obj is self.lidar)

    def test_measurment(self):
        """Getter test for measurment"""
        self.assertTrue(self.abstract_measurment.measurment is self.initial_measurment)
    
    def test_object_name(self):
        """Getter test for object name"""
        self.assertEqual(self.abstract_measurment.obj_name, self.initial_measurment.obj_name)
    
    def test_measurment_time(self):
        """Getter test for measurment time"""
        self.assertEqual(self.abstract_measurment.measurment_time, self.initial_measurment.measurment_time)
    
    def test_sensor_type(self):
        """Getter test for sensor type"""
        self.assertEqual(self.abstract_measurment.sensor_type, test_sensor_properties["Lidar"]["sensor_type"])
    
    def test_sensor_id(self):
        """Getter test for sensor id"""
        self.assertEqual(self.abstract_measurment.sensor_id, test_sensor_properties["Lidar"]["sensor_id"])
    
    def test_z_dim(self):
        """Getter test for sensor meausrment dimention"""
        self.assertEqual(self.abstract_measurment.z_dim, test_sensor_properties["Lidar"]["measurment_dimention"])
    
    def test_z_dim(self):
        """Getter test for sensor meausrment dimention"""
        self.assertEqual(self.abstract_measurment.R.tolist(), test_sensor_properties["Lidar"]["sensor_noise_matric"].tolist())
    
    def test_measurment_matrix(self):
        """Getter test for sensor measurment matrix"""
        self.assertEqual(self.abstract_measurment.measurment_matrix.tolist(), [self.initial_measurment.x,self.initial_measurment.y])
    
    def test_measurment_matrix_in_track_cor_system(self):
        """Getter test for measurment matrix in track cor system"""
        self.assertEqual(self.abstract_measurment.measurment_matrix_in_track_cor_system.tolist(),
                         [self.initial_measurment.x,self.initial_measurment.y])
    
    def test_measurment_matrix_in_cost_function_cor_system(self):
        """Getter test for measurment matrix in cost function cor system"""
        self.assertEqual(self.abstract_measurment.measurment_matrix_in_cost_function_cor_system,
                         (self.initial_measurment.x,self.initial_measurment.y))
