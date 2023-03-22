import unittest

import numpy as np
from utils.parameters.unit_test.test_sensor_parameters_config import test_sensor_properties
from utils.measurment.concrete_classes.incoming_measurments import InitialCameraMeasurment, InitialLidarMeasurment, InitialRadarMeasurment
from utils.sensor.abstract_classes.abstract_sensor import AbstractSensor
from utils.sensor.concrete_classes.camera_sensor import CameraSensor
from utils.sensor.concrete_classes.camera_sensor_properties import CameraProperties
from utils.sensor.concrete_classes.lidar_sensor import LidarSensor
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties
from utils.sensor.concrete_classes.radar_sensor import RadarSensor
from utils.sensor.concrete_classes.radar_sensor_properties import RadarProperties

class TestAbstractSensor(AbstractSensor, unittest.TestCase):
    pass

class TestCameraSensor(unittest.TestCase):
    
    def setUp(self):
        self.camera_sensor = CameraSensor(sensor_properties_obj=CameraProperties(sensor_config= test_sensor_properties["Camera"]))
        # tracked_obj_position = TrackobjectPosition(15, 2, 0, 0)
        self.obj_position_in_meter = [14.877+0.168,  1.989] # 0.168 comes from diff between camera and lidar px=800,py=677 is without that differance
        self.incoming_measurment_from_sensor = InitialCameraMeasurment(px=800,py=677, obj_name="car", measurment_time=1)
    
    def test_sensor_type(self):
        """Getter test for sensor type"""
        self.assertEqual(self.camera_sensor.sensor_type, test_sensor_properties["Camera"]["sensor_type"])
    
    def test_sensor_id(self):
        """Getter test for sensor id"""
        self.assertEqual(self.camera_sensor.sensor_id, test_sensor_properties["Camera"]["sensor_id"])
    
    def test_measurment_dimention(self):
        """Getter test for measurment dimention"""
        self.assertEqual(self.camera_sensor.z_dim, test_sensor_properties["Camera"]["measurment_dimention"])

    def test_sensor_noise_matric(self):
        """Getter test for sensor noise matric"""
        self.assertEqual(self.camera_sensor.R.tolist(), test_sensor_properties["Camera"]["sensor_noise_matric"].tolist())
    
    def test_ext_inverse_intrinsic_multiplication(self):
        """Getter test multiplication of extrinsic matrix with inverse intrinsic"""
        self.assertEqual(self.camera_sensor.ext_inverse_intrinsic_multiplication.tolist(),
                         np.matmul(np.linalg.inv(test_sensor_properties["Camera"]["extrinsic_cam_to_world"][:3,:3]),
                                   np.linalg.inv(test_sensor_properties["Camera"]["camera_intrinsic"])).tolist())

    def test_inv_camera_intrinsic(self):
        """Getter test for inverse camera intrinsic"""
        self.assertEqual(self.camera_sensor.inverse_cam_intrinsic.tolist(), np.linalg.inv(test_sensor_properties["Camera"]["camera_intrinsic"]).tolist())

    def test_measurment_matrix(self):
        """Getter test for measurment matrix"""
        self.assertEqual(self.camera_sensor.measurment_matrix(self.incoming_measurment_from_sensor).tolist(),
                         np.array([self.incoming_measurment_from_sensor.px,
                                   self.incoming_measurment_from_sensor.py]).tolist())
    
    def test_measurment_matrix_in_track_cor_system(self):
        """Test converting from sensor cordinate system to track cor system"""
        np.testing.assert_array_almost_equal([*self.camera_sensor.measurment_matrix_in_track_cor_system(self.incoming_measurment_from_sensor)],
                                             self.obj_position_in_meter, decimal=1)
    
    def test_measurment_matrix_in_cost_function_cor_system(self):
        """Test converting from sensor cordinate system to track cor system"""
        np.testing.assert_array_almost_equal(self.camera_sensor.measurment_matrix_in_cost_function_cor_system(self.incoming_measurment_from_sensor),
                                             self.obj_position_in_meter, decimal=1)
    
    def test_convert_tracks_to_sensor_cor_system(self):
        sigma_points = np.array([[10, 10.5, 11],[2,  2.2,  2.3]])
        sigma_points_in_camera_cor = np.array([[719.967453, 708.741773, 709.446086],
                                              [746.187958, 736.209834, 727.15288 ]])
        """Test converting from sensor cordinate system to track cor system"""
        np.testing.assert_array_almost_equal(self.camera_sensor.convert_tracks_to_sensor_cor_system(sigma_points),
                                             sigma_points_in_camera_cor)
    def test_is_object_in_FOV(self):
        """Test if the track object in sensor FOV"""
        self.assertEqual(self.camera_sensor.is_object_in_FOV(np.degrees(np.arctan2(5,50))), True)
        self.assertEqual(self.camera_sensor.is_object_in_FOV(np.degrees(np.arctan2(5,-50))), False)

class TestLidarSensor(unittest.TestCase):
    
    def setUp(self):
        self.lidar_sensor = LidarSensor(sensor_properties_obj=LidarProperties(sensor_config= test_sensor_properties["Lidar"]))
        self.obj_position_in_meter = [10,  2]
        self.incoming_measurment_from_sensor = InitialLidarMeasurment(x=10,y=2, obj_name="car", measurment_time=1)
    
    def test_sensor_type(self):
        """Getter test for sensor type"""
        self.assertEqual(self.lidar_sensor.sensor_type, test_sensor_properties["Lidar"]["sensor_type"])
    
    def test_sensor_id(self):
        """Getter test for sensor id"""
        self.assertEqual(self.lidar_sensor.sensor_id, test_sensor_properties["Lidar"]["sensor_id"])
    
    def test_measurment_dimention(self):
        """Getter test for measurment dimention"""
        self.assertEqual(self.lidar_sensor.z_dim, test_sensor_properties["Lidar"]["measurment_dimention"])

    def test_sensor_noise_matric(self):
        """Getter test for sensor noise matric"""
        self.assertEqual(self.lidar_sensor.R.tolist(), test_sensor_properties["Lidar"]["sensor_noise_matric"].tolist())
    
    def test_measurment_matrix(self):
        """Getter test for measurment matrix"""
        self.assertEqual(self.lidar_sensor.measurment_matrix(self.incoming_measurment_from_sensor).tolist(),
                         np.array([self.incoming_measurment_from_sensor.x,
                                   self.incoming_measurment_from_sensor.y]).tolist())
    
    def test_measurment_matrix_in_track_cor_system(self):
        """Test converting from sensor cordinate system to track cor system"""
        np.testing.assert_array_almost_equal(self.lidar_sensor.measurment_matrix_in_track_cor_system(self.incoming_measurment_from_sensor).tolist(),
                                             self.obj_position_in_meter)
    
    def test_measurment_matrix_in_cost_function_cor_system(self):
        """Test converting from sensor cordinate system to track cor system"""
        np.testing.assert_array_almost_equal(self.lidar_sensor.measurment_matrix_in_cost_function_cor_system(self.incoming_measurment_from_sensor),
                                             self.obj_position_in_meter)
    
    def test_convert_tracks_to_sensor_cor_system(self):
        sigma_points = np.array([[10, 10.5, 11],[2,  2.2,  2.3]])
        sigma_points_in_lidar_cor = np.array([[10, 10.5, 11], [2,  2.2,  2.3]])
        """Test converting from sensor cordinate system to track cor system"""
        np.testing.assert_array_almost_equal(self.lidar_sensor.convert_tracks_to_sensor_cor_system(sigma_points),
                                             sigma_points_in_lidar_cor)
    
    def test_is_object_in_FOV(self):
        """Test if the track object in sensor FOV"""
        self.assertEqual(self.lidar_sensor.is_object_in_FOV(np.degrees(np.arctan2(5,50))), True)
        self.assertEqual(self.lidar_sensor.is_object_in_FOV(np.degrees(np.arctan2(5,-50))), False)

class TestRadarSensor(unittest.TestCase):
    
    def setUp(self):
        self.radar_sensor = RadarSensor(sensor_properties_obj=RadarProperties(sensor_config= test_sensor_properties["Radar"]))
        self.obj_position_in_meter = [8.0,  0.0]
        self.incoming_measurment_from_sensor = InitialRadarMeasurment(range=8.016,bearing=-0.06242,range_rate=0, obj_name="car", measurment_time=1)
    
    def test_sensor_type(self):
        """Getter test for sensor type"""
        self.assertEqual(self.radar_sensor.sensor_type, test_sensor_properties["Radar"]["sensor_type"])
    
    def test_sensor_id(self):
        """Getter test for sensor id"""
        self.assertEqual(self.radar_sensor.sensor_id, test_sensor_properties["Radar"]["sensor_id"])
    
    def test_measurment_dimention(self):
        """Getter test for measurment dimention"""
        self.assertEqual(self.radar_sensor.z_dim, test_sensor_properties["Radar"]["measurment_dimention"])

    def test_sensor_noise_matric(self):
        """Getter test for sensor noise matric"""
        self.assertEqual(self.radar_sensor.R.tolist(), test_sensor_properties["Radar"]["sensor_noise_matric"].tolist())
    
    def test_measurment_matrix(self):
        """Getter test for measurment matrix"""
        self.assertEqual(self.radar_sensor.measurment_matrix(self.incoming_measurment_from_sensor).tolist(),
                         np.array([self.incoming_measurment_from_sensor.range,
                                   self.incoming_measurment_from_sensor.bearing,
                                   self.incoming_measurment_from_sensor.range_rate]).tolist())
    
    def test_measurment_matrix_in_track_cor_system(self):
        """Test converting from sensor cordinate system to track cor system"""
        np.testing.assert_array_almost_equal(self.radar_sensor.measurment_matrix_in_track_cor_system(self.incoming_measurment_from_sensor).tolist(),
                                             self.obj_position_in_meter, decimal=3)
    
    def test_measurment_matrix_in_cost_function_cor_system(self):
        """Test converting from sensor cordinate system to track cor system"""
        np.testing.assert_array_almost_equal(self.radar_sensor.measurment_matrix_in_cost_function_cor_system(self.incoming_measurment_from_sensor),
                                             self.obj_position_in_meter, decimal=3)
    
    def test_convert_tracks_to_sensor_cor_system(self):
        """Test converting from sensor cordinate system to track cor system"""
        sigma_points = np.array([[7.99297704, 8.80054989, 7.99297704],
                                 [0.,         0.,         0.80757285],
                                 [0.,         0.,         0.        ],
                                 [0.,         0.,         0.        ],
                                 [0.,         0.,         0.        ]])
        sigma_points_in_radar_cor = np.array([[ 8.009,  8.815,  7.999],
                                              [-0.062, -0.057,  0.038],
                                              [ 0.   ,  0.   ,  0.   ]])
        np.testing.assert_array_almost_equal(self.radar_sensor.convert_tracks_to_sensor_cor_system(sigma_points),
                                             sigma_points_in_radar_cor, decimal=3)
    
    def test_is_object_in_FOV(self):
        """Test if the track object in sensor FOV"""
        self.assertEqual(self.radar_sensor.is_object_in_FOV(np.degrees(np.arctan2(5,50))), True)
        self.assertEqual(self.radar_sensor.is_object_in_FOV(np.degrees(np.arctan2(5,-50))), False)