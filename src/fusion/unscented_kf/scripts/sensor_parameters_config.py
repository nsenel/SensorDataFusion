from pickle import FALSE
import numpy as np
import cv2

from utils.common.enums import SensorTypes
from utils.sensor.concrete_classes.camera_sensor import CameraSensor
from utils.sensor.concrete_classes.camera_sensor_properties import CameraProperties
from utils.sensor.concrete_classes.lidar_sensor import LidarSensor
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties
from utils.sensor.concrete_classes.radar_sensor import RadarSensor
from utils.sensor.concrete_classes.cartesian_radar_sensor import CartesianRadarSensor
from utils.sensor.concrete_classes.radar_sensor_properties import RadarProperties


"""    Info
sensor_field_of_view:
It repesent in global frame which area sensor can see for example camera looking in right side of the road cant see left side of the road.
Left side sensor parameter can be for example (90,270) thats meean sensor able to see objects from 90 degree to 270 degree.(order is important it is clockwise)
Likewise right side of sensor can be (270,90). If you dont know what assign you can put (0,360) thats mean sensor can see all the objects.
"""

class sensor_property_calculator():
       def inverse_extrinsic(extrinsic):
              return np.linalg.inv(np.vstack((extrinsic,np.array([0,0,0,1]).reshape(1,4))))[:3,:]
       
       def noise_matrix(sensor_noise_parameters):
              sensor_noise_parameters = [param**2 for param in sensor_noise_parameters]
              noise_matrix = np.zeros((len(sensor_noise_parameters), len(sensor_noise_parameters)))
              np.fill_diagonal(noise_matrix, sensor_noise_parameters)
              return noise_matrix
       
       def generate_sensors_properties(sensor_list: dict):
              sensors_properties = {}
              for key in sensor_list.keys():
                     sensors_properties[key] = {
                            "sensor_type": sensor_list[key]["sensor_type"],
                            "sensor_id": sensor_list[key]["sensor_id"],
                            "measurment_dimention": sensor_list[key]["measurment_dimention"],
                            "sensor_noise_matric":sensor_property_calculator.noise_matrix(sensor_list[key]["sensor_noise_matric"]),
                            "RT_mtx_track_to_sensor": sensor_list[key]["extrinsic"],
                            "RT_mtx_sensor_to_track": sensor_property_calculator.inverse_extrinsic(sensor_list[key]["extrinsic"]),
                            "sensor_field_of_view": sensor_list[key]["sensor_field_of_view"]
                     }
                     if "Camera" in key: ### Camera sensor has extra parameters should be provided.
                            sensors_properties[key]["extrinsic_cam_to_world"] = sensor_list[key]["extrinsic_cam_to_world"]
                            sensors_properties[key]["camera_intrinsic"] = sensor_list[key]["camera_intrinsic"]
                            sensors_properties[key]["camera_height"] = sensor_list[key]["camera_height"]
                            sensors_properties[key]["D"] = sensor_list[key]["D"]
                            sensors_properties[key]["use_rectified_img"] = sensor_list[key]["use_rectified_img"]
                            if sensors_properties[key]["use_rectified_img"]:
                                   if "scaled_camera_matrix" in sensor_list[key].keys():
                                          scaled_camera_matrix = sensor_list[key]["scaled_camera_matrix"]
                                   else:
                                          scaled_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(sensor_list[key]["camera_intrinsic"], np.array(sensor_list[key]["D"]), (1216,1936), 1, (1216,1936))
                                   sensors_properties[key]["scaled_camera_matrix"] = scaled_camera_matrix
                            if "fish_eye" in sensor_list[key].keys() and sensor_list[key]["fish_eye"]:
                                   sensors_properties[key]["scaled_camera_matrix"] = sensor_list[key]["scaled_camera_matrix"]
                                   sensors_properties[key]["fish_eye"] = True

                                          
              return sensors_properties
       
       def generate_sensor_objs_from_sensors(sensors_properties:dict):
              sensor_objs = dict()
              for sensor_name, sensor_properties in sensors_properties.items():
                     if sensor_properties["sensor_type"] == SensorTypes.camera or sensor_properties["sensor_type"] == SensorTypes.fisheye:
                            cam_properties   = CameraProperties(sensor_properties)
                            sensor_objs[sensor_name] = CameraSensor(cam_properties)
                     elif sensor_properties["sensor_type"] == SensorTypes.lidar:
                            lidar_properties = LidarProperties(sensor_properties)
                            sensor_objs[sensor_name] = LidarSensor(lidar_properties)
                     elif sensor_properties["sensor_type"] == SensorTypes.radar_polar:
                            radar_properties = RadarProperties(sensor_properties)
                            sensor_objs[sensor_name] = RadarSensor(radar_properties)
                     elif sensor_properties["sensor_type"] == SensorTypes.radar_cartesian:
                            radar_properties = RadarProperties(sensor_properties)
                            sensor_objs[sensor_name] = CartesianRadarSensor(radar_properties)
              return sensor_objs


sensor_list_lgsvl = {"Front_Camera":{
                            "sensor_type": SensorTypes.camera,
                            "sensor_id":1,
                            "measurment_dimention":2,
                            "camera_intrinsic":np.array([[1180, 0,    960],
                                                        [0,    1180, 540],
                                                        [0,    0,    1]]),
                            "extrinsic":np.array([[1., 0., 0.,  0.168], #camera is in from depend on camera location
                                                 [0., 1., 0., -0.],
                                                 [0., 0., 1., -0.]]), ### height between lidar and camera -0.612
                            "extrinsic_cam_to_world":np.array([[-0., -1., -0.,  0],
                                                               [-0., -0., -1.,  1.718],
                                                               [ 1.,  0.,  0.,  0]]),
                            "D": np.array([-0., 0. ,-0., -0., 0.]),
                            "sensor_noise_matric": [4,4], #Camera measurement noise standard deviation hornizantal axis-x  and vertical axis-y as pixel
                            "camera_height": 1.718,
                            "sensor_field_of_view": (0,360), ### Since I make object list I take care of it there for simulation it doesnt have use
                            "use_rectified_img": False ## Simulation images are already rectified
                            },
                     "Front_Lidar":{
                            "sensor_type": SensorTypes.lidar,
                            "sensor_id":2,
                            "measurment_dimention":2,
                            "extrinsic":np.array([[1., 0., 0., 0.],
                                                 [0., 1., 0., 0.],
                                                 [0., 0., 1., -2.312]]),
                            "sensor_noise_matric": [0.15,0.15], #Laser measurement noise standard deviation depth and hornizan(left-right) in m
                            "sensor_field_of_view": (0,360)
                            },
                     "Front_Radar":{
                            "sensor_type": SensorTypes.radar_polar,
                            "sensor_id":3,
                            "measurment_dimention":3,
                            "extrinsic":np.array([[1., 0., 0.,  0. ],
                                                 [0., 1., 0., -0.5],
                                                 [0., 0., 1.,  0. ]]),
                            "sensor_noise_matric": [0.3, 0.03, 0.3], #Radar measurement noise standard deviation radius in m, angle in rad,radius change in m/s
                            "sensor_field_of_view": (0,360)
                            }
                     }


sensors_properties_lgsvl = sensor_property_calculator.generate_sensors_properties(sensor_list_lgsvl)
configured_sensor_objs_lgsvl = sensor_property_calculator.generate_sensor_objs_from_sensors(sensors_properties_lgsvl)


in2lab_sensor_config_map = {"configured_sensor_objs_lgsvl":configured_sensor_objs_lgsvl}