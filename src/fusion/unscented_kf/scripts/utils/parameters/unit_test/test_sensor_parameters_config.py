import numpy as np
import cv2
from utils.common.enums import SensorTypes

class sensor_property_calculator():
       def inverse_extrinsic(extrinsic):
              return np.linalg.inv(np.vstack((extrinsic,np.array([0,0,0,1]).reshape(1,4))))[:3,:]
       
       def noise_matrix(sensor_noise_parameters):
              noise_matrix = np.zeros((len(sensor_noise_parameters), len(sensor_noise_parameters)))
              np.fill_diagonal(noise_matrix, sensor_noise_parameters)
              return noise_matrix
       
       def sensor_properties(sensor_list: dict):
              sensor_properties = {}
              for key in sensor_list.keys():
                     sensor_properties[key] = {
                            "sensor_type": sensor_list[key]["sensor_type"],
                            "sensor_id": sensor_list[key]["sensor_id"],
                            "measurment_dimention": sensor_list[key]["measurment_dimention"],
                            "sensor_noise_matric":sensor_property_calculator.noise_matrix(sensor_list[key]["sensor_noise_matric"]),
                            "RT_mtx_track_to_sensor": sensor_list[key]["extrinsic"],
                            "RT_mtx_sensor_to_track": sensor_property_calculator.inverse_extrinsic(sensor_list[key]["extrinsic"]),
                            "sensor_field_of_view": sensor_list[key]["sensor_field_of_view"]
                     }
                     if "Camera" in key: ### Camera sensor has extra parameters should be provided.
                            sensor_properties[key]["extrinsic_cam_to_world"] = sensor_list[key]["extrinsic_cam_to_world"]
                            sensor_properties[key]["camera_intrinsic"] = sensor_list[key]["camera_intrinsic"]
                            sensor_properties[key]["camera_height"] = sensor_list[key]["camera_height"]
                            sensor_properties[key]["D"] = sensor_list[key]["D"]
                            sensor_properties[key]["use_rectified_img"] = sensor_list[key]["use_rectified_img"]
                            if sensor_properties[key]["use_rectified_img"]:
                                   if "scaled_camera_matrix" in sensor_list[key].keys():
                                          scaled_camera_matrix = sensor_list[key]["scaled_camera_matrix"]
                                   else:
                                          scaled_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(sensor_list[key]["camera_intrinsic"], np.array(sensor_list[key]["D"]), (1216,1936), 1, (1216,1936))
                                   sensor_properties[key]["scaled_camera_matrix"] = scaled_camera_matrix
                            if "fish_eye" in sensor_list[key].keys() and sensor_list[key]["fish_eye"]:
                                   sensor_properties[key]["scaled_camera_matrix"] = sensor_list[key]["scaled_camera_matrix"]
                                   sensor_properties[key]["fish_eye"] = True
              return sensor_properties

test_sensor_list = {
                    "Camera":{
                        "sensor_type": SensorTypes.camera,
                        "sensor_id":1,
                        "measurment_dimention":2,
                        "camera_intrinsic":np.array([[1180, 0,    960],
                                                    [0,    1180, 540],
                                                    [0,    0,    1]]),
                        "extrinsic":np.array([[1., 0., 0.,  -0.168], #camera is in from depend on camera location
                                           [0., 1., 0., -0.],
                                           [0., 0., 1., -0.]]), ### height between lidar and camera -0.612
                        "extrinsic_cam_to_world":np.array([[-0., -1., -0.,  0],
                                                           [-0., -0., -1.,  1.718],
                                                           [ 1.,  0.,  0.,  0]]),
                        "sensor_noise_matric": [4,4], #Camera measurement noise standard deviation hornizantal axis-x  and vertical axis-y as pixel
                        "camera_height": 1.718,
                        "D": np.array([0,0,0,0,0]),
                        "sensor_field_of_view": (270,90),
                        "use_rectified_img": False
                     },
                    "Lidar":{
                            "sensor_type": SensorTypes.lidar,
                            "sensor_id":2,
                            "measurment_dimention":2,
                            "extrinsic":np.array([[1., 0., 0., 0.],
                                                [0., 1., 0., 0.],
                                                [0., 0., 1., 0.]]),
                            "sensor_noise_matric": [0.15,0.15], #Laser measurement noise standard deviation depth and hornizan(left-right) in m
                            "sensor_field_of_view": (270,90),
                            },
                    "Radar":{
                            "sensor_type": SensorTypes.radar_cartesian,
                            "sensor_id":3,
                            "measurment_dimention":3,
                            "extrinsic":np.array([[1., 0., 0.,  0. ],
                                                [0., 1., 0., -0.5],
                                                [0., 0., 1.,  0. ]]),
                            "sensor_noise_matric": [0.3, 0.03, 0.3], #Radar measurement noise standard deviation radius in m, angle in rad,radius change in m/s
                            "sensor_field_of_view": (270,90),
                            }
                    }
test_sensor_properties = sensor_property_calculator.sensor_properties(test_sensor_list)
# test_sensor_properties ={
#                         "Camera":{
#                                 "sensor_type": test_sensor_list["Camera"]["sensor_type"],
#                                 "sensor_id": test_sensor_list["Camera"]["sensor_id"],
#                                 "measurment_dimention":2,
#                                 "sensor_noise_matric":sensor_property_calculator.noise_matrix(test_sensor_list["Camera"]["sensor_noise_matric"]),
#                                 "RT_mtx_track_to_sensor": test_sensor_list["Camera"]["extrinsic"],
#                                 "RT_mtx_sensor_to_track": sensor_property_calculator.inverse_extrinsic(test_sensor_list["Camera"]["extrinsic"]),
#                                 "camera_intrinsic":test_sensor_list["Camera"]["camera_intrinsic"],
#                                 "camera_height": test_sensor_list["Camera"]["camera_height"]
#                             },
#                         "Lidar":{
#                                 "sensor_type": test_sensor_list["Lidar"]["sensor_type"],
#                                 "sensor_id": test_sensor_list["Lidar"]["sensor_id"],
#                                 "measurment_dimention":2,
#                                 "sensor_noise_matric":sensor_property_calculator.noise_matrix(test_sensor_list["Lidar"]["sensor_noise_matric"]),
#                                 "RT_mtx_track_to_sensor": test_sensor_list["Lidar"]["extrinsic"],
#                                 "RT_mtx_sensor_to_track": sensor_property_calculator.inverse_extrinsic(test_sensor_list["Lidar"]["extrinsic"]),
#                                 },
#                         "Radar":{
#                                 "sensor_type": test_sensor_list["Radar"]["sensor_type"],
#                                 "sensor_id": test_sensor_list["Radar"]["sensor_id"],
#                                 "measurment_dimention":3,
#                                 "sensor_noise_matric":sensor_property_calculator.noise_matrix(test_sensor_list["Radar"]["sensor_noise_matric"]),
#                                 "RT_mtx_track_to_sensor": test_sensor_list["Radar"]["extrinsic"],
#                                 "RT_mtx_sensor_to_track": sensor_property_calculator.inverse_extrinsic(test_sensor_list["Radar"]["extrinsic"]),
#                                 },
#                     }