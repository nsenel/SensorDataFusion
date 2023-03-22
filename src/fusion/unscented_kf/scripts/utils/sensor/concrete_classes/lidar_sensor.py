from utils.sensor.abstract_classes.abstract_sensor import AbstractSensor
import numpy as np

##Typing imports
from utils.sensor.concrete_classes.lidar_sensor_properties import LidarProperties
from utils.measurment.concrete_classes.incoming_measurments import InitialLidarMeasurment, TrackedObject

class LidarSensor(AbstractSensor):
    def __init__(self, sensor_properties_obj: LidarProperties) -> None:
        super().__init__(sensor_properties_obj)

    def measurment_matrix(self, incoming_measurment_from_sensor: InitialLidarMeasurment) -> np.ndarray: 
        return np.array([incoming_measurment_from_sensor.x, incoming_measurment_from_sensor.y])
    
    def measurment_matrix_in_track_cor_system(self, incoming_measurment_from_sensor: InitialLidarMeasurment, include_heigt: bool=False) -> np.ndarray:
        position_in_track_cordinate = np.matmul(self._sensor_properties.RT_mtx_sensor_to_track, np.array([incoming_measurment_from_sensor.x,
                                                                                                          incoming_measurment_from_sensor.y,
                                                                                                          incoming_measurment_from_sensor.z,1]))
        if include_heigt:
            return position_in_track_cordinate[0:3]
        return position_in_track_cordinate[0:2]

    def measurment_matrix_in_cost_function_cor_system(self, incoming_measurment_from_sensor: InitialLidarMeasurment) -> np.ndarray:
        if not incoming_measurment_from_sensor.transformed_cordinates_setted:
            incoming_measurment_from_sensor.set_transformed_cordinates(self.measurment_matrix_in_track_cor_system(incoming_measurment_from_sensor=incoming_measurment_from_sensor))
        return incoming_measurment_from_sensor.transformed_cordinates

    def convert_track_cor_to_sensor_cor_system(self, track_obj_position: TrackedObject, object_height: int=0) -> np.ndarray: # Dont use height
        obj_in_track_cor_system = np.array([track_obj_position.x, track_obj_position.y,0 ,1]).reshape(4,1)
        position_in_lidar_cordinate = np.matmul(self._sensor_properties.RT_mtx_track_to_sensor, obj_in_track_cor_system)
        return position_in_lidar_cordinate[:2,0]

    def convert_tracks_to_sensor_cor_system(self, predicted_sigma_points: np.ndarray, object_height: int=0) -> np.ndarray:
        sigma_points_in_lidar_frame = np.zeros((self.z_dim+2, predicted_sigma_points.shape[1]))
        sigma_points_in_lidar_frame[:self.z_dim,:]=predicted_sigma_points[:self.z_dim,:]
        sigma_points_in_lidar_frame[3,:]=1
        sigma_points_in_lidar_frame = np.matmul(self._sensor_properties.RT_mtx_track_to_sensor,sigma_points_in_lidar_frame)[0:2]
        return sigma_points_in_lidar_frame


