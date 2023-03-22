from utils.sensor.abstract_classes.abstract_sensor import AbstractSensor
import numpy as np


##Typing imports
from utils.sensor.concrete_classes.radar_sensor_properties import RadarProperties
from utils.measurment.concrete_classes.incoming_measurments import InitialCartesianRadarMeasurment, InitialRadarMeasurment, TrackedObject


class CartesianRadarSensor(AbstractSensor):
    def __init__(self, sensor_properties_obj: RadarProperties) -> None:
        super().__init__(sensor_properties_obj)

    def measurment_matrix(self, incoming_measurment_from_sensor: InitialRadarMeasurment) -> np.ndarray: 
        return np.array([incoming_measurment_from_sensor.x,
                         incoming_measurment_from_sensor.y,
                         incoming_measurment_from_sensor.v]) 

    def measurment_matrix_in_track_cor_system(self,incoming_measurment_from_sensor: InitialCartesianRadarMeasurment) -> np.ndarray:
        obj_in_radar_cor_system = np.array([incoming_measurment_from_sensor.x,incoming_measurment_from_sensor.y,0,1]).reshape(4,1)
        position_in_track_cordinate = np.matmul(self._sensor_properties.RT_mtx_sensor_to_track, obj_in_radar_cor_system)[:2,0]
        return position_in_track_cordinate

    def measurment_matrix_in_cost_function_cor_system(self,incoming_measurment_from_sensor: InitialRadarMeasurment) -> np.ndarray:
        if not incoming_measurment_from_sensor.transformed_cordinates_setted:
            incoming_measurment_from_sensor.set_transformed_cordinates(self.measurment_matrix_in_track_cor_system(incoming_measurment_from_sensor=incoming_measurment_from_sensor))
        return incoming_measurment_from_sensor.transformed_cordinates

    def convert_track_cor_to_sensor_cor_system(self, track_obj_position: TrackedObject, object_height: int=0) -> np.ndarray: # Dont use height
        obj_in_track_cor_system = np.array([track_obj_position.x, track_obj_position.y,0,1]).reshape(4,1)
        position_in_lidar_cordinate = np.matmul(self._sensor_properties.RT_mtx_track_to_sensor, obj_in_track_cor_system)
        return position_in_lidar_cordinate[:2,0]

    def convert_tracks_to_sensor_cor_system(self, predicted_sigma_points: np.ndarray, object_height: int=0) -> np.ndarray: # Dont use height
        sigma_points_in_radar_frame = np.zeros((self.z_dim, predicted_sigma_points.shape[1]))
        for idx,sigma_point in enumerate(predicted_sigma_points.T):
            position_in_radar_cordinate = np.matmul(self._sensor_properties.RT_mtx_track_to_sensor, np.array([sigma_point[0], sigma_point[1],0,1]).reshape(4,1))
            sigma_points_in_radar_frame[0,idx] = position_in_radar_cordinate[0,0]
            sigma_points_in_radar_frame[1,idx] = position_in_radar_cordinate[1,0]
            sigma_points_in_radar_frame[2,idx] = sigma_point[2]
            
        return sigma_points_in_radar_frame