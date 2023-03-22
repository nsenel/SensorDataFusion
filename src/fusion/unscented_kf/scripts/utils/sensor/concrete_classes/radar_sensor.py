from utils.sensor.abstract_classes.abstract_sensor import AbstractSensor
import numpy as np


##Typing imports
from utils.sensor.concrete_classes.radar_sensor_properties import RadarProperties
from utils.measurment.concrete_classes.incoming_measurments import InitialRadarMeasurment, TrackedObject

class RadarSensor(AbstractSensor):
    def __init__(self, sensor_properties_obj: RadarProperties) -> None:
        super().__init__(sensor_properties_obj)

    def measurment_matrix(self, incoming_measurment_from_sensor: InitialRadarMeasurment) -> np.ndarray: 
        return np.array([incoming_measurment_from_sensor.range,
                         incoming_measurment_from_sensor.bearing,
                         incoming_measurment_from_sensor.range_rate]) 

    def measurment_matrix_in_track_cor_system(self,incoming_measurment_from_sensor: InitialRadarMeasurment) -> np.ndarray:
        x = np.cos(incoming_measurment_from_sensor.bearing)*incoming_measurment_from_sensor.range
        y = np.sin(incoming_measurment_from_sensor.bearing)*incoming_measurment_from_sensor.range
        obj_in_radar_cor_system = np.array([x,y,0,1]).reshape(4,1)
        position_in_track_cordinate = np.matmul(self._sensor_properties.RT_mtx_sensor_to_track, obj_in_radar_cor_system)[:2,0]
        return position_in_track_cordinate

    def measurment_matrix_in_cost_function_cor_system(self,incoming_measurment_from_sensor: InitialRadarMeasurment) -> np.ndarray:
        if not incoming_measurment_from_sensor.transformed_cordinates_setted:
            incoming_measurment_from_sensor.set_transformed_cordinates(self.measurment_matrix_in_track_cor_system(incoming_measurment_from_sensor=incoming_measurment_from_sensor))
        return incoming_measurment_from_sensor.transformed_cordinates

    def convert_track_cor_to_sensor_cor_system(self, track_obj_position: TrackedObject, object_height: int=0) -> np.ndarray: ### Dont use height
        obj_in_track_cor_system = np.array([track_obj_position.x,track_obj_position.y,0,1]).reshape(4,1)
        position_in_radar_cordinate = np.matmul(self._sensor_properties.RT_mtx_track_to_sensor, obj_in_track_cor_system)[:2,0]
        x,y = position_in_radar_cordinate[0], position_in_radar_cordinate[1]
        v,yaw = track_obj_position.v,track_obj_position.yaw
        ### Convert cartesian coordinate system to polar coordinate system
        range = pow((x**2 + y**2),0.5)
        bearing = np.arctan2(y,x)
        v_x,v_y = np.cos(yaw)*v, np.sin(yaw)*v
        range_rate = (x*v_x + y*v_y ) / range
        
        return np.array([range,bearing,range_rate])

    def convert_tracks_to_sensor_cor_system(self, predicted_sigma_points: np.ndarray, object_height: int=0) -> np.ndarray: # Dont use height
        sigma_points_in_radar_frame = np.zeros((self.z_dim, predicted_sigma_points.shape[1]))
        for idx,sigma_point in enumerate(predicted_sigma_points.T):
            x,y,v = sigma_point[0],sigma_point[1],sigma_point[2]
            converted_point_from_ego_to_radar = np.matmul(self._sensor_properties.RT_mtx_track_to_sensor,np.array([x,y,0,1]))
            x,y = converted_point_from_ego_to_radar[0], converted_point_from_ego_to_radar[1]
            sigma_points_in_radar_frame[0,idx] = pow((x**2 + y**2),0.5) #range
            sigma_points_in_radar_frame[1,idx] = np.arctan2(y,x)        #bearing
            v_x,v_y = np.cos(sigma_points_in_radar_frame[1,idx])*v, np.sin(sigma_points_in_radar_frame[1,idx])*v
            radial_velocity = (v_x*sigma_points_in_radar_frame[0,idx]*np.cos(sigma_points_in_radar_frame[1,idx]) + v_y*sigma_points_in_radar_frame[0,idx]*np.sin(sigma_points_in_radar_frame[1,idx]))/sigma_points_in_radar_frame[0,idx]
            sigma_points_in_radar_frame[2,idx] = radial_velocity
        return sigma_points_in_radar_frame
