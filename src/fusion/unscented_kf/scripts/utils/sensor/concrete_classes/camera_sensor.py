
from utils.sensor.abstract_classes.abstract_sensor import AbstractSensor
import numpy as np

##Typing imports
from utils.sensor.concrete_classes.camera_sensor_properties import CameraProperties
from utils.measurment.concrete_classes.incoming_measurments import InitialCameraMeasurment
from utils.measurment.concrete_classes.incoming_measurments import TrackedObject

class CameraSensor(AbstractSensor):
    def __init__(self, sensor_properties_obj: CameraProperties) -> None:
        super().__init__(sensor_properties_obj)
        self._inverse_cam_intrinsic = np.linalg.inv(self.sensor_properties.project_intrinsic)
        self._inverse_cam_to_world_ext = np.linalg.inv(self.sensor_properties.extrinsic_cam_to_world[:3,:3])
        self._ext_inverse_intrinsic_multiplication = np.matmul(self._inverse_cam_to_world_ext,
                                                               self._inverse_cam_intrinsic)

    @property
    def sensor_properties(self) -> CameraProperties:
        return self._sensor_properties
    
    @property
    def inverse_cam_to_world_ext(self) -> np.ndarray:
        return self._inverse_cam_to_world_ext

    @property
    def ext_inverse_intrinsic_multiplication(self) -> np.ndarray:
        return self._ext_inverse_intrinsic_multiplication
    
    @property
    def inverse_cam_intrinsic(self) -> np.ndarray:
        return self._inverse_cam_intrinsic

    def measurment_matrix(self, incoming_measurment_from_sensor: InitialCameraMeasurment) -> np.ndarray: 
        return np.array([incoming_measurment_from_sensor.px, incoming_measurment_from_sensor.py])

    def measurment_matrix_in_track_cor_system(self, incoming_measurment_from_sensor: InitialCameraMeasurment) -> np.ndarray:
        homo_img_point = np.array([incoming_measurment_from_sensor.px, incoming_measurment_from_sensor.py, 1]).reshape(3,1)
        # left_side = np.matmul(self.ext_inverse_intrinsic_multiplication, homo_img_point)
        # right_side =  self.inverse_cam_to_world_ext * self.sensor_properties.extrinsic_cam_to_world[:,3].reshape(3,1)

        # s = (-self.sensor_properties.camera_height + right_side[2,0])/left_side[2,0]
        # obj_in_tract_cordinate = np.matmul(self.inverse_cam_to_world_ext,
        #                                    (np.matmul(s * self.inverse_cam_intrinsic, homo_img_point) - self.sensor_properties.extrinsic_cam_to_world[:,3].reshape(3,1)))
        # mea_in_tract_cor = np.matmul(self.sensor_properties.RT_mtx_sensor_to_track,
        #                             np.array([obj_in_tract_cordinate[0,0],obj_in_tract_cordinate[1,0], 0 , 1]))[:2]
        ## Shorter way of calculation ##
        mea_in_tract_cor = self.sensor_properties.projection_matrix_camera_to_track.dot(homo_img_point)
        mea_in_tract_cor = (mea_in_tract_cor[0,0]/mea_in_tract_cor[2,0],mea_in_tract_cor[1,0]/mea_in_tract_cor[2,0])
        if self.sensor_properties.fish_eye:
            return (mea_in_tract_cor[1],mea_in_tract_cor[0])
        
        return mea_in_tract_cor

    def measurment_matrix_in_cost_function_cor_system(self, incoming_measurment_from_sensor: InitialCameraMeasurment) -> np.ndarray:
        if not incoming_measurment_from_sensor.transformed_cordinates_setted:
            incoming_measurment_from_sensor.set_transformed_cordinates(self.measurment_matrix_in_track_cor_system(incoming_measurment_from_sensor=incoming_measurment_from_sensor))
        return incoming_measurment_from_sensor.transformed_cordinates

    def convert_track_cor_to_sensor_cor_system(self, track_obj_position: TrackedObject, object_height: int=0) -> np.ndarray:
        obj_in_track_cor_system = np.array([track_obj_position.x,track_obj_position.y,0,1]).reshape(4,1)
        position_in_image_cordinate = np.matmul(self.sensor_properties.projection_matrix_track_to_camera, obj_in_track_cor_system)
        return np.array((position_in_image_cordinate[:,0]/position_in_image_cordinate[2,0])[:2])
    
    def convert_tracks_to_sensor_cor_system(self, predicted_sigma_points: np.ndarray, object_height: int=0) -> np.ndarray: ## Dont use object height it is included in calibration and considered road is flat!!!!
        sigma_points_in_cam_frame = np.zeros((4, predicted_sigma_points.shape[1]))
        sigma_points_in_cam_frame[0:2,:] = predicted_sigma_points[0:2,:].copy()
        if self.sensor_properties.fish_eye:
            sigma_points_in_cam_frame[[0, 1]] = sigma_points_in_cam_frame[[1, 0]]
        sigma_points_in_cam_frame[2,:] = 0
        sigma_points_in_cam_frame[3,:] = 1
        sigma_points_in_cam_frame = np.matmul(self.sensor_properties.projection_matrix_track_to_camera,sigma_points_in_cam_frame)
        sigma_points_in_cam_frame = (sigma_points_in_cam_frame/sigma_points_in_cam_frame[2:])[0:2,:]
        return sigma_points_in_cam_frame
