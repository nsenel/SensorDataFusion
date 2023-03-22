from utils.sensor.abstract_classes.abstract_sensor_properties import AbstractSensorProperties
#Typing exports
import numpy as np
import enum

class CameraProperties(AbstractSensorProperties):
    def __init__(self, sensor_config: dict) -> None:
        super().__init__(sensor_config)
        
        self._project_intrinsic = self.camera_intrinsic
        if self.use_rectified_img:
            self._project_intrinsic = self.scaled_camera_matrix
        #Calculate the projectin matrix
        self._projection_matrix_track_to_camera = np.matmul(self._project_intrinsic,
                                                            np.matmul(np.vstack((self.extrinsic_cam_to_world, np.array([0,0,0,1]))),
                                                                      np.vstack((self.RT_mtx_track_to_sensor, np.array([0,0,0,1]))))[:3,:])### Dont like it

        # Where image point real height is 0 image point can be converted to world point
        self._zero_height_inverse_projection_cam_to_track = np.linalg.inv(np.delete(self._projection_matrix_track_to_camera.copy(), 2, 1))
    
    @property ##camera spesific
    def project_intrinsic(self) -> np.ndarray:
        return self._project_intrinsic.copy()

    @property ##camera spesific
    def camera_intrinsic(self) -> np.ndarray:
        return self.get_sensor_property("camera_intrinsic")
    
    @property ##camera spesific
    def projection_matrix_track_to_camera(self) -> np.ndarray:
        return self._projection_matrix_track_to_camera
    
    @property ##camera spesific
    def projection_matrix_camera_to_track(self) -> np.ndarray:
        return self._zero_height_inverse_projection_cam_to_track
    
    @property ##camera spesific
    def camera_height(self) -> float:
        return self.get_sensor_property("camera_height")
    
    @property ##camera spesific
    def extrinsic_cam_to_world(self) ->np.ndarray:
        return self.get_sensor_property("extrinsic_cam_to_world")

    @property
    def sensor_type(self) -> enum:
        return self.get_sensor_property("sensor_type")

    @property
    def sensor_id(self) -> int:
        return self.get_sensor_property("sensor_id")
    
    @property
    def measurment_dimention(self) -> int:
        return self.get_sensor_property("measurment_dimention")
    
    @property
    def sensor_noise_matric(self) -> np.ndarray:
        return self.get_sensor_property("sensor_noise_matric")
    
    @property
    def RT_mtx_sensor_to_track(self) -> np.ndarray:
        return self.get_sensor_property("RT_mtx_sensor_to_track")
    
    @property
    def RT_mtx_track_to_sensor(self) -> np.ndarray:
        return self.get_sensor_property("RT_mtx_track_to_sensor")
    
    @property
    def sensor_field_of_view(self)->tuple: #((x_min,x_max), (y_min,y_max))
        return self.get_sensor_property("sensor_field_of_view")
    
    @property
    def distortion_coefficients(self)->np.ndarray:
        return self.get_sensor_property("D")

    @property
    def use_rectified_img(self)->bool:
        return self.get_sensor_property("use_rectified_img")
    
    @property
    def scaled_camera_matrix(self)->np.ndarray:
        return self.get_sensor_property("scaled_camera_matrix")
    
    @property
    def fish_eye(self)->np.ndarray:
        return self.get_sensor_property("fish_eye")
        