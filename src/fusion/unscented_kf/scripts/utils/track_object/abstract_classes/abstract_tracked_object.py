"""
    @author: Numan Senel
    @email: Numan.Senel@thi.de
"""
# Import python libraries
from abc import ABC, abstractmethod, abstractproperty
import numpy as np
from utils.common.enums import SensorTypes

#Typing imports
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.kalman_filter.abstract_classes.abstract_kf import AbstractKalmanFilterImplementer
from utils.kalman_filter.abstract_classes.abstract_kf_bridge import AbstractKalmanFilterBridge

class AbstractTrack(ABC):
    """Track class for every object to be tracked"""
    def __init__(self, track_id: int, kalman_filter_bridge: AbstractKalmanFilterBridge, initial_measurment: AbstractMeasurment= None):
        """Initialize variables used by Track class
        Args:
            track_id: Identification of each track object
            detected_object: Detected object expected proporties (center,bbox and prediction)
            max_trace_length: Trace path history length
        Return:
            None
        """
        self._track_id = track_id  # identification of each track object
        self._kalman_filter_bridge = kalman_filter_bridge#UKF(detected_object.time_stamp, track_id)  # KF instance to track this object
        self._skipped_frames = 0  # number of frames skipped undetected
        self._hits = 1 # number of detection matches
        self._tracked_obj_name = None
        self._detected_by_sensors = {sensor:{"detected_in_last":False, "last_measurment":None, "detected_cnt":0} for sensor in (SensorTypes)}
        if initial_measurment:
            self._detected_by_sensors[initial_measurment.sensor_type]["detected_in_last"]=True
            self._detected_by_sensors[initial_measurment.sensor_type]["last_measurment"]=initial_measurment
            self._detected_by_sensors[initial_measurment.sensor_type]["detected_cnt"]=1
        self._last_measurment_time = self._kalman_filter_bridge.KF.last_update_time ## Dont use KF.last_update_time exept here, it is updated not only new measurmnet but also new prediction
        self._remove_me = False
        self._valid_object = False
        self._additional_time_before_remove = {"valid_until": 0, "additional_time": 0}
    
    @abstractproperty
    def track_id(self) ->int:
        """ Getter for _track_id
        Args:
            None
        Return:
            track_id 
        """
        return self._track_id
    
    @abstractproperty
    def kalman_filter_bridge(self) ->AbstractKalmanFilterBridge:
        """ Getter for kalman filter implementer bridge
        Args:
            None
        Return:
            AbstractKalmanFilterBridge 
        """
        return self._kalman_filter_bridge
    
    @abstractproperty
    def KF(self) ->AbstractKalmanFilterImplementer:
        """ Getter for kalman filter implementer
        Args:
            None
        Return:
            AbstractKalmanFilterImplementer 
        """
        return self._kalman_filter_bridge.KF

    @abstractproperty
    def skipped_frames(self) ->int:
        """ Getter for _skipped_frames
        Args:
            None
        Return:
            skipped_frames 
        """
        return self._skipped_frames
    
    @abstractproperty
    def hits(self) ->int:
        """ Getter for _hits
        Args:
            None
        Return:
            number of hits
        """
        return self._hits
    
    @abstractproperty
    def last_measurment_time(self) ->int:
        """ Getter for _last_measurment_time
        Args:
            None
        Return:
            number of hits
        """
        return self._last_measurment_time
    
    @abstractproperty
    def track_object_type(self) ->str:
        """ Getter for __tracked_obj_name
        Args:
            None
        Return:
            name of track object type
        """
        return self._tracked_obj_name
    
    @abstractproperty
    def prediction(self) ->np.ndarray:
        """ Getter for track object position(x,y)
        Args:
            None
        Return:
            Predicted state vector values for (x,y) 
        """
        return np.array([self.KF.x_postion, self.KF.y_postion])
    
    @abstractproperty
    def detected_by_sensors(self) ->dict:
        """ Getter for seing which sensors are detected this object in their "last" object detection time
        Args:
            None
        Return:
            sensor_type:Bool
        """
        return self._detected_by_sensors
    
    @abstractproperty
    def remove_me(self) ->bool:
        """ Getter for _remove_me
        Args:
            None
        Return:
            bool
        """
        return self._remove_me

    @abstractproperty
    def valid_object(self) ->bool:
        """ Getter for _valid_object
        Args:
            None
        Return:
            bool
        """
        return self._valid_object
    
    @abstractproperty
    def additional_time_before_remove(self) ->int:
        """ Getter for _additional_time_before_remove
        Args:
            None
        Return:
            int
        """
        if self._additional_time_before_remove["valid_until"]>self._kalman_filter_bridge.KF.last_update_time:
            return self._additional_time_before_remove["additional_time"]
        return 0

    @abstractmethod
    def set_skipped_frames(self, skipped_frames: int) ->None:
        self._skipped_frames= skipped_frames

    @abstractmethod
    def set_last_meaurment_time(self,measurment_time: int) ->None:
        self._last_measurment_time=measurment_time
    
    @abstractmethod
    def set_object_type(self, object_type: str) ->None:
        self._tracked_obj_name=object_type
    
    @abstractmethod
    def set_number_of_hits(self,hits: int) ->None:
        self._hits=hits
    
    @abstractmethod
    def set_valid_obj_state(self,is_valid: bool) ->None:
        self._valid_object = is_valid
    
    @abstractmethod
    def set_remove_state(self,self_remove: bool) ->None:
        self._remove_me = self_remove
    
    @abstractmethod
    def set_additional_time_before_remove(self, aditional_time_valid_until:int, aditional_time: int) ->None:
        """ Setter for _set_additional_time_before_remove
        Args:
            aditional_time_valid_until: Until when this additional time extantian will be present
            aditional_time: How long is the additional time before object removed
        Return:
            Nome
        """
        self._additional_time_before_remove = {"valid_until": aditional_time_valid_until, "additional_time": aditional_time}
        
    @abstractmethod
    def predict_next_state(self,measurment_time: int, sensor_type:SensorTypes=None) ->None:
        if not self.additional_time_before_remove:
            self._skipped_frames += 1
        try:
            self._kalman_filter_bridge.predict_with_KF(measurment_time)
            # if self._track_id==1 :
            # #     print("P",self.KF._P.diagonal())
            #     print(self._track_id, self.KF._x, measurment_time, sensor_type)
                # print(self.KF.remove)
                # print(self.KF._Xsig_pred)
        except Exception as e:
            raise e
    
    @abstractmethod
    def correct_prediction(self, measurment: AbstractMeasurment) ->None:
        #print(self._track_id,self.KF._x,measurment.measurment_matrix_in_track_cor_system, measurment.sensor_type.name, measurment.measurment_matrix)
        self._hits += 1
        self._skipped_frames = 0
        self._kalman_filter_bridge.update_KF(measurment)
        # if self._track_id==1 :
        #     print(self._track_id, self.KF._x,measurment.measurment_matrix_in_track_cor_system, measurment.measurment_matrix,  measurment.sensor_type)
        # #     print("C",self.KF._P.diagonal())
        #     print("----")
        #     print("---",self.KF._P[2,2])
        # if self._last_measurment_time and self._track_id==1 and measurment.measurment_time - self._last_measurment_time <0: ### Double update result when update made with already pasted timetemp
        #     print("--",self._track_id, self.KF._x,measurment.measurment_matrix_in_track_cor_system, measurment.measurment_matrix, self.KF._last_update_time)
        self._last_measurment_time = measurment.measurment_time
        # if self._track_id==2:
        #     print("after",self.KF._x,measurment.measurment_matrix_in_track_cor_system, measurment.measurment_matrix, self.hits)
