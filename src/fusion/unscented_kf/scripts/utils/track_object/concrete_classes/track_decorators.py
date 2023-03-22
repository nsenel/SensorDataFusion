
"""
    @author: Numan Senel
    @email: Numan.Senel@thi.de
"""
# Import python libraries
from collections import deque

import numpy as np
from utils.common.enums import ObjectTypes, SensorTypes
from utils.movement_compensation.abstract_classes.abstract_movement_compensation import AbstractMovmentCompensation

# Import dependent classes
from utils.track_object.abstract_classes.tracked_obj_decorator import TrackDecorator
from utils.track_object.concrete_classes.track_visualization import TrackVisualization
from utils.track_object.concrete_classes.auto_calibration import Calibrator, CalibrationDataCollector

#Typing imports
from utils.track_object.abstract_classes.abstract_tracked_object import AbstractTrack
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment



class AddOnGroundTruth(TrackDecorator):
    """ When there is GT ID assign to measurment class will be initiated with this ID and assign ID.
        When assign ID changes(Remove object from track, id swap etc.) it will give warning.
    """
    def __init__(self, track_object):
        super().__init__(track_object)
        self._ground_truth_id = None

    def correct_prediction(self, measurment: AbstractMeasurment):
        if measurment.gt_id !=-1 and self._ground_truth_id == None: ##measurment.gt_id !=-1 for waymo there is no id for camera only for the lidar
            self._ground_truth_id = measurment.gt_id
        if measurment.gt_id != -1 and measurment.gt_id != self._ground_truth_id:
            print("Correct prediction in coming meaurment gt_id:%d expected:%d wrong assigment !!!!!!!!!!!!!"%(measurment.gt_id, self._ground_truth_id))
            print("Over writed obj gt_id with current match gt_id")
            self._ground_truth_id = measurment.gt_id
        self._track.correct_prediction(measurment)

class AddOnTrace(TrackDecorator):
    """ Keep history about track object trajectory. """
    def __init__(self,track_object: AbstractTrack, max_trace_length: int):
        super().__init__(track_object)
        self._trace = deque(maxlen=max_trace_length)
    
    @property
    def trace(self): # -> list[np.ndarray]
        """ Getter for _trace
        Args:
            None
        Return:
            trace path 
        """
        return self._trace
    
    def correct_prediction(self, measurment: AbstractMeasurment) -> None:
        self._track.correct_prediction(measurment)
        self._trace.append(self.prediction)

class AddOnMovementCompensation(TrackDecorator):
    """ Modifies track object state depending on ego motion """
    def __init__(self,track_object: AbstractTrack, movement_compensation_method: AbstractMovmentCompensation):
        super().__init__(track_object)
        self._mv_method = movement_compensation_method
    
    def predict_next_state(self, measurment_time, sensor_type:SensorTypes=None) -> None:
        movement_in_x, movement_in_y= self._mv_method.get_movement_compensation(track_x=self.KF.x_postion, track_y=self.KF.y_postion,
                                                                                measurment_time=measurment_time,
                                                                                last_update_time=self._track.KF.last_update_time)
        if movement_in_x != 0 or movement_in_y != 0:
            # print(f"x goes from {self.KF.x_postion} to {self.KF.x_postion+movement_in_x} diff: {movement_in_x}")
            # print(f"y goes from {self.KF.y_postion} to {self.KF.y_postion+movement_in_y} diff: {movement_in_y}")
            if self._track.skipped_frames <3:
                self.KF.x_postion = self.KF.x_postion+movement_in_x
                self.KF.y_postion = self.KF.y_postion+movement_in_y
            else:
                self.KF.x_postion = (self.KF.x_postion+movement_in_x,0.00001)##### if an object doesnt seen for long time and update its location only with car uncurtiny will be so big that sigma points goes crazy
                self.KF.y_postion = (self.KF.y_postion+movement_in_y,0.00001)
        self._track.predict_next_state(measurment_time, sensor_type=sensor_type)

class AddOnTrackVisualization(TrackDecorator):
    """ Data collector for ploting track and measurment data"""
    def __init__(self, track_object: AbstractTrack, live_plot=False, plot_data_on_delete=False):
        super().__init__(track_object)
        self._data_visualazation = TrackVisualization(live_plot=live_plot, plot_data_on_delete=plot_data_on_delete)
    
    def correct_prediction(self, measurment: AbstractMeasurment) -> None:
        self._track.correct_prediction(measurment)
        self._data_visualazation.add_measurment_data(measurment)
        self._data_visualazation.add_prediction(self._track.KF.x)

class AddOnCalibration(TrackDecorator):
    """Generate data collecter object and on delete registers all the collected data to calibrater class.
       To calculate calibration valeus Calibration.calculate should be called in some other place this class only collect data. 
    """
    def __init__(self, track_object: AbstractTrack, calibrator:Calibrator, target_sensor: SensorTypes, sensor_in_track_cor: SensorTypes):
        super().__init__(track_object)
        self._calibration_data_collector = CalibrationDataCollector(target_sensor, sensor_in_track_cor)
        self._calibrator = calibrator
    
    def correct_prediction(self, measurment: AbstractMeasurment) -> None:
        self._track.correct_prediction(measurment)
        if self._calibrator.target_sensor_id == measurment.sensor_id:
            self._calibration_data_collector.add_measurment_data(self.detected_by_sensors)
    
    def __del__(self):
        if self.hits>10 and len(self._calibration_data_collector.sensor1)>10:
            self._calibrator.sensor1 += self._calibration_data_collector.sensor1
            self._calibrator.sensor2 += self._calibration_data_collector.sensor2

class AddOnSensorDetectInfo(TrackDecorator):
    """ 
    Holds information about which sensor or sensors detect tracked object
    When new object list arrives from object all tracks will have false detection for that sensor type
    if there is match between object list and track object sensor detection will turn to true state.
    """
    def __init__(self, track_object: AbstractTrack):
        super().__init__(track_object)
    
    def correct_prediction(self, measurment: AbstractMeasurment) -> None:
        self.detected_by_sensors[measurment.sensor_type]["detected_in_last"] = True
        self.detected_by_sensors[measurment.sensor_type]["last_measurment"] = measurment
        self.detected_by_sensors[measurment.sensor_type]["detected_cnt"] += 1
        if self.track_object_type and self.valid_object == False:
            if self.detected_by_sensors[SensorTypes.camera]["detected_cnt"]<2 and self._track.hits>3:## Force object to have atleast two image detection to count as valid obj
                self._track.set_number_of_hits(self._track.hits-1)
        self._track.correct_prediction(measurment)
        
    def predict_next_state(self, measurment_time, sensor_type:SensorTypes) -> None:
        self.detected_by_sensors[sensor_type]["detected_in_last"] = False
        self._track.predict_next_state(measurment_time, sensor_type=sensor_type)

class AddOnMap(TrackDecorator):
    """ Register and update position of track object in a virtual grid map. """
    def __init__(self, track_object: AbstractTrack, object_map):
        super().__init__(track_object)
        self.object_map = object_map
        self.obj_in_map_boundry = True
        if not self.object_map.register_obj(self.prediction[0], self.prediction[1], self._track):
            self.set_remove_state(self_remove=True)
            self.obj_in_map_boundry = False
    
    def correct_prediction(self, measurment: AbstractMeasurment) -> None:
        if self.obj_in_map_boundry:
            prev_x, prev_y = self.prediction[0], self.prediction[1]
            self._track.correct_prediction(measurment)
            self.obj_in_map_boundry, success = self.object_map.update_obj_location(prev_x, prev_y, self.prediction[0], self.prediction[1], self.track_id)
            #self.control_clean_cell()
            if not success:
                print("AddOnSensorDetectInfo correct_prediction faild track_id: ", self.track_id)
        else:
            self._track.correct_prediction(measurment)

    def predict_next_state(self, measurment_time, sensor_type:SensorTypes) -> None:
        if self.obj_in_map_boundry:
            prev_x, prev_y = self.prediction[0], self.prediction[1]
            self._track.predict_next_state(measurment_time, sensor_type=sensor_type)
            self.obj_in_map_boundry, success = self.object_map.update_obj_location(prev_x, prev_y, self.prediction[0], self.prediction[1], self.track_id)
            if not success:
                print("AddOnSensorDetectInfo predict_next_state failed track_id: ", self.track_id)
        else:
            self._track.predict_next_state(measurment_time, sensor_type=sensor_type)

    def control_clean_cell(self):
        """ Check accopied cells if there is more than one track object and remove,modifie etc. depending on defined rules. """
        if self.obj_in_map_boundry:
            cell_objs = self.object_map.cell_info(self.prediction[0], self.prediction[1])
            if cell_objs._object_cnt > 1:
                for obj in cell_objs._objects_info.values():
                    if not obj.track_object_type:
                        obj.set_remove_state(self_remove=True)
    def __del__(self):
        if self.obj_in_map_boundry: # it obj out of the obj_in_map_boundry it already deleted or never registered
            self.object_map.remove_obj(self.prediction[0], self.prediction[1], self.track_id)

        
