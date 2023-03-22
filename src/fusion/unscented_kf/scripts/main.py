import numpy as np
import cv2
np.set_printoptions(suppress=True)

from utils.measurment.concrete_classes.incoming_measurments import EgoCarOdomMessage, InitialCameraMeasurment, InitialLidarMeasurment, InitialRadarMeasurment
from utils.measurment.concrete_classes.parsed_measurment import ParsedMeasurmentWithGT

#Typing imports
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.tracker.abstract_classes.abstract_object_tracker import AbstractTracker
from utils.movement_compensation.abstract_classes.abstract_movement_compensation import AbstractMovmentCompensation
from utils.common.enums import SensorTypes

from timeit import default_timer as timer
import time

# import cProfile, pstats, io
# def profile(fnc):
    
#     """A decorator that uses cProfile to profile a function"""
    
#     def inner(*args, **kwargs):
        
#         pr = cProfile.Profile()
#         pr.enable()
#         retval = fnc(*args, **kwargs)
#         pr.disable()
#         s = io.StringIO()
#         sortby = 'cumulative'
#         ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#         ps.print_stats()
#         print(s.getvalue())
#         return retval

#     return inner
    
class TrackerRunner():
    def __init__(self, abstract_measurment_type: AbstractMeasurment, configured_sensor_objs: dict,
                 tracker: AbstractTracker, movement_compensetor: AbstractMovmentCompensation=None):
        self._abstract_measurment_type = abstract_measurment_type
        self._configured_sensor_objs = configured_sensor_objs
        self._tracker = tracker
        self._movement_compensetor = movement_compensetor

    def parse_incomming_meaurment(self, sensor_name: str, incomimg_measurment, sensor_type: SensorTypes) -> list:
        return [self._abstract_measurment_type(self._configured_sensor_objs[sensor_name],measurment) for measurment in incomimg_measurment]
    #@profile
    def update_tracker_with_meaurments(self, sensor_type: SensorTypes, sensor_name: str, incomimg_measurment, measurment_time) -> None: #incomimg_measurment can be InitialCameraMeasurment, InitialLidarMeasurment, InitialRadarMeasurment
        self._tracker.update_tracks(self.parse_incomming_meaurment(sensor_name, incomimg_measurment,sensor_type), measurment_time.to_sec(), self._configured_sensor_objs[sensor_name].sensor_properties)

    def update_movement_compensetor(self, odom_message: EgoCarOdomMessage) -> None:
        self._movement_compensetor.update_state(odom_message)



if __name__ == "__main__":
    from sensor_parameters_config import configured_sensor_objs_in2lab_6
    from tracker_parameters_config import movement_compensetor, generate_traker_twizy, generate_traker_in2lab
    test = TrackerRunner(ParsedMeasurmentWithGT, configured_sensor_objs_in2lab_6, generate_traker_in2lab(), movement_compensetor)
    measurment = InitialCameraMeasurment(500,600, "person", 1, 0)
    test.update_tracker_with_meaurments(SensorTypes.camera, "Right_Camera",[measurment],1)

    