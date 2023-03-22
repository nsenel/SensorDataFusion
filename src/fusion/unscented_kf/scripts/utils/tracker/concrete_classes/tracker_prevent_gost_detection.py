import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.common.enums import ObjectTypes, SensorTypes
from utils.custom_exceptions.defined_exceptions import MeasurmentTimeNone, MeasurmentTimeSmaller, PMatrixHasNotPossitiveDefine
from utils.tracker.abstract_classes.abstract_object_tracker import AbstractTracker

#Typing imports
from utils.tracker.abstract_classes.abstrack_cost_calculator import AbstractCostCalculator
from utils.object_factories.abstract_classes.abstract_track_factory import AbstractTrackFactory
from utils.track_object.abstract_classes.abstract_tracked_object import AbstractTrack

class DoubleTrackTracker(AbstractTracker):
    """ Holds 2 seperate track object list one is waiting track list and one valid object list.
        Track objects first goes to waitobject list and when they fulfill defined creteria they go to valib object list
    """
    def __init__(self, track_generator: AbstractTrackFactory, cost_calculator: AbstractCostCalculator,
                 dist_thresh: int, time_treshhold_to_remove: int, waiting_list_time_treshhold_to_remove: int,
                 waiting_list_max_hit:int):
        super().__init__(track_generator=track_generator, cost_calculator=cost_calculator)
        self.dist_thresh = dist_thresh
        self.time_treshhold_to_remove = time_treshhold_to_remove
        self.waiting_list_time_treshhold_to_remove = waiting_list_time_treshhold_to_remove
        self.waiting_list_max_hit = waiting_list_max_hit
        self._waiting_list = []
    
    def what_i_am_doing(self,new_tracks,track_obj_list):### Until map is implemented this function try to prevent duplication of same object already been tracked !
        for new_track in new_tracks:
            x = new_track.measurment_matrix_in_cost_function_cor_system[0]
            good_one = True
            for know_track in track_obj_list:
                if abs(know_track.prediction[0]-x)<2:
                    good_one = False
            if good_one:
                self._waiting_list.append(self.track_generator.generate_track_obj(measurment=new_track))


    def update_waiting_list(self, detections: list, measurment_time: int, measurment_sensor_properties):
        if len(self._waiting_list) == 0:
            self.what_i_am_doing(detections, self._tracked_objs)
        else:
            in_FOV_tracks = [track for track in self._waiting_list if measurment_sensor_properties.is_object_in_FOV(np.degrees(np.arctan2(track.prediction[1],track.prediction[0])))]
            cost_matrix = self.calculate_cost(detections, in_FOV_tracks)
            _assigned_tracks, _un_assigned_tracks, _un_assigned_detects = self.assign_detections_to_trackers(cost_matrix, in_FOV_tracks)
            self.update_tracks_filter(_assigned_tracks, _un_assigned_tracks, detections, in_FOV_tracks, measurment_time=measurment_time, time_treshhold_to_remove=self.waiting_list_time_treshhold_to_remove)

            if len(_un_assigned_detects)>0:
                self.what_i_am_doing([detection for idx,detection in enumerate(detections) if idx in _un_assigned_detects], self._tracked_objs+self._waiting_list)

    def add_new_tracks(self, un_assigned_detects_measurments: list, measurment_time: int, measurment_sensor_properties) ->None:
        """
        Adds new tracks to _tracked_objs
        Args:
            un_assigned_detects_measurments: List of unassign detected objects type AbstractMeasurment
        Return:
            None
        """
        self.update_waiting_list(un_assigned_detects_measurments, measurment_time, measurment_sensor_properties)
        idxs_obj = []
        for waiting_object in self._waiting_list:
            add_to_track_list = True
            if waiting_object.valid_object or (waiting_object.hits > self.waiting_list_max_hit and waiting_object.track_object_type != None):
                if waiting_object.track_object_type != ObjectTypes.person:
                    all_tracks = np.array([[i.KF._x[0],i.KF._x[1]] for i in self.tracked_object_list if i.track_object_type == waiting_object.track_object_type]).T
                    new_track_loc = np.array([waiting_object.KF._x[0],waiting_object.KF._x[1]]).reshape(2,1)
                    euclidean_dist_to_tracked_objs = np.linalg.norm(new_track_loc-all_tracks, axis=0.)
                    min_=100 if not len(euclidean_dist_to_tracked_objs) else np.min(euclidean_dist_to_tracked_objs)
                    if min_<3:###If new track to close to any known track do not add it
                        add_to_track_list = False
                        #idxs_obj.append(id(waiting_object))
                if add_to_track_list:
                    waiting_object.set_valid_obj_state(True)
                    self.tracked_object_list.append(waiting_object)
                    idxs_obj.append(id(waiting_object)) ###can be same track_id since there might me different object types with same id
        
        if len(idxs_obj)>0:
            self._waiting_list[:] = [x for x in self._waiting_list if id(x) not in idxs_obj] # [:] makes inplace change in list doest over write it with new list
        

    def calculate_cost(self, detections:list, tracked_objects: list) -> np.ndarray:
        """
        Calculates distance cost between new detections and exist track objects
        Args:
            detections: List of detected objects type AbstractMeasurment
        Return:
            ndarray
        """
        try:
            return self.cost_calculator.calculate_cost(tracked_objects, detections)
        except Exception as e:
            raise e
    
    def assign_detections_to_trackers(self, cost_matrix: np.array, tracked_objects: list):
        """
        Using Hungarian Algorithm determine the detected tracks, unmatchted tracks
        and unmatched detections with using cost matrix.
        Args:
            cost_matrix: Cost matrix(len(self.tracks)xlen(detections))
            calculated by predicted state vektor and new measurments
        Return:
            assigned_tracks: Matched detections as list-(Holds tuple indexes.(Those indexes are matching elements in self.traces and detection list))
            un_assigned_tracks: Unmatchted tracks as list-(Holds indexes.(This indexes correspond track list elements))
            un_assigned_detects: Unmatched detections as list-(Holds indexes. (This indexes correspond detection list elements) )
        """
        assignment = [-1]*len(tracked_objects) # Initiate list with -1's means that all tracks are unmatchted tracks
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i] # Update the list elements which corresponds matched detection index in detections list

        # Identify tracks with no assignment, if any
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost_matrix[i][assignment[i]] > self.dist_thresh):
                    assignment[i] = -1
                pass
        # Look for not assigned detections
        un_assigned_detects = [] # Holds index. (This indexes correspond detection list elements) 
        for i in range(cost_matrix.shape[1]):
                if i not in assignment:
                    un_assigned_detects.append(i)

        assigned_tracks = [(track_idx,detection_idx) for track_idx,detection_idx in enumerate(assignment) if detection_idx!=-1]
        un_assigned_tracks = [idx for idx in range(len(assignment)) if assignment[idx]==-1]

        return assigned_tracks, un_assigned_tracks, un_assigned_detects
    
    def update_tracks_filter(self, assigned_tracks: list, un_assigned_tracks: list, detections:list, tracked_objects: list, measurment_time:int, time_treshhold_to_remove: int):
        """
        Updates KalmanFilter states, hits and skipped_frames depending on time_treshhold_to_remove
        treshhold removes trace object from list
        Args:
            assigned_tracks:Matched detections as list-(Holds tuple indexes.(Those indexes are matching elements in self.traces and detection list))
            un_assigned_tracks: Unmatchted tracks as list-(Holds indexes.(This indexes correspond track list elements))
            detections: List of detected objects AbstractMeasurment
        Return:
            None
        """
        error_occur_tracks = []
        for track_idx, detection_idx in assigned_tracks:
            try:
                tracked_objects[track_idx].correct_prediction(detections[detection_idx])
            except Exception as e:
                error_occur_tracks.append((e,tracked_objects[track_idx]))###If you delete them here than you are braking the list dont !!!!!
            if tracked_objects[track_idx].track_object_type == None and detections[detection_idx].obj_name != None:
                # Object first created with default tracking settings and here we reset the object to have correct parameters ...(Keeps state vector and p matrix values)
                for idx in range(len(self._waiting_list)): ### I hate it but list not _waiting_list it is object which are in FOV. therefore tracked_objects[track_idx]= BLABLA will not change real object in the wait_object_list!!
                    if self._waiting_list[idx] is tracked_objects[track_idx]:
                        self._waiting_list[idx]=self.track_generator.reset_tracking_settings(tracked_objects[track_idx], detections[detection_idx].obj_name)
        
        for idx in un_assigned_tracks:
            if ((measurment_time-tracked_objects[idx].last_measurment_time) > time_treshhold_to_remove + tracked_objects[idx].additional_time_before_remove or
                tracked_objects[idx].skipped_frames/2>tracked_objects[idx].hits):
                tracked_objects[idx].set_remove_state(True)
                tracked_objects[idx].KF.roll_back_prediction()
        
        for error in error_occur_tracks:
            self.exception_handler(*error)
    
    def predict_new_state(self, measurment_time, sensor_properties, track_obj_list): #Predict new state over kf only should call here!
        remove_idx = []
        for idx,track in enumerate(track_obj_list):
            if track.remove_me:
                remove_idx.append(idx)
            else:
                if sensor_properties.is_object_in_FOV(np.degrees(np.arctan2(track.prediction[1],track.prediction[0]))):
                    try:
                        track.predict_next_state(measurment_time, sensor_properties.sensor_type)
                    except Exception as e:
                        self.exception_handler(e,track)
        
        for idx,track_idx in enumerate(remove_idx):
            # If there is more than one value to delete
            # Index value should be reduce because track_idx
            # Indef number when all items in the list
            self.remove_track_obj_by_index(track_idx-idx, track_obj_list)
        return track_obj_list
                

    def update_tracks(self, detections, measurment_time, measurment_sensor_properties) -> None:
        try:
            in_FOV_tracks = self.predict_new_state(measurment_time,measurment_sensor_properties, self.tracked_object_list)
            in_FOV_tracks_waiting =self.predict_new_state(measurment_time,measurment_sensor_properties, self._waiting_list)
            if len(detections)==0:
                    self.update_tracks_filter(assigned_tracks=[], un_assigned_tracks=[idx for idx in range(len(in_FOV_tracks))], detections=[],
                                          tracked_objects=self.tracked_object_list, measurment_time=measurment_time, time_treshhold_to_remove=self.time_treshhold_to_remove)
                    self.update_tracks_filter(assigned_tracks=[], un_assigned_tracks=[idx for idx in range(len(in_FOV_tracks_waiting))], detections=[],
                                            tracked_objects=self._waiting_list, measurment_time=measurment_time, time_treshhold_to_remove=self.time_treshhold_to_remove)

            else:
                cost_matrix = self.calculate_cost(detections, in_FOV_tracks)
                assigned_tracks, un_assigned_tracks, un_assigned_detects = self.assign_detections_to_trackers(cost_matrix, in_FOV_tracks)
                self.update_tracks_filter(assigned_tracks, un_assigned_tracks, detections, in_FOV_tracks, measurment_time=measurment_time, time_treshhold_to_remove=self.time_treshhold_to_remove)
                un_assigned_detects_measurments = [detection for idx,detection in enumerate(detections) if idx in un_assigned_detects]
                self.add_new_tracks(un_assigned_detects_measurments, measurment_time, measurment_sensor_properties)
                
        except Exception as e:
            self.exception_handler(e)
        
    def exception_handler(self, exception_type:Exception, track_obj:AbstractTrack=None):
        if track_obj:
            track_object_id = track_obj.track_id
            obj_memory_address = id(track_obj)
            if isinstance(exception_type,MeasurmentTimeSmaller):
                if track_object_id >0: # print only if object has clasification
                    print(f"Measurment time is older exception accoured going to remove object with following id{track_object_id}..")
                ## Remove object with bad last update time value
                for idx, track_obj in enumerate(self.tracked_object_list):
                    if id(track_obj) == obj_memory_address:
                        print("removed...")
                        self.remove_track_obj_by_index(idx, self.tracked_object_list)
                        break
                for idx, track_obj in enumerate(self._waiting_list):
                    if id(track_obj) == obj_memory_address:
                        print("removed...")
                        self.remove_track_obj_by_index(idx, self._waiting_list)
                        break
            
            elif isinstance(exception_type,PMatrixHasNotPossitiveDefine):
                if track_object_id >0: # print only if object has clasification
                    print(f"PMatrixHasNotPossitiveDefine exception accoured going to remove object with following id{track_object_id}..")
                ## Remove object with bad P matrix
                for idx, track_obj in enumerate(self.tracked_object_list):
                    if id(track_obj) == obj_memory_address:
                        self.remove_track_obj_by_index(idx, self.tracked_object_list)
                        break
                for idx, track_obj in enumerate(self._waiting_list):
                    if id(track_obj) == obj_memory_address:
                        self.remove_track_obj_by_index(idx, self._waiting_list)
                        break
            else:
                print("unknown error fusion node terminated !!!")
                raise exception_type
        elif isinstance(exception_type,MeasurmentTimeNone):
            print("Measurment time is not given cant update state ...")
        else:
            print("unknown error fusion node terminated !!!")
            raise exception_type
