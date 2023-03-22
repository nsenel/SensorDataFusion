from utils.custom_exceptions.defined_exceptions import MeasurmentTimeNone, MeasurmentTimeSmaller, PMatrixHasNotPossitiveDefine
from utils.tracker.abstract_classes.abstract_object_tracker import AbstractTracker
from scipy.optimize import linear_sum_assignment

#Typing imports
from utils.tracker.abstract_classes.abstrack_cost_calculator import AbstractCostCalculator
from utils.object_factories.abstract_classes.abstract_track_factory import AbstractTrackFactory
from utils.common.enums import SensorTypes
import numpy as np

class BasicTracker(AbstractTracker):
    """It might get out dated TODO test it"""
    def __init__(self, track_generator: AbstractTrackFactory, cost_calculator: AbstractCostCalculator,
                 dist_thresh: int, max_frames_to_skip: int):
        super().__init__(track_generator=track_generator, cost_calculator=cost_calculator)
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip

    def add_new_tracks(self, un_assigned_detects_measurments: list) ->None:
        """
        Adds new tracks to _tracked_objs
        Args:
            un_assigned_detects_measurments: List of unassign detected objects type AbstractMeasurment
        Return:
            None
        """
        for detection in un_assigned_detects_measurments:
            self.tracked_object_list.append(self.track_generator.generate_track_obj(measurment=detection))

    def calculate_cost(self, detections:list, measurment_time) -> np.ndarray:
        """
        Calculates distance cost between new detections and exist track objects
        Args:
            detections: List of detected objects type AbstractMeasurment
        Return:
            ndarray
        """
        try:
            return self.cost_calculator.calculate_cost(self.tracked_object_list, detections, measurment_time)
        except Exception as e:
            raise e
    
    def assign_detections_to_trackers(self,cost_matrix):
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
        assignment = [-1]*len(self.tracked_object_list) # Initiate list with -1's means that all tracks are unmatchted tracks
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i] # Update the list elements which corresponds matched detection index in detections list

        # Identify tracks with no assignment, if any
        for i in range(len(assignment)):
            if (assignment[i] != -1):
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track
                if (cost_matrix[i][assignment[i]] > self.dist_thresh):
                    #print("remove assigmnet cost is :",cost_matrix[i][assignment[i]])
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
    
    def update_tracks_filter(self, assigned_tracks: list, un_assigned_tracks: list, detections:list):
        """
        Updates KalmanFilter states, hits and skipped_frames depending on max_frames_to_skip
        treshhold removes trace object from list
        Args:
            assigned_tracks:Matched detections as list-(Holds tuple indexes.(Those indexes are matching elements in self.traces and detection list))
            un_assigned_tracks: Unmatchted tracks as list-(Holds indexes.(This indexes correspond track list elements))
            detections: List of detected objects AbstractMeasurment
        Return:
            None
        """
        # Update KalmanFilter state, hits and skipped_frames
        for track_idx, detection_idx in assigned_tracks:
            self.tracked_object_list[track_idx].correct_prediction(detections[detection_idx])

        delete_list_idx = []
        for idx in un_assigned_tracks:
            #If tracks are not detected more than treshhold(max_frames_to_skip) remove them
            if self.tracked_object_list[idx].skipped_frames > self.max_frames_to_skip:
                delete_list_idx.append(idx)

        for idx,track_idx in enumerate(delete_list_idx):
            # If there is more than one value to delete
            # Index value should be reduce because track_idx
            # Indef number when all items in the list
            self.remove_track_obj_by_index(track_idx-idx)

    def predict_new_state(self, measurment_time, sensor_properties, track_obj_list):
        pass # will be implemented
    def update_tracks(self, detections, measurment_time) -> None:
        try:
            if len(detections)==0:
                for track in self.tracked_object_list:
                    track.predict_next_state(measurment_time)
                self.update_tracks_filter(assigned_tracks=[], un_assigned_tracks=[idx for idx in range(len(self.tracked_object_list))], detections=[])
            else:
                cost_matrix = self.calculate_cost(detections, measurment_time)
                assigned_tracks, un_assigned_tracks, un_assigned_detects = self.assign_detections_to_trackers(cost_matrix)
                un_assigned_detects_measurments = [detection for idx,detection in enumerate(detections) if idx in un_assigned_detects]
                self.add_new_tracks(un_assigned_detects_measurments)
                self.update_tracks_filter(assigned_tracks, un_assigned_tracks, detections)
        except Exception as e:
            if isinstance(e,MeasurmentTimeSmaller):
                print("Measurment time is older ??")
                self.update_tracks_filter(assigned_tracks=[], un_assigned_tracks=[idx for idx in range(len(self.tracked_object_list))], detections=[]) ## Made it for rosbag on repeat after max_skiped frame you will not get error
                pass
            elif isinstance(e,MeasurmentTimeNone):
                print("Measurment time is not given cant update state ...")
            elif isinstance(e,PMatrixHasNotPossitiveDefine):
                ## Remove object with bad P matrix
                for idx, track_obj in enumerate(self.tracked_object_list):
                    if track_obj.track_id == e.track_object_id:
                        self.remove_track_obj_by_index(idx)
                        break
            else:
                print("unknown error !!!")
                raise e
