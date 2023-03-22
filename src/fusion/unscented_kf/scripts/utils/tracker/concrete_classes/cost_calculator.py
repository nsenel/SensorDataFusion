from utils.common.enums import SensorTypes
from utils.custom_exceptions.defined_exceptions import PMatrixHasNotPossitiveDefine, MeasurmentTimeSmaller
from utils.tracker.abstract_classes.abstrack_cost_calculator import AbstractCostCalculator
import numpy as np

class NormCostCalculator(AbstractCostCalculator):
    def calculate_cost(self, tracks, detections, measurment_time) -> np.ndarray:
        """Calculate cost using sum of square distance between predicted object location vs detected centroids
        Args:
            tracks: List of tracks type AbstractTrack
            detections: List of detected objects type AbstractMeasurment
        Return:
            Cost matrix
        """
        N = len(tracks) # Rows
        M = len(detections) # Columns

        try:
            for track in tracks:
                track.predict_next_state(measurment_time) #Predict new object location with using kalman filter.
            
            cost = np.zeros(shape=(N, M)) # Cost matrix
            for i in range(len(tracks)):
                for j in range(len(detections)):
                    cost[i][j] = np.linalg.norm(tracks[i].prediction - detections[j].measurment_matrix_in_cost_function_cor_system)
            return cost
        except Exception as e:
            if isinstance(e,PMatrixHasNotPossitiveDefine) and e.track_object_id==-1: ## -1 means is not defined
                print("PMatrixHasNotPossitiveDefine error ??")
                e.track_object_id = track.track_id
                e.obj_memory_address = id(track)
            raise e

class NormCostCalculatorSeperateByObjectType(AbstractCostCalculator):
    def calculate_cost(self, tracks, detections, measurment_time) ->np.ndarray:
        """Calculate cost using sum of square distance between predicted object location vs detected centroids
        Args:
            tracks: List of tracks type AbstractTrack
            detections: List of detected objects type AbstractMeasurment
        Return:
            Cost matrix
        """
        N = len(tracks) # Rows
        M = len(detections) # Columns
        try:
            for track in tracks:
                track.predict_next_state(measurment_time) #Predict new object location with using kalman filter.
            
            cost = np.zeros(shape=(N, M)) # Cost matrix
            for i in range(len(tracks)):
                for j in range(len(detections)):
                    if tracks[i].track_object_type != detections[j].obj_name:
                        cost[i][j] = 10000 ### TODO I dont like constant here !!!
                    else:
                        cost[i][j] = np.linalg.norm(tracks[i].prediction - detections[j].measurment_matrix_in_cost_function_cor_system)

            return cost
        except Exception as e:
            if isinstance(e,PMatrixHasNotPossitiveDefine) and e.track_object_id==-1: ## -1 means is not defined
                print("PMatrixHasNotPossitiveDefine error ??")
                e.track_object_id = track.track_id
                e.obj_memory_address = id(track)
            raise e

class NormCostCalculatorSeperateByObjectExeptUnknown(AbstractCostCalculator):
    def calculate_cost(self, tracks, detections) ->np.ndarray:
        """Calculate cost using sum of square distance between predicted object location vs detected centroids
        Args:
            tracks: List of tracks type AbstractTrack
            detections: List of detected objects type AbstractMeasurment
        Return:
            Cost matrix
        """
        N = len(tracks) # Rows
        M = len(detections) # Columns
        
        try:
            cost = np.zeros(shape=(N, M)) # Cost matrix
            for i in range(len(tracks)):
                for j in range(len(detections)):
                    if detections[j].obj_name != None and tracks[i].track_object_type!=None and tracks[i].track_object_type != detections[j].obj_name: ### This is the correct version if you want to overwrite None track object with new class name 
                        cost[i][j] = 10000 ### TODO I dont like constant here !!!
                    else:
                        distance_diff = tracks[i].prediction - detections[j].measurment_matrix_in_cost_function_cor_system
                        if detections[j].sensor_type == SensorTypes.radar_cartesian:
                            speed_factor = min(1, (0.1*tracks[i].hits))
                            distance_diff = np.array([distance_diff[0],distance_diff[1],(detections[j].measurment_matrix[2]-tracks[i].KF._x[2])*speed_factor])
                            if (detections[j].measurment_matrix[2]<0 and tracks[i].KF._x[2]<0) or (detections[j].measurment_matrix[2]>0 and tracks[i].KF._x[2]>0):
                                cost[i][j] = 10000 if abs(distance_diff[1])>3 else np.linalg.norm(distance_diff)
                            else:
                                cost[i][j] = 10000
                        else:
                            if detections[j].sensor_type == SensorTypes.camera:
                                #### In long distance camera meaurment not very releable there for temping to create new object maybe this prevents it.
                                distance_to_mast_depth = abs(detections[j].measurment_matrix_in_cost_function_cor_system[0])
                                if distance_to_mast_depth>40 and distance_to_mast_depth<70:
                                    distance_diff[0]*=0.8
                                elif distance_to_mast_depth>70 and distance_to_mast_depth<100:
                                    distance_diff[0]*=0.7 
                                elif distance_to_mast_depth>100:
                                    distance_diff[0]*=0.4
                            cost[i][j] = 1000 if abs(distance_diff[1])>2 else np.linalg.norm(distance_diff)
            return cost
        except Exception as e:
            raise e

class NormCostCalculatorSeperateByObjectExeptUnknownMAP(AbstractCostCalculator):
    """
    Utilize track object locations to narow down search space in new detections.
    """
    def __init__(self, map) -> None:
        self.map = map
    
    def surrounding_keys(self, key_loc):
        row, column = key_loc[0], key_loc[1]
        return ((row-1,column-1), (row-1,column), (row-1,column+1),
                (row,column-1),   (row,column),   (row,column+1),
                (row+1,column-1), (row+1,column), (row+1,column+1))
    
    def debug(self,un_assigned_tracks, un_assigned_detects, assigned_tracks, assigned_detects):
        print("Not matched")
        for track in un_assigned_tracks:
            print("track: ",track.track_id, track.KF._x)
        for detect in un_assigned_detects:
            print(detect.sensor_type, detect.measurment_matrix_in_cost_function_cor_system, detect.measurment_matrix)
        print(self.calculate_cost(un_assigned_tracks,un_assigned_detects))
        print("Matched")
        for t,d in zip(assigned_tracks,assigned_detects):
            print(t.track_id, t.KF._x, d.measurment_matrix_in_cost_function_cor_system, d.measurment_matrix, d.sensor_type)
        print("----")

    def calculate_cost(self, tracks, detections) ->np.ndarray:
        """Calculate cost using sum of square distance between predicted object location vs detected centroids
        Args:
            tracks: List of tracks type AbstractTrack
            detections: List of detected objects type AbstractMeasurment
        Return:
            Cost matrix
        """
        N = len(tracks) # Rows
        M = len(detections) # Columns
        ### find detections map location
        detections_in_map = {}
        for idx, detection in enumerate(detections):
            key_exist, loc =self.map.get_key(detection.measurment_matrix_in_cost_function_cor_system[0], detection.measurment_matrix_in_cost_function_cor_system[1])
            if key_exist:
                detections_in_map[loc] = idx
        try:
            cost = np.ones(shape=(N, M))*1000 # Cost matrix
            for i in range(len(tracks)):
                _, loc = self.map.get_key(tracks[i].prediction[0], tracks[i].prediction[1])
                possible_key_areas = self.surrounding_keys(loc)
                for j in [detections_in_map[map_key] for map_key in detections_in_map.keys() if map_key in possible_key_areas]:
                    if detections[j].obj_name != None and tracks[i].track_object_type!=None and tracks[i].track_object_type != detections[j].obj_name: ### This is the correct version if you want to overwrite None track object with new class name 
                        continue
                    else:
                        distance_diff = tracks[i].prediction - detections[j].measurment_matrix_in_cost_function_cor_system
                        if detections[j].sensor_type == SensorTypes.radar_cartesian:
                            speed_factor = min(1, (0.1*tracks[i].hits))
                            distance_diff = np.array([distance_diff[0],distance_diff[1],(detections[j].measurment_matrix[2]-tracks[i].KF._x[2])*speed_factor])
                            if (detections[j].measurment_matrix[2]<0 and tracks[i].KF._x[2]<0) or (detections[j].measurment_matrix[2]>0 and tracks[i].KF._x[2]>0):
                                cost[i][j] = cost[i][j] if abs(distance_diff[1])>3 else np.linalg.norm(distance_diff)
                        else:
                            if detections[j].sensor_type == SensorTypes.camera:
                                #### In long distance camera meaurment not very releable there for temping to create new object maybe this prevents it.
                                distance_to_mast_depth = abs(detections[j].measurment_matrix_in_cost_function_cor_system[0])
                                if distance_to_mast_depth>40 and distance_to_mast_depth<70:
                                    distance_diff[0]*=0.8
                                elif distance_to_mast_depth>70 and distance_to_mast_depth<100:
                                    distance_diff[0]*=0.7 
                                elif distance_to_mast_depth>100:
                                    distance_diff[0]*=0.4
                            cost[i][j] = cost[i][j] if abs(distance_diff[1])>2 else np.linalg.norm(distance_diff)
            return cost
        except Exception as e:
            raise e
