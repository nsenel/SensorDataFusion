from utils.object_factories.abstract_classes.abstract_track_factory import AbstractTrackFactory

#Typing exports
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.parameters.concrete_classes.tracker_parameters import TrackerParameters
from utils.track_object.abstract_classes.abstract_tracked_object import AbstractTrack

class TrackKFObjectFactory(AbstractTrackFactory):
    def __init__(self, track_kf_object_factory_settings_default: TrackerParameters, generate_id_from_dict: dict=None,
                object_type_based_trackKF_settings: dict = {}) -> None:
        super().__init__(generate_id_from_dict)
        self.default_track_params = track_kf_object_factory_settings_default
        self.obj_track_settings = object_type_based_trackKF_settings

    def generate_track_obj(self, measurment: AbstractMeasurment) -> AbstractTrack:
        if measurment.obj_name in self.obj_track_settings: #### Generate track objects with specified settings if object is known other case use default parameters
            track_params: TrackerParameters = self.obj_track_settings[measurment.obj_name]
        else:
            track_params: TrackerParameters = self.default_track_params
        kf_bridge_obj = track_params.kf_bridge.generate_kf_obj_bridge(measurment)
        track_obj = track_params.tracking_strategy(track_id=self.generate_new_track_id(measurment.obj_name), kalman_filter_bridge=kf_bridge_obj,
                                                                                       track_object_type=measurment.obj_name, initial_measurment=measurment)
        for add_on_cls, params in track_params.track_add_ons.items():
            track_obj = add_on_cls(track_obj, **params)
        return track_obj
    
    def reset_tracking_settings(self, tracked_obj:AbstractTrack, object_type: str):
        if not isinstance(tracked_obj, AbstractTrack):
            print("tracked_obj type is not AbstractTrack return None...")
            return tracked_obj
        elif not object_type in self.obj_track_settings:
            print("object_type is not know returns without change...")
            return tracked_obj
        #print("reset_tracking_settings called for track object id:", tracked_obj.track_id)
        track_params: TrackerParameters = self.obj_track_settings[object_type]
        kf_bridge_obj = track_params.kf_bridge.reset_kf_obj_bridge(tracked_obj)
        track_obj = track_params.tracking_strategy(track_id=self.generate_new_track_id(object_type), kalman_filter_bridge=kf_bridge_obj, track_object_type=object_type)
        for add_on_cls, params in track_params.track_add_ons.items():
            track_obj = add_on_cls(track_obj, **params)
        #print("reset_tracking_settings complete new track object id:", track_obj.track_id)

        return track_obj
        
        
class TrackKFOObjectTypeBasedFactory(AbstractTrackFactory):
    def __init__(self,  object_type_based_trackKF_settings: dict, generate_id_from_dict: dict= None) -> None: #object_type_based_trackKF_settings: dict: {"obj_name":TrackerParameters}
        super().__init__(generate_id_from_dict)
        self.obj_track_settings = object_type_based_trackKF_settings

    def generate_track_obj(self, measurment: AbstractMeasurment) -> AbstractTrack:
        track_params: TrackerParameters = self.obj_track_settings[measurment.obj_name]
        kf_bridge_obj = track_params.kf_bridge.generate_kf_obj_bridge(initial_measurment=measurment)

        track_obj = track_params.tracking_strategy(track_id=self.generate_new_track_id(measurment.obj_name), kalman_filter_bridge=kf_bridge_obj, track_object_type=measurment.obj_name)
        for add_on_cls, params in track_params.track_add_ons.items():
            track_obj = add_on_cls(track_obj, **params)
        return track_obj
    
    def reset_tracking_settings(self, tracked_obj:AbstractTrack, object_type: str):
        print("You can not reset tacking parameters with 'TrackKFOObjectTypeBasedFactory' return None...")
        return None