from utils.object_factories.concrete_classes.kf_object_factory import KalmanFilterObjectFactory
#Typing exports
from utils.track_object.abstract_classes.abstract_tracked_object import AbstractTrack

class TrackerParameters():
    def __init__(self, tracking_parameters: dict) -> None:
        self._tracking_strategy = tracking_parameters["tracking_strategy"]
        self._track_add_ons = tracking_parameters["add_on_classes_dict"]
        self._kf_bridge_factory = KalmanFilterObjectFactory(kalman_filter=tracking_parameters["kalman_filter"],
                                                            kalman_filter_bridge=tracking_parameters["kf_bridge"],
                                                            filter_settings=tracking_parameters["kf_settings"])

    @property
    def tracking_strategy(self) -> AbstractTrack:
        return self._tracking_strategy

    @property
    def track_add_ons(self) -> dict:
        return self._track_add_ons
    
    @property
    def kf_bridge(self) -> KalmanFilterObjectFactory:
        return self._kf_bridge_factory
