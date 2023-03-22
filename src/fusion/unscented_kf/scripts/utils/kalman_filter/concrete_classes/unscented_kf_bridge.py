from utils.kalman_filter.abstract_classes.abstract_kf_bridge import AbstractKalmanFilterBridge

#Typing exports
from utils.kalman_filter.abstract_classes.abstract_kf import AbstractKalmanFilterImplementer
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment

class UnscentedKalmanFilterBridge(AbstractKalmanFilterBridge):
    def __init__(self, kalman_filter_obj: AbstractKalmanFilterImplementer) -> None:
        super().__init__(kalman_filter_obj)

    def predict_with_KF(self, measurment_time: int) -> None:
        try:
            self.KF.predict(measurment_time)
        except Exception as e:
            raise e

    def update_KF(self, measurment: AbstractMeasurment) -> None:
        self.KF.update(measurment)
