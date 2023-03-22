from utils.movement_compensation.abstract_classes.abstract_movement_compensation import AbstractMovmentCompensation
# Typing imports
from utils.measurment.concrete_classes.incoming_measurments import EgoCarOdomMessage
import numpy as np

class MVWithGlobalPostionChange(AbstractMovmentCompensation):
    # def __init__(self, initial_ego_position: EgoCarOdomMessage) -> None:
    def __init__(self) -> None:
        self._last_odom_message = None#initial_ego_position
        self._change_from_last_update = {"x":0, "y":0, "total_yaw_change":0}
    
    def get_movement_compensation(self, track_x, track_y, measurment_time, last_update_time) -> tuple:
        if self._last_odom_message==None or last_update_time > self._last_odom_message.timestamp: ## Check if current odom already applied ...
            return 0,0

        moved_in_x, moved_in_y = self._change_from_last_update["x"], self._change_from_last_update["y"]
        # If car orientation has been changed then transfer track object position to new orientation
        if abs(self._change_from_last_update["total_yaw_change"]) > np.finfo(float).eps:
            trasformed_point = self.trasform_point_with_car_orientation(track_x, track_y,
                                                                        moved_in_x, moved_in_y,
                                                                        self._change_from_last_update["total_yaw_change"])
            moved_in_y = trasformed_point[1]-track_y
        return moved_in_x, moved_in_y
    
    def update_state(self, incoming_ego_data: EgoCarOdomMessage) -> None:
        if self._last_odom_message != None:
            self._change_from_last_update["x"] = incoming_ego_data.x - self._last_odom_message.x
            self._change_from_last_update["y"] = incoming_ego_data.y - self._last_odom_message.y
            total_yaw_change = 0
            if (incoming_ego_data.yaw>=0 and self._last_odom_message.yaw>=0) and (incoming_ego_data.yaw<=0 and self._last_odom_message.yaw<=0):
                total_yaw_change = self._last_odom_message.yaw - incoming_ego_data.yaw
            else:
                total_yaw_change = abs(incoming_ego_data.yaw) - abs(self._last_odom_message.yaw)
                total_yaw_change = total_yaw_change if self._last_odom_message.yaw<0 else -total_yaw_change
            self._change_from_last_update["total_yaw_change"] = total_yaw_change
        self._last_odom_message = incoming_ego_data
    # Angle of the car in LGSVL
    #              3.14
    #         -2.9      2.9
    #     -2.5             2.5
    # -1.9                    1.9