from utils.movement_compensation.abstract_classes.abstract_movement_compensation import AbstractMovmentCompensation
import numpy as np
# Typing imports
from utils.measurment.concrete_classes.incoming_measurments import EgoCarOdomMessage


class MVWithSpeed(AbstractMovmentCompensation):
    def __init__(self) -> None:
        self._last_odom_message = None#EgoCarOdomMessage(x=0, y=0, timestamp=0, yaw=0, vx=0, vy=0)
        self._yaw_rate = 0

    def ego_movement_in_x(self, time_diff) ->float:
        return time_diff*self._last_odom_message.vx
   
    def ego_movement_in_y(self, time_diff) ->float:
        return time_diff*self._last_odom_message.vy
        
    def get_movement_compensation(self, track_x, track_y, measurment_time, last_update_time):
        time_diff = measurment_time-last_update_time
        moved_in_x = self.ego_movement_in_x(time_diff)
        moved_in_y = self.ego_movement_in_y(time_diff)
        
        total_yaw_change = self._yaw_rate*(time_diff)

        ## If car orientation has been changed then transfer track object position to new orientation
        if abs(total_yaw_change) > np.finfo(float).eps: 
            trasformed_point = self.trasform_point_with_car_orientation(track_x, track_y,
                                                                        moved_in_x, moved_in_y,
                                                                        total_yaw_change)
            moved_in_y = trasformed_point[1]-track_y
        return moved_in_x, moved_in_y

    def update_state(self, incoming_ego_data: EgoCarOdomMessage) -> None:
        if self._last_odom_message:
            time_diff = (incoming_ego_data.timestamp - self._last_odom_message.timestamp).to_sec()
            # time_diff = 1.0/10
            if (incoming_ego_data.yaw>=0 and self._last_odom_message.yaw>=0) and (incoming_ego_data.yaw<=0 and self._last_odom_message.yaw<=0):
                self._yaw_rate = (self._last_odom_message.yaw - incoming_ego_data.yaw)/time_diff
            else:
                self._yaw_rate = (abs(incoming_ego_data.yaw) - abs(self._last_odom_message.yaw)) /time_diff
                self._yaw_rate = self._yaw_rate if self._last_odom_message.yaw<0 else -self._yaw_rate
        self._last_odom_message = incoming_ego_data