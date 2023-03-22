#!/usr/bin/env python2

import numpy as np
from tf.transformations import euler_from_quaternion, euler_matrix, quaternion_multiply


class rotationTranslationCalculator():
    def __init__(self):
        self.previous_pose = [0,0,0,0]
        self.previous_position = np.array([0,0,0])
        self.R = np.zeros((4, 4), int)
        np.fill_diagonal(self.R, 1)
    
    def relativ_angle_change(self, quaternion):
        relativ_angle =  quaternion_multiply(quaternion, self.previous_pose)
        self.previous_pose = quaternion
        self.previous_pose[3] = self.previous_pose[3]*-1 # Reverse the matrix for next iteration
        #relativ_angle[3] = -relativ_angle[3]
        return relativ_angle

    def calculate_rotation_matrix(self,quaternion):
        (roll, pitch, yaw) = euler_from_quaternion (self.relativ_angle_change(quaternion))
        self.R = euler_matrix(roll, pitch, yaw)
    
    def calculate_trasformation_matrix(self,position):
        relativ_pos = self.previous_position - position # I swap this 09.07.2021  24s 2021-07-05-12-42-10.bag cars in other road look better
        self.previous_position = position
        self.R[:3,3] = relativ_pos

    def update_with_odom_message(self, odm_message):
        self.previous_odom_msg = odm_message
        qorientation_q = odm_message.pose.pose.orientation
        position = odm_message.pose.pose.position
        qorientation_list = [qorientation_q.x, qorientation_q.y, qorientation_q.z, qorientation_q.w]
        position_list = np.array([position.x, position.y, position.z])
        self.calculate_rotation_matrix(qorientation_list)
        self.calculate_trasformation_matrix(position_list)
        return self.R
    
