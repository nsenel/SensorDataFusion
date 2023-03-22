# Default value for all below parameters: Types int64 and float64 is 0 and type string=''
# Default conventions:  x - longitudinal direction
#			y - lateral direction
#			z - Upward direction from ground.

# tracking_id: unique id for individual object. Valid Track ID values from 1 to 255. Changed by Numan->"It is unique for object type and can go more than 255"
# 	tracking_id = 0 is invalid
int64 tracking_id

#timetamp of the frame copied to each individual object detected in the frame.
time timestamp

#Bounding Box parameters
float64 obj_box_x
float64 obj_box_y
float64 obj_box_z

#Object Position and Orientation
# Position in meter
# Orientation as Euler Angles in radians.
float64 obj_pose_x
float64 obj_pose_y
float64 obj_pose_z
float64 obj_orientation_x
float64 obj_orientation_y
float64 obj_orientation_z

# Velocity of object given in m/s
float64 vx
float64 vy
float64 vz

# Type of the object detected
# Categories: car, pedestrian, bicycle, bus, truck, unknown 
string obj_class

# Object detection and class Confidence values: between 0 and 1
float64 obj_class_confidence
float64 obj_det_confidence

# Covariance matrix converted as array of float values
float64[] covariance
