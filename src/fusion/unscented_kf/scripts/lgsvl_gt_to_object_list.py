#!/usr/bin/env python2

import rospy
import os, random
import math
import numpy as np
from lgsvl_msgs.msg import Detection3D
from lgsvl_msgs.msg import Detection3DArray
from object_detection_msgs.msg import radar_detections_with_GT,lidar_detections_with_GT,camera_detections_with_GT,radar_detection_with_GT,lidar_detection_with_GT,camera_detection_with_GT, ego_car_data


from rotation_translation_calculator import rotationTranslationCalculator
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import message_filters

from tf.transformations import euler_from_quaternion

from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import cv2

np.set_printoptions(suppress=True)


class LGSVL3DtoLidarRadarDetection():
    def __init__(self,save_data_to_text):

        self.rt_calculator = rotationTranslationCalculator()
        odom_sub = message_filters.Subscriber("/odom", Odometry)

        detection3d_sub = message_filters.Subscriber("/simulator/ground_truth/3d_detections/lidar", Detection3DArray)
        ts = message_filters.ApproximateTimeSynchronizer([detection3d_sub, odom_sub], 10, 0.05, allow_headerless=True)
        ts.registerCallback(self.detection3d_lidar)

        detection3d_sub2 = message_filters.Subscriber("/simulator/ground_truth/3d_detections/radar", Detection3DArray)
        ts2 = message_filters.ApproximateTimeSynchronizer([detection3d_sub2, odom_sub], 10, 0.05, allow_headerless=True)
        ts2.registerCallback(self.detection3d_radar)

        detection3d_sub3 = message_filters.Subscriber("/simulator/ground_truth/3d_detections/camera", Detection3DArray)
        ts3 = message_filters.ApproximateTimeSynchronizer([detection3d_sub3, odom_sub], 10, 0.05, allow_headerless=True)
        ts3.registerCallback(self.detection3d_camera)

        rospy.Subscriber("/camera/image_raw", Image, self.image_cb)
        self.bridge = CvBridge()
        self.px,self.py,self.x,self.y=0,0,0,0


        self.cnt=0
        self.pub_radar = rospy.Publisher('/radar/detections', radar_detections_with_GT, queue_size=10)
        self.pub_lidar = rospy.Publisher('/lidar/detections', lidar_detections_with_GT, queue_size=10)
        self.pub_camera = rospy.Publisher('/camera/detections', camera_detections_with_GT, queue_size=10)
        self.ego_car_data = rospy.Publisher('/ego_car_data', ego_car_data, queue_size=10)
        self.save_data_to_text = save_data_to_text
        self.data = []

        camera_extrinsic = np.array([[-0., -1., -0., 0], ###### ground truth generated from lidar data in simulation thats why I calibrate it regarding lidar position
                                    [-0., -0., -1., -0.],
                                    [ 1.,  0.,  0.,  0.]])
        camera_intirinsic = np.array([[1180, 0,    960],
                                      [0,    1180, 540],
                                      [0,    0,    1]])
        self.P = np.matmul(camera_intirinsic, camera_extrinsic)
        self.sensor_filter = {"L1":{"x-axis_max":50,"y-axis_max":30, "x-axis_min":0,   "y-axis_min":-30},
                              "R1":{"x-axis_max":80,"y-axis_max":30, "x-axis_min":1,   "y-axis_min":-30},
                              "C1":{"x-axis_max":50,"y-axis_max":30, "x-axis_min":1, "y-axis_min":-30},
                              "C1_px":{"x-axis_max":1920,"y-axis_max":1080, "x-axis_min":0, "y-axis_min":0}}
    def image_cb(self,image):
        pass#image = self.bridge.imgmsg_to_cv2(image, "bgr8")  
        #cv2.circle(image, (self.px,self.py), 3, (0,0,0), -1)
        #print(self.px,self.py,self.x,self.y)
        #cv2.imshow("Sim cam",image)
        #cv2.waitKey(1) 
    
    def generate_noise(self, value, range):
        noise = (random.random()-0.5)*2 # Generate random value between -1,1
        return value+noise*range

    def filter_by_distance(self, x, y, sensor_type):
        filter_conditions = self.sensor_filter[sensor_type]
        if (x > filter_conditions["x-axis_min"] and x< filter_conditions["x-axis_max"] and
            y > filter_conditions["y-axis_min"] and y< filter_conditions["y-axis_max"]):
            return True
        return False
    
    def filter_randomly(self):
        return True#random.random()>0.33 # Generate random value between 0,1
    
    def detection3d_camera(self, detection_array, odom):
        data_to_save = []
        self.cnt +=1
        (roll, pitch, yaw) = euler_from_quaternion(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)
        self.ground_truth_to_camera([x for x in detection_array.detections], detection_array.header.stamp, data_to_save)

    def detection3d_radar(self, detection_array, odom):
        data_to_save = []
        self.cnt +=1
        (roll, pitch, yaw) = euler_from_quaternion(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)
        self.ground_truth_to_radar([x for x in detection_array.detections], detection_array.header.stamp, data_to_save)
    
    def detection3d_lidar(self, detection_array, odom):
        data_to_save = []
        self.cnt +=1
        (roll, pitch, yaw) = euler_from_quaternion(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)
        px,py,pz = odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z
        vx,vy = odom.twist.twist.linear.x*np.cos(yaw),odom.twist.twist.linear.x*np.sin(yaw)

        current_time = rospy.Time.now()
        if self.save_data_to_text:
            data_to_save = ["C %f %f %d %f %f %f \n"%(px, py, current_time.to_nsec(), yaw, vx, vy)]
        msg = ego_car_data(Header(stamp = current_time),px,py,yaw,vx,vy)
        self.ego_car_data.publish(msg)
        self.ground_truth_to_lidar([x for x in detection_array.detections], detection_array.header.stamp, data_to_save)
    
    def ground_truth_to_radar(self, detection_array, time, data_to_save):
        current_time = rospy.Time.now()
        msg = radar_detections_with_GT(Header(stamp = current_time,frame_id="radar"),[])
        for detection in detection_array:
            obj_name = "car" if detection.label != "person" else "person"
            px,py = detection.bbox.position.position.x,detection.bbox.position.position.y
        
            vx,vy = detection.velocity.linear.x, detection.velocity.linear.y
            if detection.bbox.position.orientation.z > 0.5:
                vx*=-1
            gt_range = pow(pow(px,2)+pow(py,2),0.5)
            gt_bearing = math.atan2(py,px)
            gt_radial_velocity = (vx*gt_range*np.cos(gt_bearing) + vy*gt_range*np.sin(gt_bearing))/gt_range
            #gt_radial_velocity = (vx*px + vy*py)/gt_range #### This is same as (vx*gt_range*np.cos(gt_bearing) + vy*gt_range*np.sin(gt_bearing))/gt_range
            if np.isnan(gt_radial_velocity):
                gt_radial_velocity=0
            #print(gt_range,gt_bearing,gt_radial_velocity)
            #gt_radial_velocity = ego_car_movement[2]-((px*vx+py*vy)/gt_range)
            range_,bearing, radial_velocity = self.generate_noise(gt_range,0.3), self.generate_noise(gt_bearing,0.03), self.generate_noise(gt_radial_velocity,0.3)
            current_time = rospy.Time.now()

            if self.filter_randomly() and self.filter_by_distance(px,py,"R1"):
                v_x, v_y = detection.velocity.linear.x, detection.velocity.linear.y
                msg.radar_detections.append(radar_detection_with_GT(Header(stamp = time,frame_id="radar"),range_, bearing, radial_velocity, obj_name, px, py+0.5, v_x, v_y, gt_bearing, 0.0002, detection.id))
                if self.save_data_to_text:
                    data_to_save.append("R %f %f %f %d %f %f %f %f %f %f %d\n"%(range_, bearing, radial_velocity, current_time.to_nsec(), px, py+0.5, v_x, v_y, gt_bearing, 0.0002, detection.id))
        if self.save_data_to_text:
            data_to_save.append("---\n")
            self.data += data_to_save
        self.pub_radar.publish(msg)
    
    def ground_truth_to_camera(self, detection_array, time, data_to_save):
        current_time = rospy.Time.now()
        msg = camera_detections_with_GT(Header(stamp = current_time,frame_id="camera"),[])
        for detection in detection_array:
            obj_name = "car" if detection.label != "person" else "person"
            px,py = detection.bbox.position.position.x,detection.bbox.position.position.y
            gt_px,gt_py = px,py
            px,py,pz = gt_px, gt_py, -1.718
            img_pos = np.matmul(self.P,np.array([px,py,pz,1]).reshape(4,1))
            img_pos = (img_pos/img_pos[2,0]).astype(int)
            self.px,self.py,self.x,self.y = int(img_pos[0,0]), int(img_pos[1,0]), gt_px, gt_py
            px,py = self.generate_noise(img_pos[0,0],4),self.generate_noise(img_pos[1,0],4)

            current_time = rospy.Time.now()
            
            if self.filter_randomly() and self.filter_by_distance(gt_px, gt_py,"C1") and self.filter_by_distance(px, py, "C1_px"):
                v_x, v_y = detection.velocity.linear.x, detection.velocity.linear.y
                msg.camera_detections.append(camera_detection_with_GT(Header(stamp = time, frame_id="camera"), px, py, obj_name, gt_px+0.168, gt_py, v_x, v_y, math.atan2(gt_py,gt_px), 0.0002, detection.id))
                if self.save_data_to_text:
                    data_to_save.append("cam %f %f %d %f %f %f %f %f %f %d\n"%(px, py, current_time.to_nsec(), gt_px+0.168, gt_py, v_x, v_y, math.atan2(gt_py,gt_px), 0.0002, detection.id ))
        if self.save_data_to_text:
            data_to_save.append("---\n")
            self.data += data_to_save
        self.pub_camera.publish(msg)

    def ground_truth_to_lidar(self, detection_array, time, data_to_save):
        current_time = rospy.Time.now()
        msg = lidar_detections_with_GT(Header(stamp = current_time,frame_id="lidar"),[])
        for detection in detection_array:
            obj_name = "car" if detection.label != "person" else "person"
            px,py = detection.bbox.position.position.x,detection.bbox.position.position.y
            gt_px,gt_py = px,py
            px,py = self.generate_noise(gt_px,0.15), self.generate_noise(gt_py,0.15)

            current_time = rospy.Time.now()
            if self.filter_randomly() and self.filter_by_distance(gt_px,gt_py,"L1"):
                v_x, v_y = detection.velocity.linear.x, detection.velocity.linear.y
                msg.lidar_detections.append(lidar_detection_with_GT(Header(stamp = time, frame_id="lidar"), px, py, obj_name, gt_px, gt_py, v_x, v_y, math.atan2(py,px), 0.0002, detection.id ))
                if self.save_data_to_text:
                    data_to_save.append("L %f %f %d %f %f %f %f %f %f %d\n"%(px, py, current_time.to_nsec(), gt_px, gt_py, v_x, v_y, math.atan2(py,px), 0.0002, detection.id ))
        if self.save_data_to_text:
            data_to_save.append("---\n")
            self.data += data_to_save
        self.pub_lidar.publish(msg)
 
def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
        return roll_x, pitch_y, yaw_z # in radians

if __name__ == '__main__':
    rospy.init_node('LGSVL_gt_to_object_list')
    save_data_to_text = True
    boundingBoxFilter = LGSVL3DtoLidarRadarDetection(save_data_to_text=save_data_to_text)
    rate = rospy.Rate(50) # 10hz
    while not rospy.is_shutdown():
        rate.sleep()
    if save_data_to_text:
        file_name = "recaftored_ukf_drive_paper_random_drop_detect3.txt"
        print("Going to save data...")
        with open(os.path.join('/home/numan/Desktop/simulation_ws/src/unscented_kf/scripts',file_name), 'w') as writer:
            for measurment in boundingBoxFilter.data:
                writer.write(measurment)
