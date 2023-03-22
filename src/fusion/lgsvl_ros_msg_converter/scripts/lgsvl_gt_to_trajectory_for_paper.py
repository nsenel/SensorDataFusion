#!/usr/bin/env python2

import rospy
import os, random
import math
import numpy as np
from lgsvl_msgs.msg import Detection3D
from lgsvl_msgs.msg import Detection3DArray
from lgsvl_message_converter.msg import radar_detections,lidar_detections,radar_detection,lidar_detection, ego_car_data


from rotation_translation_calculator import rotationTranslationCalculator
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import message_filters

from tf.transformations import euler_from_quaternion

np.set_printoptions(suppress=True)

from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import cv2

class LGSVL3DtoLidarRadarDetection():
    def __init__(self,save_data_to_text, use_global_cordinate_system=True, save_ego_car_postion=True):
        # Ground truth subcribers
        self.use_global_cordinate_system = use_global_cordinate_system
        self.save_ego_car_postion = save_ego_car_postion
        
        if self.use_global_cordinate_system:
            self.init_position_sub = rospy.Subscriber("/odom", Odometry, self.odometry_init_callback)
            self.initial_ego_car_possition = None
            self.total_ego_car_position_change = {"x":0, "y":0, "z":0}

        self.rt_calculator = rotationTranslationCalculator()
        detection3d_sub = message_filters.Subscriber("/simulator/ground_truth/3d_detections/lidar", Detection3DArray)
        odom_sub = message_filters.Subscriber("/odom", Odometry)
        ts = message_filters.ApproximateTimeSynchronizer([detection3d_sub, odom_sub], 10, 0.05, allow_headerless=True)
        ts.registerCallback(self.detection3d_callback)
        self.cnt=0
        self.pub_radar = rospy.Publisher('/radar/detections', radar_detections, queue_size=10)
        self.pub_lidar = rospy.Publisher('/lidar/detections', lidar_detections, queue_size=10)
        self.ego_car_data = rospy.Publisher('/ego_car_data', ego_car_data, queue_size=10)
        self.save_data_to_text = save_data_to_text
        self.data = []
        self.radar_time=False

        rospy.Subscriber("/camera/image_raw", Image, self.image_cb)
        self.px,self.py,self.x,self.y=0,0,0,0
        self.image_show_info = []
        self.trejectory_data = {"Ego Car":{"x":[],"y":[]}}
        camera_extrinsic = np.array([[-0., -1., -0., 0], ###### ground truth generated from lidar data in simulation thats why I calibrate it regarding lidar position
                                    [-0., -0., -1., -0.],
                                    [ 1.,  0.,  0.,  0.]])
        camera_intirinsic = np.array([[1180, 0,    960],
                                      [0,    1180, 540],
                                      [0,    0,    1]])
        self.P = np.matmul(camera_intirinsic, camera_extrinsic)
        self.sensor_filter = {"L1":{"x-axis_max":80,"y-axis_max":30, "x-axis_min":-20,   "y-axis_min":-30},
                              "R1":{"x-axis_max":80,"y-axis_max":30, "x-axis_min":1,   "y-axis_min":-30},
                              "C1":{"x-axis_max":50,"y-axis_max":30, "x-axis_min":1, "y-axis_min":-30},
                              "C1_px":{"x-axis_max":1920,"y-axis_max":1080, "x-axis_min":0, "y-axis_min":0}}
        

    def image_cb(self,image):
        image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        # image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        for element in self.image_show_info:
            px,py,x,y = element[0],element[1],element[2],element[3]
            cv2.circle(image, (px,py), 3, (0,0,0), -1)
            cv2.putText(image, "{:.1f}".format(x), (px,py),
                        cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),1, cv2.LINE_AA)
            cv2.putText(image, "{:.1f}".format(y), (px,py-20),
                        cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),1, cv2.LINE_AA)
        self.image_show_info = []
        cv2.imshow("sdwd",image)
        cv2.waitKey(1)
    
    def filter_by_distance(self, x, y, sensor_type):
        filter_conditions = self.sensor_filter[sensor_type]
        if (x > filter_conditions["x-axis_min"] and x< filter_conditions["x-axis_max"] and
            y > filter_conditions["y-axis_min"] and y< filter_conditions["y-axis_max"]):
            return True
        return False

    def odometry_init_callback(self,odom):
        self.initial_ego_car_possition = {"x":odom.pose.pose.position.x, "y":odom.pose.pose.position.y, "z":odom.pose.pose.position.z}
        self.init_position_sub.unregister()
    
    def generate_noise(self, value, range):
        noise = (random.random()-0.5)*2 # Generate random value between -1,1
        return value+noise*range
    
    def detection3d_callback(self, detection_array, odom):
        print("---",self.cnt,"---",self.use_global_cordinate_system)
        self.cnt +=1
        (roll, pitch, yaw) = euler_from_quaternion(odom.pose.pose.orientation.x, odom.pose.pose.orientation.y,odom.pose.pose.orientation.z,odom.pose.pose.orientation.w)
        px,py,pz = odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z
        vx,vy = odom.twist.twist.linear.x*np.cos(yaw),odom.twist.twist.linear.x*np.sin(yaw)

        if self.save_data_to_text and not self.use_global_cordinate_system:
            current_time = rospy.Time.now()
            self.data.append("C %f %f %d %f %f %f \n"%(px, py, current_time.to_nsec()/1000, yaw, vx, vy))
            msg = ego_car_data(px,py,yaw,vx,vy)
            self.ego_car_data.publish(msg)

        if self.use_global_cordinate_system:
            self.total_ego_car_position_change["x"] = px-self.initial_ego_car_possition["x"]
            self.total_ego_car_position_change["y"] = py-self.initial_ego_car_possition["y"]
            self.total_ego_car_position_change["z"] = pz-self.initial_ego_car_possition["z"]
            if self.save_ego_car_postion:
                ego_car_as_detecion = Detection3D()
                ego_car_as_detecion.id=0
                ego_car_as_detecion.bbox.position.position.x = px-self.initial_ego_car_possition["x"]
                ego_car_as_detecion.bbox.position.position.y = py-self.initial_ego_car_possition["y"]
                ego_car_as_detecion.velocity.linear.x = vx
                ego_car_as_detecion.velocity.linear.y = vy
                ego_car_as_detecion.velocity.angular.z = odom.twist.twist.angular.z
                # if self.radar_time:
                #     self.ground_truth_to_radar([ego_car_as_detecion],0, True)
                # else:
                self.ground_truth_to_lidar([ego_car_as_detecion],0, True)
            
        # if self.radar_time:
        #     self.radar_time=False
        #     self.ground_truth_to_radar(detection_array.detections, yaw)
        # else:
        #     self.radar_time=True
        #     self.ground_truth_to_lidar(detection_array.detections, yaw)
        self.ground_truth_to_lidar(detection_array.detections, yaw)
        self.data.append("---\n")
    
    def ground_truth_to_radar(self, detection_array, yaw, called_from_ego=False):
        current_time = rospy.Time.now()
        msg = radar_detections(Header(stamp = current_time,frame_id="radar"),[])
        transformation = np.array([[np.cos(yaw), -np.sin(yaw)],
                                   [np.sin(yaw),  np.cos(yaw)]])
        for detection in detection_array:
            px,py = detection.bbox.position.position.x,detection.bbox.position.position.y
            transformed_point = np.matmul(transformation, np.array([px,py]).reshape(2,1))
            px,py = -transformed_point[0,0],-transformed_point[1,0]
            #print("1 - ",px,py)
            if self.use_global_cordinate_system:
                px,py = px-self.total_ego_car_position_change["x"],py-self.total_ego_car_position_change["y"]
                #print("2 - ",px,py)
            vx,vy = detection.velocity.linear.x, detection.velocity.linear.y
            gt_range = pow(pow(px,2)+pow(py,2),0.5)
            gt_bearing = math.atan2(py,px)
            #rho_dot = (car.velocity*cos(car.angle)*rho*cos(phi) + car.velocity*sin(car.angle)*rho*sin(phi))/rho;
            gt_radial_velocity = (vx*gt_range*np.cos(gt_bearing) + vy*gt_range*np.sin(gt_bearing))/gt_range
            if np.isnan(gt_radial_velocity):
                gt_radial_velocity=0
            if called_from_ego:
                print(px,py,gt_range,gt_bearing,gt_radial_velocity)
            #print(gt_range,gt_bearing,gt_radial_velocity)
            #gt_radial_velocity = ego_car_movement[2]-((px*vx+py*vy)/gt_range)
            range_,bearing, radial_velocity = self.generate_noise(gt_range,0.3), self.generate_noise(gt_bearing,0.03), self.generate_noise(gt_radial_velocity,0.3)
            msg.radar_detections.append(radar_detection(Header(stamp = current_time,frame_id="radar"),range_, bearing, radial_velocity))
            
            if self.save_data_to_text:
                v_x, v_y = detection.velocity.linear.x, detection.velocity.linear.y
                self.data.append("R %f %f %f %d %f %f %f %f %f %f %d\n"%(range_, bearing, radial_velocity, current_time.to_nsec()/1000, px, py, v_x, v_y, gt_bearing, 0.0002, detection.id))
                # if not called_from_ego:
                #     self.data.append("---\n")
            #print("radar")
        if len(msg.radar_detections)>0:
            self.pub_radar.publish(msg)
    
    def ground_truth_to_lidar(self, detection_array, yaw, called_from_ego=False):
        current_time = rospy.Time.now()
        msg = lidar_detections(Header(stamp = current_time,frame_id="radar"),[])
        transformation = np.array([[np.cos(yaw), -np.sin(yaw)],
                                    [np.sin(yaw), np.cos(yaw)]])
        self.trejectory_data["Ego Car"]["x"].append(-self.total_ego_car_position_change["x"])
        self.trejectory_data["Ego Car"]["y"].append(-self.total_ego_car_position_change["y"])
        for detection in detection_array:
            px,py = detection.bbox.position.position.x,detection.bbox.position.position.y
            transformed_point = np.matmul(transformation, np.array([px,py]).reshape(2,1))
            gt_px,gt_py = -transformed_point[0,0],-transformed_point[1,0]
            if self.use_global_cordinate_system:
                if self.filter_by_distance(gt_px,gt_py,"L1") and detection.id!=7:
                    gt_px,gt_py = gt_px-self.total_ego_car_position_change["x"],gt_py-self.total_ego_car_position_change["y"]
                #if detection.id==3:
                    img_pos = np.matmul(self.P,np.array([detection.bbox.position.position.x,detection.bbox.position.position.y,-1.718,1]).reshape(4,1))
                    img_pos = (img_pos/img_pos[2,0]).astype(int)
                    self.image_show_info.append((int(img_pos[0,0]), int(img_pos[1,0]), gt_px,gt_py))
                    if "Target Car "+str(detection.id) not in self.trejectory_data:
                        self.trejectory_data["Target Car "+str(detection.id)] = {"x":[gt_px],"y":[gt_py]}
                    else:
                        self.trejectory_data["Target Car "+str(detection.id)]["x"].append(gt_px)
                        if detection.id == 2:
                            if gt_py<3.8:
                                gt_py = 3.8
                            if gt_py>4.2:
                                gt_py = 4.2
                        self.trejectory_data["Target Car "+str(detection.id)]["y"].append(gt_py)
                    self.px,self.py,self.x,self.y = int(img_pos[0,0]), int(img_pos[1,0]), gt_px,gt_py

            px,py = self.generate_noise(gt_px,0.15), self.generate_noise(gt_py,0.15)

            current_time = rospy.Time.now()
            msg.lidar_detections.append(lidar_detection(Header(stamp = current_time, frame_id="lidar"), px, py))
            
            if self.save_data_to_text:
                v_x, v_y = detection.velocity.linear.x, detection.velocity.linear.y
                self.data.append("L %f %f %d %f %f %f %f %f %f %d\n"%(px, py, current_time.to_nsec()/1000, gt_px, gt_py, v_x, v_y, math.atan2(py,px), 0.0002, detection.id ))
                # if not called_from_ego:
                #     self.data.append("---\n")
            #print("lidar")
        if len(msg.lidar_detections)>0:
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
    rospy.init_node('LGSVL_3Dto_lidar_radar_detection_')
    save_data_to_text = False
    use_global_cordinate_system = True
    save_ego_car_postion = False if use_global_cordinate_system else True  ####You cant use this without use_global_cordinate_system==True !!! ###Radar seems problemetic dont use it
    boundingBoxFilter = LGSVL3DtoLidarRadarDetection(save_data_to_text=save_data_to_text, use_global_cordinate_system=use_global_cordinate_system, save_ego_car_postion=save_ego_car_postion)
    rospy.spin()
    if save_data_to_text:
        file_name = "data_from_ros_combined_use_glabal_%s_save_ego_car_position_%s.txt"%(use_global_cordinate_system,save_ego_car_postion)
        with open(os.path.join('/home/numan/Desktop/simulation_ws/src/unscented_kf/scripts',file_name), 'w') as writer:
            for measurment in boundingBoxFilter.data:
                # text = ""
                # if type(measurment).__name__=="Radar":
                #     text = format_radar(measurment, idx)
                # else:
                #     text = format_lidar(measurment, idx)
                writer.write(measurment)
    print(boundingBoxFilter.trejectory_data)
    ### Modified for ~/Desktop/simulation_ws$ rosbag play 2021-09-29-13-46-02.bag !!!!!!!!!!!!!!!!!!
    import matplotlib.pyplot as plt
    for key,value in boundingBoxFilter.trejectory_data.items():
        plt.plot([i*-1 for i in value["y"]],value["x"])
    # plt.legend(boundingBoxFilter.trejectory_data.keys(),loc='upper right')
    plt.legend(boundingBoxFilter.trejectory_data.keys())
        #plt.plot(value["y"],value["x"])
    plt.ylabel('Longitudinal(px) Trajectory', fontsize=16)
    plt.xlabel('Lateral(py) Trajectory', fontsize=16)
    plt.show()
