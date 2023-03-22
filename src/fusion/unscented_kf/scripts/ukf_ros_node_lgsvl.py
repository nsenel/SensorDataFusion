#!/usr/bin/env python3
#import matplotlib.pyplot as plt
import threading
import gi
from sympy import true
gi.require_version('Gtk', '2.0')
import os, sys
import numpy as np
from collections import namedtuple,defaultdict
import rospy
from lgsvl_msgs.msg import Detection3DArray

from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import message_filters
from utils.common.enums import SensorTypes, ObjectTypes
from object_detection_msgs.msg import radar_detections_with_GT,lidar_detections_with_GT,camera_detections_with_GT, ego_car_data
from utils.measurment.concrete_classes.incoming_measurments import EgoCarOdomMessage, CameraMeasurmentwithGT, LidarMeasurmentwithGT, RadarMeasurmentwithGT
from main import TrackerRunner
from utils.measurment.concrete_classes.parsed_measurment import ParsedMeasurmentWithGT
from tracker_parameters_config import movement_compensetor, generate_traker
from sensor_parameters_config import configured_sensor_objs_lgsvl
np.set_printoptions(suppress=True)
# np.set_printoptions(linewidth=np.nan)
# np.set_printoptions(suppress=True)


#Car_Pos = namedtuple('Ego_car', 'px py time_stamp yaw vx vy')
class UKFNode():
    def __init__(self,tracker_obj: TrackerRunner, front_cam_topic_name: str = "/camera/image_raw"):
        print(sys.version_info, "!!!!")
        self.object_tracker = tracker_obj
        rospy.Subscriber('/radar/detections', radar_detections_with_GT, self.radar_detections_cb)
        rospy.Subscriber('/lidar/detections', lidar_detections_with_GT, self.lidar_detections_cb)
        rospy.Subscriber('/camera/detections', camera_detections_with_GT, self.camera_detections_cb)
        rospy.Subscriber("/ego_car_data", ego_car_data, self.ego_car_data_cb)

        #rospy.Subscriber("/camera/image_raw", Image, self.image_cb)
        #rospy.Subscriber("/simulator/ground_truth/3d_detections", Detection3DArray, self.ground_truth_3D_detection_cb)
        
        self.bridge = CvBridge()
        self.cnt=0
        self._draw_heading = False

        #### Project tracks back to image for visualezition
        image_sub = message_filters.Subscriber(front_cam_topic_name, Image)
        # image_sub = message_filters.Subscriber(, Image)## Waymo
        gt_3d = message_filters.Subscriber("/simulator/ground_truth/3d_detections/lidar", Detection3DArray)
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, gt_3d], 10, 0.05, allow_headerless=True)
        ts.registerCallback(self.img_gt3_cb)

        self.image = np.zeros((100,100,3),dtype=np.int)
        self.P = self.object_tracker._configured_sensor_objs["Front_Camera"].sensor_properties._projection_matrix_track_to_camera
        self.last_from_img = []

        self.measurment_queue = []
        self.my_lock = threading.Lock()
    def project_point_in_image(self, obj_id, obj_point, color, v=None,yaw=None, t=None):
        # obj_position = np.ones((4,1))
        # obj_position[0:2,0] = np.matmul(transformation,track.prediction)
        img_pos = np.matmul(self.P,obj_point)
        img_pos = (img_pos/img_pos[2,0]).astype(int)
        cv2.circle(self.image, (img_pos[0,0],img_pos[1,0]), 3, color, -1)
        cv2.putText(self.image, str(obj_id), (img_pos[0,0],img_pos[1,0]),
                    cv2.FONT_HERSHEY_DUPLEX, 1,color,1, cv2.LINE_AA)

    def img_gt3_cb(self,image,data):
        #print("img_gt3_cb")
        self.image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        for detection in data.detections: ## Draw ground truth data
            px,py,pz = detection.bbox.position.position.x,detection.bbox.position.position.y, 0
            gt_obj_id = detection.id
            if px >3:
                self.project_point_in_image(obj_id=gt_obj_id, obj_point=np.array([px,py,pz,1]).reshape(4,1), color=(255,255,255))
        if self._draw_heading: self.heading_direction_draw(None, None, None, None, true)
        for track in self.object_tracker._tracker.tracked_object_list: ## Draw tracked data
            obj_position = np.ones((4,1))
            obj_position[2,0]=0
            obj_position[0:2,0] = track.prediction
            if obj_position[0]>0:
                self.project_point_in_image(obj_id=track.track_id, obj_point=obj_position, color=(255, 1, 255))
        cv2.imshow("Tracking Result",self.image)
        cv2.waitKey(1)

    def ego_car_data_cb(self,data):
        if self._draw_heading:
            if self._last_odom_message != None:
                total_yaw_change = 0
                if (data.yaw>=0 and self._last_odom_message.yaw>=0) and (data.yaw<=0 and self._last_odom_message.yaw<=0):
                    total_yaw_change = self._last_odom_message.yaw - data.yaw
                else:
                    total_yaw_change = abs(data.yaw) - abs(self._last_odom_message.yaw)
                    total_yaw_change = total_yaw_change if self._last_odom_message.yaw<0 else -total_yaw_change
                self._yaw_change = total_yaw_change
            self._last_odom_message = data
            self.ego_current_yaw = data.yaw
        ego_measurment = EgoCarOdomMessage(x=data.x, y=data.y, timestamp=data.header.stamp,yaw=data.yaw,vx=data.vx,vy=data.vy)
        self.object_tracker.update_movement_compensetor(ego_measurment)
        
    def radar_detections_cb(self,detections):
        #print("radar")
        measurments = []
        for detection in detections.radar_detections:
            obj_type = ObjectTypes.car if detection.obj_name!="person" else ObjectTypes.person
            measurments.append(RadarMeasurmentwithGT(detection.range,detection.bearing,detection.radial_velocity,detections.header.stamp.to_sec(),
                              obj_type,detection.x_gt,detection.y_gt,detection.vx_gt,detection.vy_gt,detection.yaw_gt,detection.yawrate_gt,detection.gt_id))
        #self.measurment_queue.append(("Front_Radar",measurments))
        self.measurment_queue.append((SensorTypes.radar_polar, "Front_Radar", measurments,detections.header.stamp))
        self.cnt+=1

    def lidar_detections_cb(self,detections):
        measurments = []
        for detection in detections.lidar_detections:
            obj_type = ObjectTypes.car if detection.obj_name!="person" else ObjectTypes.person
            measurments.append(LidarMeasurmentwithGT(detection.x,detection.y,detections.header.stamp.to_sec(),
                                                     obj_type,detection.x_gt,detection.y_gt,detection.vx_gt,detection.vy_gt,detection.yaw_gt,detection.yawrate_gt,detection.gt_id))
        #self.measurment_queue.append(("Front_Lidar",measurments))
        # self.object_tracker.update_tracker_with_meaurments("Front_Lidar",measurments)
        self.measurment_queue.append((SensorTypes.lidar, "Front_Lidar", measurments,detections.header.stamp))
        self.cnt+=1

    def camera_detections_cb(self,detections):
        measurments = []
        for detection in detections.camera_detections:
            obj_type = ObjectTypes.car if detection.obj_name!="person" else ObjectTypes.person
            measurments.append(CameraMeasurmentwithGT(detection.x,detection.y,detections.header.stamp.to_sec(),
                               obj_type, detection.x_gt,detection.y_gt,detection.vx_gt,detection.vy_gt,detection.yaw_gt,detection.yawrate_gt,detection.gt_id))
            x,y = int(detection.x),int(detection.y)
        #     cv2.circle(self.image, (x,y), 3, (255,255,255), -1)
        #     cv2.putText(self.image, str(detection.gt_id), (x,y),
        #                 cv2.FONT_HERSHEY_DUPLEX, 1,(255,255,255),1, cv2.LINE_AA)
        # cv2.imshow("image_object_label",self.image)
        # cv2.waitKey(1)
        #self.measurment_queue.append(("Front_Camera",measurments))
        self.measurment_queue.append((SensorTypes.camera, "Front_Camera", measurments,detections.header.stamp))
        # self.object_tracker.update_tracker_with_meaurments("Front_Camera",measurments)
        self.cnt+=1

if __name__ == '__main__':
    rospy.init_node('UKF_object_tracker_v1')
    trackker_runner_settings = rospy.get_param('~trackker_runner_settings', 'LGSVL')
    if trackker_runner_settings == "LGSVL":
        ####### Lgsvl #######
        object_tracker = TrackerRunner(ParsedMeasurmentWithGT, configured_sensor_objs_lgsvl, generate_traker(), movement_compensetor)
        ukf_node = UKFNode(tracker_obj=object_tracker)
    rate = rospy.Rate(50) # 50hz
    while not rospy.is_shutdown():
        with ukf_node.my_lock:
            if len(ukf_node.measurment_queue)>0:
                ukf_node.object_tracker.update_tracker_with_meaurments(*ukf_node.measurment_queue.pop(0))
                #print("ukf_node.measurment_queue size:", len(ukf_node.measurment_queue))
        rate.sleep()
        
    