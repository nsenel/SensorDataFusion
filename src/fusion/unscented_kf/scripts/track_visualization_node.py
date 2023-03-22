#!/usr/bin/env python3
#import matplotlib.pyplot as plt
import threading
import gi

gi.require_version('Gtk', '2.0')
import os, sys, time
import numpy as np
import rospy

from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image

from utils.measurment.concrete_classes.incoming_measurments import InitialLidarMeasurment, RadarMeasurmentwithGT, InitialCartesianRadarMeasurment, InitialCameraMeasurment
from utils.common.enums import ObjectTypes
from sensor_parameters_config import  in2lab_sensor_config_map
from in2lab_msgs.msg import TrackObjectList, BoundingBoxes, BoundingBox
np.set_printoptions(suppress=True)


class UKFVisualizationNode():
    def __init__(self,configured_sensor_objs: dict, show_image: bool=True): #### positive_depth only track object which has positive x(depth)
        self.image = np.zeros((100,100,3),dtype=np.int)
        self.allowed_obj_types = [member.value for member in ObjectTypes]
        self.camera_name = rospy.get_param('~camera_name', 'Front_Camera')
        self.camera_obj = configured_sensor_objs[self.camera_name]
        self.P = configured_sensor_objs[self.camera_name].sensor_properties.projection_matrix_track_to_camera
        self.use_rectified_img = configured_sensor_objs[self.camera_name].sensor_properties.use_rectified_img
        self.fisheye = configured_sensor_objs[self.camera_name].sensor_properties.fish_eye
        if self.use_rectified_img or self.fisheye:
            self.K = configured_sensor_objs[self.camera_name].sensor_properties.camera_intrinsic
            self.D = configured_sensor_objs[self.camera_name].sensor_properties.distortion_coefficients
            self.scaled_camera_matrix = configured_sensor_objs[self.camera_name].sensor_properties.scaled_camera_matrix
            if self.fisheye:
                self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.scaled_camera_matrix, self.D, np.eye(3), self.K, (1936, 1216), cv2.CV_16SC2)
        self.show_image = show_image
        self.my_lock = threading.Lock()
        rospy.Subscriber(rospy.get_param('~publish_tracks_topic_name',"/fused_tracked_objects"), TrackObjectList, self.tracked_objects_cb)

        cam_sub_topic_name = rospy.get_param('~subcribe_image_topic_name', '/image_raw')
        rospy.Subscriber(cam_sub_topic_name, Image, self.img_cb)
        
        self.image_pub = rospy.Publisher("~track_object_visualization",Image)

        self.positive_depth=rospy.get_param('~show_only_positive_depth', True)
        self.radar_detections_in_px = []
        self.lidar_detections_in_px = []
        self.bounding_boxes = []
        self.tracked_objects = []
        self.label_colors = {"GT":(255,255,255),"UKF":(255, 1, 255),"Lidar":(100, 255, 100),
                             "Cam":(255, 100, 100), "Radar":(100, 100, 100)}
        self.configured_sensor_objs = configured_sensor_objs
        self.color_info_img = np.concatenate(list(self.generate_info_image(k,v) for k,v in self.label_colors.items()), axis=0)#self.generate_info_image()
        self.color_info_img_h, self.color_info_img_w, _ = self.color_info_img.shape
        self.bridge = CvBridge()
        self.raw_images = []
        #you can delete this time stuff used for see time differance between bbox and shown image ..
        self.last_bbox_update = rospy.get_rostime()
        self.last_image_update_time = rospy.get_rostime()
        self.image_size= (1936,1216)

    def generate_info_image(self, label, label_color):
        FONT,FONT_SCALE, FONT_THICKNESS = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
        bg_color = (0, 0, 0)
        (label_width, label_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
        label_height = label_height + baseline+5
        label_patch = np.zeros((label_height, 100, 3), np.uint8)
        label_patch[:,:] = bg_color
        cv2.circle(label_patch, (5,int(label_height/2)), 4, label_color, -1)
        cv2.putText(label_patch, label, (10, label_height - 1), FONT, FONT_SCALE, label_color, FONT_THICKNESS)
        return label_patch

    
    def point3d_to_2d(self, point3d):
        img_pos = np.matmul(self.P,point3d)
        img_pos = (img_pos/img_pos[2,0]).astype(int)
        return [img_pos[0,0],img_pos[1,0]] if img_pos[0,0]>0 and img_pos[0,0]<self.image_size[0] and img_pos[1,0]>0 and img_pos[1,0]<self.image_size[1] else [0,0]
        

    def img_cb(self,image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        if self.use_rectified_img:
            image = cv2.undistort(image, self.K, self.D, None, self.scaled_camera_matrix,)
        if self.fisheye:
            image = cv2.remap(image, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.raw_images.append((image_msg.header.stamp,image))
    def generate_vis_image(self,image,bboxes,radar_points,lidar_points, tracked_objects, img_time=None):
        self.image = image
        if self.image.shape[0]>100:#### Onceki gelen resmin bbox larina sahibiz o yuzden eski resmin uzerine yeni bboxlari koyuyorum
            
            for bounding_box in bboxes:
                cv2.rectangle(self.image,(bounding_box.xmin,bounding_box.ymin),(bounding_box.xmax,bounding_box.ymax),self.label_colors["Cam"],1)

            for img_pos in lidar_points:
                cv2.circle(self.image, (img_pos[0],img_pos[1]), 3, self.label_colors["Lidar"], -1)
            
            for img_pos in radar_points:
                cv2.circle(self.image, (img_pos[0],img_pos[1]), 10, self.label_colors["Radar"], -1)
            
            for img_pos in tracked_objects:
                #print(img_pos["original_message"])
                original_message = img_pos["original_message"]
                img_pos = img_pos["px"]
                cv2.circle(self.image, (img_pos[0],img_pos[1]), 3, self.label_colors["UKF"], -1)
                prediction_text= f" id:{img_pos[2]}\n y:{original_message.obj_pose_y:.1f}\n x:{original_message.obj_pose_x:.1f}"
                text = "This is \n some text"
                y0, dy = img_pos[1], -15
                for i, line in enumerate(prediction_text.split('\n')):
                    y = y0 + i*dy
                    cv2.putText(self.image, line, (img_pos[0]-10, y ), cv2.FONT_HERSHEY_DUPLEX, 1/1.75,self.label_colors["UKF"],1, cv2.LINE_AA)
            
            self.image[0:self.color_info_img_h,0:self.color_info_img_w,:] = self.color_info_img
            
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.image, "bgr8"))
            except CvBridgeError as e:
                print(e)
            if self.show_image:
                cv2.imshow("track_visualization",self.image)
                cv2.setMouseCallback("track_visualization", self.mouse_event_target)
                cv2.waitKey(1)
    
    def point_cloud_cb(self,detections, args):
        #class_radar = ["Car","Truck","Motorcycle","Bicycle","Pedestrian","Animal","Hazard","Unknown","Overdrivable","Underdrivable"]
        point_cloud = np.array(list(pc2.read_points(detections, skip_nans=True, field_names=("class_of_obj","rel_vel_x","z", "x", "y"))))
        measurments = []
        for detection in point_cloud:
            if detection[1]>1.5 and int(detection[0]) in (0,1,2,3,4):
                obj_class = "person" if int(detection[0])==4 else "car"
                measurments.append(InitialLidarMeasurment(detection[2],detection[3], None,detections.header.stamp.to_nsec()/1000000000.0,0,detection[4]))
        lidar_detections_in_track_cor= [self.configured_sensor_objs[args["sensor_name"]].measurment_matrix_in_track_cor_system(incoming_measurment_from_sensor=lidar_datection,
                                                                                                                              include_heigt=True) for lidar_datection in measurments]
        self.lidar_detections_in_px.append((detections.header.stamp,[self.point3d_to_2d(np.array([x[0],x[1],0,1]).reshape(4,1)) for x in lidar_detections_in_track_cor]))

    def radar_cb(self,detections, args):
        #return
        measurments = []
        for detection in detections.object_list:
                measurments.append(InitialCartesianRadarMeasurment(detection.obj_pose_x,detection.obj_pose_y, abs(detection.vx), None,detections.header.stamp.to_nsec()/1000000000.0))
        radar_detections_in_track_cor= [self.configured_sensor_objs[args["sensor_name"]].measurment_matrix_in_track_cor_system(radar_datection) for radar_datection in measurments]
        self.radar_detections_in_px.append((detections.header.stamp,[self.point3d_to_2d(np.array([x[0],x[1],0,1]).reshape(4,1)) for x in radar_detections_in_track_cor]))

    def lidar_cb(self,detections, args):
        measurments = []
        for detection in detections.object_list:
                measurments.append(InitialLidarMeasurment(detection.obj_pose_x,detection.obj_pose_y, None,detections.header.stamp.to_nsec()/1000000000.0,detection.obj_pose_z))
        lidar_detections_in_track_cor= [self.configured_sensor_objs[args["sensor_name"]].measurment_matrix_in_track_cor_system(incoming_measurment_from_sensor=lidar_datection,
                                                                                                                              include_heigt=True) for lidar_datection in measurments]
        self.lidar_detections_in_px.append((detections.header.stamp,[self.point3d_to_2d(np.array([x[0],x[1],0,1]).reshape(4,1)) for x in lidar_detections_in_track_cor]))
    
    def camera_detections_bbox_cb(self, detections):
        measurments = []
        if self.use_rectified_img or self.fisheye:
            valid_measurment = [obj for obj in detections.bounding_boxes if (obj.id in self.allowed_obj_types) and obj.probability>0.6]
            img_points = [[measurment.xmin, measurment.ymin, measurment.xmax,measurment.ymax] for measurment in valid_measurment]
            
            if len(img_points):
                if self.fisheye:
                    rectified_img_points = cv2.fisheye.undistortPoints(np.array(img_points, dtype=np.float32).reshape(1,len(img_points)*2,2), self.scaled_camera_matrix,
                                                                                self.D, None, 
                                                                                self.K).reshape(len(img_points)*2,2).astype(int)
                else:
                    rectified_img_points = cv2.undistortPoints(np.array(img_points, dtype=np.float32).reshape(1,len(img_points)*2,2), self.K,
                                                                self.D, None, 
                                                                self.scaled_camera_matrix).reshape(len(img_points)*2,2).astype(int)
                for idx,measurment in enumerate(valid_measurment):
                    idx *= 2 ## for each object 2 img point is required therefore it is idx multiplyed with two
                    measurments.append(BoundingBox(measurment.probability,
                                                    rectified_img_points[idx][0], rectified_img_points[idx][1],
                                                    rectified_img_points[idx+1][0], rectified_img_points[idx+1][1],
                                                    measurment.id, measurment.Class))
        else:
            measurments = [obj for obj in detections.bounding_boxes if (obj.id in self.allowed_obj_types) and obj.probability>0.6]
        if self.fisheye:    
            img_points = [[int(detection.xmin+(detection.xmax-detection.xmin)/2),int(detection.ymax)] for detection in detections.bounding_boxes]
            if len(img_points):
                rectified_img_points = cv2.fisheye.undistortPoints(np.array(img_points, dtype=np.float32).reshape(1,len(img_points),2), self.scaled_camera_matrix,
                                                    self.D, None, 
                                                    self.K).reshape(len(img_points),2)
        self.last_bbox_update = detections.header.stamp
        self.bounding_boxes.append((detections.header.stamp,measurments))

    def tracked_objects_cb(self,tracked_objects):###TODO change this positive_depth to field of view !!
        tracked_objects_px = []
        for tracked_object in tracked_objects.object_list:
            if (self.positive_depth and tracked_object.obj_pose_x>0) or (not self.positive_depth and tracked_object.obj_pose_x<0) or self.fisheye:
                #print("obj_pos; ", tracked_object.x,tracked_object.y)
                if self.fisheye:
                    image_cor = self.point3d_to_2d(np.array([tracked_object.obj_pose_y,tracked_object.obj_pose_x,0,1]).reshape(4,1))
                else:
                    image_cor = self.point3d_to_2d(np.array([tracked_object.obj_pose_x,tracked_object.obj_pose_y,0,1]).reshape(4,1))
                image_cor.append(tracked_object.tracking_id)
                tracked_objects_px.append({"px":image_cor,"original_message":tracked_object})
        self.tracked_objects.append((tracked_objects.header.stamp,tracked_objects_px))
    
    def best_time_selector(self, target_time, object_list, exact_time=False):
        min_time = 0.3
        index = -1
        for idx,obj in enumerate(object_list):
            time,_ = obj
            if exact_time:
                if abs((time-target_time).to_sec()) == 0:
                    index = idx
            else:
                if abs((time-target_time).to_sec()) < min_time:
                    min_time = abs((time-target_time).to_sec())
                    index = idx
        if index ==-1:
            return False,object_list[-20:] ## Dont flash the memory if there is no match for long time for some reason
        return True,object_list[index:]
    
    def convert_px_to_world(self, point, Z=None):
        meas = InitialCameraMeasurment(point[0],point[1], "s", 0)
        calculated_depth = self.camera_obj.measurment_matrix_in_cost_function_cor_system(meas)
        test= (round(calculated_depth[0], 2),round(calculated_depth[1], 2),0)
        return test

    def mouse_event_target(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("click: ",x,y, self.convert_px_to_world((x,y)))



if __name__ == '__main__':
    rospy.init_node('track_visualization_node_v1')
    trackker_runner_settings = rospy.get_param('~trackker_runner_settings', 'in2lab')
    if trackker_runner_settings == "in2lab":
        sensor_configuration_name = rospy.get_param('~sensor_configuration_name', "configured_sensor_objs_in2lab_11")
        if sensor_configuration_name not in in2lab_sensor_config_map.keys():
            print("There is not sensor configuration name for given name '{sensor_configuration_name}' be sure name exist in sensor_parameters_config.py in2lab_sensor_config_map as key !!!")
            sys.exit(0)
        vis_node = UKFVisualizationNode(in2lab_sensor_config_map[sensor_configuration_name],rospy.get_param('~open_cv_window_for_fusion_image', False))
        ### Real Sensors
        bbox_sub_topic_name = rospy.get_param('~bbox_subcribe_topic_name', '/darknet_ros/bounding_boxes')
        rospy.Subscriber(bbox_sub_topic_name, BoundingBoxes, vis_node.camera_detections_bbox_cb)
        if rospy.has_param('~radar_sensor_name') and rospy.has_param('~radar_detections_topic_name'):
            radar_detect_sub_topic_name = rospy.get_param('~radar_detections_topic_name')
            radar_sensor_name = rospy.get_param('~radar_sensor_name')
            if not rospy.get_param('~detection_msg_as_point_cloud', False):
                rospy.Subscriber(radar_detect_sub_topic_name, TrackObjectList, vis_node.radar_cb, {"sensor_name":radar_sensor_name})
            else:
                from sensor_msgs.msg import PointCloud2
                import sensor_msgs.point_cloud2 as pc2
                rospy.Subscriber(radar_detect_sub_topic_name, PointCloud2, vis_node.point_cloud_cb, {"sensor_name":radar_sensor_name})
        if rospy.has_param('~lidar_sensor_name') and rospy.has_param('~lidar_detections_topic_name'):
            lidar_detect_sub_topic_name = rospy.get_param('~lidar_detections_topic_name')
            lidar_sensor_name = rospy.get_param('~lidar_sensor_name')
            if not rospy.get_param('~detection_msg_as_point_cloud', False):
                rospy.Subscriber(lidar_detect_sub_topic_name, TrackObjectList, vis_node.lidar_cb, {"sensor_name":lidar_sensor_name})
            else:
                from sensor_msgs.msg import PointCloud2
                import sensor_msgs.point_cloud2 as pc2
                rospy.Subscriber(lidar_detect_sub_topic_name, PointCloud2, vis_node.point_cloud_cb, {"sensor_name":lidar_sensor_name})

    rate = rospy.Rate(50) # 50hz
    while not rospy.is_shutdown():
        with vis_node.my_lock:
            if len(vis_node.raw_images):
                found_matching_bbox = False
                for idx,i in enumerate(vis_node.raw_images):#### Check if there is raw_image bboxes match if there is not no need for further checks..
                    found_match_, vis_node.bounding_boxes = vis_node.best_time_selector(i[0], vis_node.bounding_boxes,exact_time=True)
                    if found_match_:
                        vis_node.raw_images = vis_node.raw_images[idx:]
                        found_matching_bbox = True

                if found_matching_bbox:
                    target_time, image = vis_node.raw_images.pop(0)
                    bboxes,radar_points,lidar_points, tracked_objects, img = [],[],[], [], image
                    if len(vis_node.bounding_boxes):
                        found_match_,vis_node.bounding_boxes = vis_node.best_time_selector(target_time, vis_node.bounding_boxes, exact_time=True)
                        if found_match_:
                            _, bboxes = vis_node.bounding_boxes.pop(0)
                    if len(vis_node.radar_detections_in_px):
                        found_match_,vis_node.radar_detections_in_px = vis_node.best_time_selector(target_time, vis_node.radar_detections_in_px)
                        if found_match_:
                            _, radar_points = vis_node.radar_detections_in_px.pop(0)
                            #print("radar_time_diff", (target_time-_).to_sec())
                        else:
                            _, radar_points = vis_node.radar_detections_in_px[-1]
                    if len(vis_node.lidar_detections_in_px):
                        found_match_,vis_node.lidar_detections_in_px = vis_node.best_time_selector(target_time, vis_node.lidar_detections_in_px)
                        if found_match_:
                            _, lidar_points = vis_node.lidar_detections_in_px.pop(0)
                            #print("lidar_time_diff", (target_time-_).to_sec())
                        else:
                            _, lidar_points = vis_node.lidar_detections_in_px[-1]
                    if len(vis_node.tracked_objects):
                        #print("len vis_node.tracked_objects: ",len(vis_node.tracked_objects))
                        found_match_,vis_node.tracked_objects = vis_node.best_time_selector(target_time, vis_node.tracked_objects)
                        if found_match_:
                            if len(vis_node.tracked_objects)>1:#### In case we dont receive new msg in between
                                _, tracked_objects = vis_node.tracked_objects.pop(0)
                            else:
                                _, tracked_objects = vis_node.tracked_objects[0]
                            #print("tracked_time_diff", (target_time-_).to_sec())
                        else:
                            _,tracked_objects = vis_node.tracked_objects[-1]
                    vis_node.generate_vis_image(img,bboxes,radar_points,lidar_points, tracked_objects, target_time)
        rate.sleep()
    
    cv2.destroyAllWindows()