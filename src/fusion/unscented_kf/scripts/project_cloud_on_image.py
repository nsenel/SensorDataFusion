#!/usr/bin/env python3

"""
    @author: Numan Senel
    @email: Numan.Senel@thi.de
"""
from sensor_parameters_config import  in2lab_sensor_config_map
import rospy
from sensor_msgs.msg import Image, PointCloud2
from in2lab_msgs.msg import TrackObjectList,ClusterList
from cv_bridge import CvBridge
import time
import cv2
import numpy as np
from geometry_msgs.msg import PointStamped
#from project_cloud_on_image.put_points_on_image import point_cloud_on_image
import sensor_msgs.point_cloud2 as pc2
from matplotlib import cm

np.set_printoptions(suppress=True)

rgb_colors=np.array(cm.get_cmap('hsv',8)(np.linspace(0, 1, 120)))[:,:3]*255 #other maps can be seen in https://matplotlib.org/stable/gallery/color/colormap_reference.html
class LidarProjecter():
    def __init__(self, window_name: str="Lidar Projector", show_image=False):
        self.window_name = window_name
        self.bridge = CvBridge()
        self.show_image = show_image
        self.image=np.zeros((500,500,3),dtype=np.uint8)
        self.point_cloud = None
        self.radar_cloud,self.radar_pc2_cloud = None, None
        self.radar_detection, self.lidar_detection = None, None
    
    def post_init(self, configured_sensor_objs: dict):
        camera_name = rospy.get_param('~camera_name', 'Right_Camera')
        lidar_name = rospy.get_param('~lidar_sensor_name', 'Right_Lidar')
        radar_name = rospy.get_param('~radar_sensor_name', 'Right_Radar')
        cam_sub_topic_name = rospy.get_param('~image_topic_name', '/image_raw')
        self.zero_height = rospy.get_param('~set_to_zero_height', False)
        self.rotating_lidar = rospy.get_param('~rotating_lidar', False)

        rospy.Subscriber(cam_sub_topic_name, Image, self.image_callback)#, queue_size=1)
        self.intrinsic = configured_sensor_objs[camera_name].sensor_properties.camera_intrinsic
        self.projection_matrix = configured_sensor_objs[camera_name].sensor_properties.projection_matrix_track_to_camera
        self.distCoeffs = configured_sensor_objs[camera_name].sensor_properties.distortion_coefficients
        self.index_info_default = {"depth_index":0,"left_right_index":1,"height_index":2}
        
        if lidar_name in configured_sensor_objs:
            ## Set up raw topic listerner
            lidar_raw_sub_topic_name = rospy.get_param('~lidar_row_topic_name', 'bf_lidar/points_raw')
            rospy.Subscriber(lidar_raw_sub_topic_name, PointCloud2, self.raw_lidar_callback)
            x_idx,y_idx,z_idx = rospy.get_param('~depth_index_lidar', 0), rospy.get_param('~left_right_index_lidar', 1), rospy.get_param('~height_index_lidar', 2)
            self.index_info_lidar = {"depth_index":x_idx,"left_right_index":y_idx,"height_index":z_idx}
            self.lidar_to_track = configured_sensor_objs[lidar_name].sensor_properties.RT_mtx_sensor_to_track
            ## Set up detection topic listerner
            lidar_detect_sub_topic_name = rospy.get_param('~lidar_detection_topic_name', '/lidar_object_detection')
            if not rospy.get_param('~detection_msg_as_point_cloud', False):
                rospy.Subscriber(lidar_detect_sub_topic_name, TrackObjectList, self.lidar_detection_callback)
            else:
                rospy.Subscriber(lidar_detect_sub_topic_name, PointCloud2, self.lidar_detection_callback_as_pc2)

        if radar_name in configured_sensor_objs:
            print("radar_name:",radar_name)
            ## Set up raw topic listerner
            radar_raw_sub_topic_name = rospy.get_param('~radar_row_topic_name', '/RadarFrames')
            if not rospy.get_param('~detection_msg_as_point_cloud', False):
                rospy.Subscriber(radar_raw_sub_topic_name, ClusterList, self.raw_radar_callback)
            else:
                rospy.Subscriber(radar_raw_sub_topic_name, PointCloud2, self.raw_radar_callback_cloud)
            self.radar_to_track = configured_sensor_objs[radar_name].sensor_properties.RT_mtx_sensor_to_track
            ## Set up detection topic listerner
            radar_detect_sub_topic_name = rospy.get_param('~radar_detection_topic_name', '/RadarDetectionObjects')
            if not rospy.get_param('~detection_msg_as_point_cloud', False):
                rospy.Subscriber(radar_detect_sub_topic_name, TrackObjectList, self.radar_detection_callback)
            else:
                rospy.Subscriber(radar_detect_sub_topic_name, PointCloud2, self.radar_detection_callback_as_pc2)

        self.image_pub = rospy.Publisher('/lidar_projection', Image, queue_size=1)
    def raw_lidar_callback(self,cloud):
        self.point_cloud = cloud
    
    def raw_radar_callback(self, clusters):#Conti 4xx radar
        radar_points = []
        for cluster in clusters.points:
            radar_points.append([cluster.longitude_dist, cluster.lateral_dist, 0])
        self.radar_cloud = np.array(radar_points).reshape(len(radar_points),3)

    def raw_radar_callback_cloud(self, cloud):#Conti 5xx radar
        self.radar_pc2_cloud = cloud
    
    def radar_detection_callback(self, detections):#Conti 4xx detections
        val_dect = []
        for detection in detections.object_list:
            val_dect.append([detection.obj_pose_x,detection.obj_pose_y,0])
        self.radar_detection = np.array(val_dect).reshape(len(val_dect),3)

    
    def radar_detection_callback_as_pc2(self, detections):#Conti 5xx detections
        self.radar_detection = detections
    
    def lidar_detection_callback(self, detections):
        val_dect = []
        for detection in detections.object_list:
            val_dect.append([detection.obj_pose_x,detection.obj_pose_y,detection.obj_pose_z])
        self.lidar_detection = np.array(val_dect).reshape(len(val_dect),3)
    
    def lidar_detection_callback_as_pc2(self, detections):#Conti 5xx detections
        self.lidar_detection = detections
    
    def image_callback(self, image):
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.image = cv2.undistort(cv_image, self.intrinsic, self.distCoeffs)
    
    def project_points(self):
        image = self.image.copy()
        if self.point_cloud:
            image =LidarProjecter.point_cloud_on_image(undestered_img=image, point_cloud=self.point_cloud, index_info=self.index_info_lidar,
                                                    projection_matrix=self.projection_matrix, sensor_to_track=self.lidar_to_track,
                                                    veloyne=self.rotating_lidar, zero_height=self.zero_height)
        if self.radar_pc2_cloud:
            image =LidarProjecter.point_cloud_on_image(undestered_img=image, point_cloud=self.point_cloud, index_info=self.index_info_default,
                                                    projection_matrix=self.projection_matrix, sensor_to_track=self.radar_to_track,
                                                    veloyne=False, zero_height=self.zero_height, marker_size=5, shape_star=True)
        if isinstance(self.radar_cloud, np.ndarray):##This can combine with self.radar_pc2_cloud snce parameters mostly same
            image =LidarProjecter.point_cloud_on_image(image, point_cloud=self.radar_cloud, index_info=self.index_info_default,
                                                    projection_matrix=self.projection_matrix, sensor_to_track=self.radar_to_track,
                                                    veloyne=False, zero_height=False, color=(0,0,0), marker_size=5, shape_star=True)
        if type(self.radar_detection)!=type(None):
            image =LidarProjecter.point_cloud_on_image(image, point_cloud=self.radar_detection, index_info=self.index_info_default,
                                                    projection_matrix=self.projection_matrix, sensor_to_track=self.radar_to_track,
                                                    veloyne=False, zero_height=False, color=(0,0,0), use_cv=True)
        if type(self.lidar_detection)!=type(None):
            image =LidarProjecter.point_cloud_on_image(image, point_cloud=self.lidar_detection, index_info=self.index_info_lidar,
                                                    projection_matrix=self.projection_matrix, sensor_to_track=self.lidar_to_track,
                                                    veloyne=False, zero_height=True, color=(160,25,200), use_cv=True)
        if self.show_image:
            cv2.imshow(self.window_name , image)
            cv2.waitKey(1)
        else:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(image.astype("uint8"), "bgr8"))
    
    @staticmethod
    def point_cloud_on_image(undestered_img,point_cloud, index_info, projection_matrix, sensor_to_track,
                            veloyne=False, zero_height=False, color=None, marker_size=3, shape_star=False, use_cv=False):
        """
        blickfeld: [0:left-right,1:depth,2:height]
        veloyne: []
        conti_radar_5x: [depth, left-right,height]
        """
        if type(point_cloud) == type(None):
            print("Emtpy point cloud returning same image")
            return undestered_img
        elif isinstance(point_cloud, np.ndarray):
            pass
        else:
            point_cloud = np.array(list(pc2.read_points(point_cloud, skip_nans=True, field_names=("z", "x", "y"))))
        
        point_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1), dtype=point_cloud.dtype)))

        if veloyne:
            point_cloud = point_cloud[point_cloud[:, index_info["depth_index"]] >= 0]

        point_cloud = np.transpose(point_cloud)
        depth = point_cloud[index_info["depth_index"], :].copy()
        point_cloud[:3,:]=np.matmul(sensor_to_track,point_cloud)
        if zero_height:
            point_cloud[index_info["height_index"], :] =  0
        point_cloud_in_px = np.matmul(projection_matrix,point_cloud)
        point_cloud_in_px = np.array([point_cloud_in_px[0,:]/point_cloud_in_px[2,:],
                                      point_cloud_in_px[1,:]/point_cloud_in_px[2,:],
                                      depth]).T
        point_cloud_in_px = point_cloud_in_px.astype(int)
        (rows,cols,channels) = undestered_img.shape
        
        point_cloud_in_px = point_cloud_in_px[
        np.logical_not(np.logical_or(
            np.logical_not(np.logical_and(point_cloud_in_px[:,0] > marker_size, point_cloud_in_px[:,0] < cols-marker_size)),
            np.logical_not(np.logical_and(point_cloud_in_px[:,1] > marker_size, point_cloud_in_px[:,1] < rows-marker_size))
        ))
        ]
        point_cloud_in_px = point_cloud_in_px[np.logical_not(np.logical_and(point_cloud_in_px[:,2] > 0, point_cloud_in_px[:,2] < 120))==False]# if you goingto use rgb_colors
        if use_cv == False:
            if color is None:
                color = np.array(rgb_colors[point_cloud_in_px[:,2]],dtype=np.uint8).reshape(undestered_img[point_cloud_in_px[:,1],point_cloud_in_px[:,0],:].shape)
            # undestered_img[point_cloud[:,1],point_cloud[:,0],:]=color
            for i in range(1,marker_size):
                undestered_img[point_cloud_in_px[:,1]+i,point_cloud_in_px[:,0],:]  =color
                undestered_img[point_cloud_in_px[:,1],point_cloud_in_px[:,0]+i,:]  =color
                undestered_img[point_cloud_in_px[:,1]-i,point_cloud_in_px[:,0],:]  =color
                undestered_img[point_cloud_in_px[:,1],point_cloud_in_px[:,0]-i,:]  =color
                if shape_star:
                    undestered_img[point_cloud_in_px[:,1]+i,point_cloud_in_px[:,0]+i,:]  =color
                    undestered_img[point_cloud_in_px[:,1]-i,point_cloud_in_px[:,0]-i,:]  =color
        else:
            for cor in point_cloud_in_px: ##4 times slower do not use it unless it is really needed.
                px_c, px_r, _ = cor
                cv2.circle(undestered_img, (px_c,px_r), 5, color, -1)
        return undestered_img        

if __name__ == '__main__':
    rospy.init_node('project_cloud_on_image')
    trackker_runner_settings = rospy.get_param('~trackker_runner_settings', 'in2lab')
    if trackker_runner_settings == "in2lab":
        sensor_configuration_name = rospy.get_param('~sensor_configuration_name', "configured_sensor_objs_longfei")
        if sensor_configuration_name not in in2lab_sensor_config_map.keys():
            print("There is not sensor configuration name for given name '{sensor_configuration_name}' be sure name exist in sensor_parameters_config.py in2lab_sensor_config_map as key !!!")
        
    lidar_projector = LidarProjecter(window_name="Projector", show_image=True)
    lidar_projector.post_init(in2lab_sensor_config_map[sensor_configuration_name])

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        # start = time.time()
        lidar_projector.project_points()
        # print(time.time()-start)
        rate.sleep()
    cv2.destroyAllWindows()
