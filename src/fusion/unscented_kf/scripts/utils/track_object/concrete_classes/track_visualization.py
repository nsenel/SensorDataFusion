
import random
import numpy as np
import matplotlib.pyplot as plt
import sys
#Typing imports
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment


np.set_printoptions(linewidth=np.nan)
np.set_printoptions(suppress=True)




class TrackVisualization():
    def __init__(self, live_plot=False, plot_data_on_delete=False):
        self.object_gt_id = None ### TODO: THERE SHOULD BE BETTER BAY OF GETTING THIS !!!
        self.live_plot = live_plot
        self.gt_x, self.gt_y, self.gt_v, self.gt_yaw, self.gt_yaw_rate = [], [], [], [], []
        self.ma_x_lidar, self.ma_y_lidar, self.gt_x_lidar, self.gt_y_lidar  =  [], [], [], []
        self.ma_x_radar, self.ma_y_radar, self.ma_v_radar =  [], [], []
        self.ma_x_cam, self.ma_y_cam = [], []
        self.x_, self.y_, self.v_, self.yaw_, self.yaw_rate_ = [], [], [], [], []
        self.rms_x, self.rms_y, self.rms_v =  [], [], []
        self.live_plot_start_point = 0
        #self.fig.set_size_inches(14, 14)
        self.plot_data_on_delete = plot_data_on_delete
        if self.live_plot:
            self.fig, self.axs = plt.subplots(3,2)
            plt.ion() ##### Keeps plot alive without blocking
    
    def __del__(self):
        ### On object delete print RMSE values and plot the recorded data.
        if sys.meta_path: ### there will be error if python closed with error etc. this will prevent unusefull output message
            self.get_RMSE()
            if self.plot_data_on_delete:
                print(f"Track object with gt_id:{self.object_gt_id} is deleted")
                self.plot_data()

    def add_measurment_data(self, measurment_class: AbstractMeasurment):
        measurment = measurment_class.measurment
        if self.object_gt_id==None: ### TODO: THERE SHOULD BE BETTER BAY OF GETTING THIS !!!
            if "gt_id" in dir(measurment):
                self.object_gt_id = measurment.gt_id
            else:
                self.object_gt_id = -1
        # print(measurment.gt_id)
        self.gt_x.append(measurment.x_gt)
        self.gt_y.append(measurment.y_gt)
        self.gt_v.append(pow((measurment.vx_gt*measurment.vx_gt+measurment.vy_gt*measurment.vy_gt),0.5))
        self.gt_yaw.append(measurment.yaw_gt)
        self.gt_yaw_rate.append(abs(measurment.yawrate_gt))
        track_cor_system_meausrment_values = measurment_class.measurment_matrix_in_cost_function_cor_system
        if type(measurment).__name__=="RadarMeasurmentwithGT":
            self.ma_x_radar.append(track_cor_system_meausrment_values[0])
            self.ma_y_radar.append(track_cor_system_meausrment_values[1])
        elif type(measurment).__name__=="LidarMeasurmentwithGT":
            self.ma_x_lidar.append(track_cor_system_meausrment_values[0])
            self.gt_x_lidar.append(measurment.x_gt)
            self.gt_y_lidar.append(measurment.y_gt)
            self.ma_y_lidar.append(track_cor_system_meausrment_values[1])
        elif type(measurment).__name__=="CameraMeasurmentwithGT":
            self.ma_x_cam.append(track_cor_system_meausrment_values[0])
            self.ma_y_cam.append(track_cor_system_meausrment_values[1])
    
    def add_prediction(self, prediction):
        self.x_.append(prediction[0])
        self.y_.append(prediction[1])
        self.v_.append(abs(prediction[2]))
        self.yaw_.append(prediction[3])
        self.yaw_rate_.append(abs(prediction[4]))

        if self.live_plot:
            self.plot_data(show_plot=False)
            plt.show(block = False)
            plt.pause(0.00001)

    
    def plot_data(self, show_plot=True):
        if show_plot:
            fig, axs = plt.subplots(3,2)
        else:
            axs = self.axs
        axs[0, 0].set_title('XY Position')
        axs[0, 0].plot(self.gt_x, self.gt_y, 'k-',linewidth = 3) # ground truth
        axs[0, 0].plot(self.x_, self.y_, 'r', linewidth = 2)  # estimate
        axs[0, 0].plot(self.ma_x_lidar, self.ma_y_lidar, 'g.', alpha = 0.7) # measurement
        axs[0, 0].plot(self.ma_x_radar, self.ma_y_radar, 'm.', alpha = 0.7) # measurement
        axs[0, 0].plot(self.ma_x_cam, self.ma_y_cam, 'y.', alpha = 0.7) # measurement
        axs[0, 0].grid()
        #axs[0, 0].axis('equal') # very important for visualization purpose

        axs[0, 1].set_title('Velocity')
        axs[0, 1].plot( self.gt_v, 'k-',linewidth = 3) # ground truth
        axs[0, 1].plot( self.v_, 'r', linewidth = 2)  # estimate
        axs[0, 1].legend(['Ground Truth', 'Estimation'])

        axs[1, 0].set_title('Yaw(Orientation)')
        axs[1, 0].plot( self.gt_yaw, 'k-',linewidth = 3) # ground truth
        axs[1, 0].plot( self.yaw_, 'r', linewidth = 2)  # estimate
        axs[1, 0].legend(['Ground Truth', 'Estimation'])

        axs[1, 1].set_title('Yaw Rate')
        axs[1, 1].plot( self.gt_yaw_rate, 'k-',linewidth = 3) # ground truth
        axs[1, 1].plot( self.yaw_rate_, 'r', linewidth = 2)  # estimate
        axs[1, 1].legend(['Ground Truth', 'Estimation'])

        axs[2, 1].set_title('Distance Differance')
        axs[2, 1].plot([x - y for x, y in zip(self.x_, self.gt_x)], 'k-',linewidth = 2) # ground truth
        axs[2, 1].plot([x - y for x, y in zip(self.y_, self.gt_y)], 'r', linewidth = 2)  # estimate
        axs[2, 1].legend(['Dist_Differance_x', 'Dist_Differance_y'])
        if show_plot:
            plt.show()
            
    def rmse(self, predictions, targets):
        data_points_len = min(predictions.shape[0], targets.shape[0])
        if predictions.shape != targets.shape:
            print(" rmse() : predictions and targets are not in same lengght", len(predictions),len(targets))
        return np.sqrt(((predictions[:data_points_len] - targets[:data_points_len]) ** 2).mean())

    def get_RMSE(self):
        
            print("RMSE X:", self.rmse(np.array(self.x_),np.array(self.gt_x)))
            print("RMSE Y:", self.rmse(np.array(self.y_),np.array(self.gt_y)))
            print("RMSE v: ", self.rmse(np.array(self.v_),np.array(self.gt_v)))
            print("RMSE X lidar:", self.rmse(np.array(self.ma_x_lidar),np.array(self.gt_x_lidar)))
            print("RMSE Y lidar:", self.rmse(np.array(self.ma_y_lidar),np.array(self.gt_y_lidar)))
