from utils.custom_exceptions.defined_exceptions import MeasurmentTimeNone, MeasurmentTimeSmaller, PMatrixHasNotPossitiveDefine
from utils.kalman_filter.abstract_classes.abstract_kf import AbstractKalmanFilterImplementer
from utils.kalman_filter.concrete_classes.kf_state_holder import KFStateHolder
from utils.parameters.concrete_classes.unscented_kf_parameters import UKFFilterParameters
from utils.common.enums import SensorTypes
import math
#Typing exports
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
import numpy as np

class UnscentedKalmanFilter(AbstractKalmanFilterImplementer):
    """ Unscented Kalman filter with CTRV as motion model implementation. """
    def __init__(self, filter_parameters: UKFFilterParameters) -> None:
        ##Assign parameters
        self._state_dim: int        = filter_parameters.state_dim
        self._P: np.ndarray         = filter_parameters.convariance_matrix
        self._process_R: np.ndarray = filter_parameters.process_noise_parameters
        self._n_aug: int            = filter_parameters.augmented_state_dimension
        self._lambda: int           = filter_parameters.sigma_point_spreading_ratio
        self._n_sig: int            = filter_parameters.sigma_points_dimension

        # Set initial state vector
        self._x = np.zeros(self.state_dimension)
        # predicted sigma points matrix
        self._Xsig_pred=None
        # Weights of sigma points
        self._weights=np.zeros(self._n_sig)
        self._weights.fill(0.5 / (self._n_aug + self._lambda))
        self._weights[0] = self._lambda / float(self._lambda + self._n_aug) ######with out float python2 makes it -1 which change float !!!!

        self._last_update_time = 0
        self.fist_detection_time = None ### Tiago said we might need it...
        self.initial_object_location = None
        # self.nis = {SensorTypes.camera:[],SensorTypes.radar_cartesian:[],SensorTypes.lidar:[]}#TODO remove temp
    
    def roll_back_prediction(self):
        """ When an object doesn't have a new measurement to update its state, option to return back to the state before prediction.
            In other words, the function makes the state return back to the last update state.
            TODO: integrate roll back to unscented_kf_bridge this function should be called directly!! for example map location wont be updated in direct call!!!"""
        self._last_update_time = self.last_update_state.update_time
        self._P = self.last_update_state.p.copy()
        self._x = self.last_update_state.x.copy()

    @property
    def x_postion(self)-> float:
        return self.x[0]
    
    @x_postion.setter
    def x_postion(self, values: tuple)-> None: #tuple(new_postion: float, uncertainty: float=0)
        if type(values) == tuple: ## If there is only position information use default uncertainty
            new_postion, uncertainty = values[0], values[1]
        else:
            new_postion, uncertainty = values, False

        self._x[0] = new_postion
        if uncertainty:
            self._P[0,0] += uncertainty
        else:
            self._P[0,0] = self._P[0,0] * 1.6

    @property
    def y_postion(self)->float:
        return self.x[1]

    @y_postion.setter
    def y_postion(self, values: tuple)-> None: #tuple(new_postion: float, uncertainty: float=0)
        if type(values) == tuple:
            new_postion, uncertainty = values[0], values[1]
        else:
            new_postion, uncertainty = values, False

        self._x[1] = new_postion
        if uncertainty:
            self._P[1,1] += uncertainty
        else:
            self._P[1,1] = self._P[1,1] * 1.6

    @property
    def state_dimension(self) -> int:
        return self._state_dim

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def P(self) -> np.ndarray:
        return self._P

    @property
    def last_update_time(self) -> int:
        return self._last_update_time

    def init_measurment(self, initial_measurment: AbstractMeasurment) -> None:
        """ Basic initial object state assignment function """
        self._x[0:2] = initial_measurment.measurment_matrix_in_track_cor_system
        if SensorTypes.radar_cartesian == initial_measurment.sensor_type: # Get first measurment velocity to init if avaible
            self._x[2]=initial_measurment.measurment_matrix[2]
        self._last_update_time = initial_measurment.measurment_time
        self.fist_detection_time = initial_measurment.measurment_time
        self.initial_object_location = initial_measurment.measurment_matrix_in_track_cor_system
        self.last_update_state = KFStateHolder(self._P.copy(), self._x.copy(), initial_measurment, initial_measurment.measurment_time)
    
    def extensive_init(self, initial_state_vector: np.ndarray, intial_p:np.ndarray, initial_measurment_time: int, initial_measutment_pos: np.ndarray, register_time: int) -> None:
        """ Initial state assigment function when there is extentive information about object state """
        self._x = initial_state_vector
        self._P = intial_p
        self._last_update_time = register_time
        self.fist_detection_time = initial_measurment_time
        self.initial_object_location = initial_measutment_pos
        self.last_update_state = KFStateHolder(self._P.copy(), self._x.copy(), None, self.last_update_time)

    def generate_sigma_points(self, x_aug: np.ndarray, P_aug: np.ndarray) -> np.ndarray:
        """ Generate sigma points depending on current uncertainty about the object state """
        n = x_aug.shape[0]
        # create sigma point matrix
        Xsig = np.zeros((n,self._n_sig))
        # calculate square root of P
        try:
            A = np.linalg.cholesky(P_aug)
        except Exception as e:
            raise PMatrixHasNotPossitiveDefine
        Xsig[:,0] = x_aug
        lambda_plue_n_x_sqrt = (self._lambda + n)**0.5
        for i in range(0,n):
            Xsig[:,i + 1] = x_aug + lambda_plue_n_x_sqrt * A[:,i]
            Xsig[:,i + 1 + n] = x_aug - lambda_plue_n_x_sqrt * A[:,i]
        return Xsig

    def predict_sigma_points(self, Xsig, delta_t) ->np.ndarray:
        """ Calculate object state for each sigma point """
        Xsig_pred = np.zeros((self.state_dimension, self._n_sig))
        for i in range(0,self._n_sig):
            #extract values for better readability
            p_x =      Xsig[0,i]
            p_y =      Xsig[1,i]
            v =        Xsig[2,i]
            yaw =      Xsig[3,i] 
            yawd =     Xsig[4,i]
            nu_a =     Xsig[5,i]
            nu_yawdd = Xsig[6,i]
            #predicted state values
            #avoid division by zero
            if (abs(yawd) > np.finfo(float).eps):
                px_p = p_x + v/yawd * ( np.sin (yaw + yawd*delta_t) - np.sin(yaw))
                py_p = p_y + v/yawd * ( np.cos(yaw) - np.cos(yaw+yawd*delta_t) )
            else:
                px_p = p_x + v*delta_t*np.cos(yaw)
                py_p = p_y + v*delta_t*np.sin(yaw)

            v_p = v
            yaw_p = yaw + yawd*delta_t
            yawd_p = yawd
            #add noise
            px_p = px_p + 0.5*nu_a*delta_t*delta_t * np.cos(yaw)
            py_p = py_p + 0.5*nu_a*delta_t*delta_t * np.sin(yaw)
            v_p = v_p + nu_a*delta_t

            yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t
            yawd_p = yawd_p + nu_yawdd*delta_t

            #write predicted sigma point into right column
            Xsig_pred[0,i] = px_p
            Xsig_pred[1,i] = py_p
            Xsig_pred[2,i] = v_p
            Xsig_pred[3,i] = yaw_p
            Xsig_pred[4,i] = yawd_p
        return Xsig_pred

    def normalize_angle(self, x: float) -> float:
        x = x % (2 * np.pi)    # force in range [0, 2 pi)
        if x > np.pi:          # move to [-pi, pi)
            x -= 2 * np.pi
        return x

    def predict(self, measurment_time: int) -> None:
        """ Predict new object state from previous state with using time differance """
        if measurment_time is None:
            raise MeasurmentTimeNone
        delta_t = measurment_time - self.last_update_time
        if delta_t>1 or delta_t<-1: ### TODO dont comment this without rollback machanisim
            print("time before last update: ", delta_t)
            #print(self._x)
            if delta_t<-0.5:
                raise MeasurmentTimeSmaller
            #delta_t=0.01
        x_aug = np.zeros(self._n_aug)
        try:
            x_aug[0:self.state_dimension] = self._x
            P_aug = np.zeros((self._n_aug,self._n_aug))
            P_aug[0:self.state_dimension,0:self.state_dimension] = self._P
            P_aug[self.state_dimension:,self.state_dimension:] = self._process_R
            Xsig_aug = self.generate_sigma_points(x_aug, P_aug)
            self._Xsig_pred = self.predict_sigma_points(Xsig_aug, delta_t)
            self._x = np.dot(self._Xsig_pred,self._weights)


            self._P.fill(0)
            for i in range(0,self._n_sig):  #iterate over sigma points
                #state difference
                x_diff = self._Xsig_pred[:,i] - self._x
                #angle normalization
                x_diff[3] = self.normalize_angle(x_diff[3])
                self._P = self._P + self._weights[i] * np.matmul(x_diff.reshape(self.state_dimension,1),
                                                                x_diff.reshape(1,self.state_dimension))
            self._last_update_time = measurment_time
        except Exception as e:
            raise e


    def update(self, measurment: AbstractMeasurment) -> None:
        """ Update predicted object state with using new measurment """
        if type(self._Xsig_pred) == type(None):
            print("Error: I(Numan) will look into it, call me if it occurs...")
            return
        Zsig = measurment.convert_tracks_to_sensor_cor_system(self._Xsig_pred)
        # mean predicted measurement
        z_pred= np.matmul(self._weights,Zsig.T)
        # measurement covariance matrix S
        S = np.zeros((measurment.z_dim,measurment.z_dim))
        for i in range(0,self._n_sig):
            z_diff = Zsig[:,i] - z_pred
            if measurment.sensor_obj.sensor_type == SensorTypes.radar_polar:
                z_diff[1] = self.normalize_angle(z_diff[1])
            #print("s",z_diff)
            S = S + self._weights[i]* np.matmul(z_diff.reshape(measurment.z_dim,1),
                                                z_diff.reshape(1,measurment.z_dim))

        # add measurement noise covariance matrix
        S = S + measurment.R

        z = measurment.measurment_matrix
        # create matrix for cross correlation Tc
        Tc = np.zeros((self.state_dimension, measurment.z_dim))

        for i in range(0,self._n_sig):
            # residual
            z_diff = Zsig[:,i] - z_pred
            if measurment.sensor_obj.sensor_type == SensorTypes.radar_polar:
                z_diff[1] = self.normalize_angle(z_diff[1])
            # state difference
            x_diff = self._Xsig_pred[:,i] - self._x
            x_diff[3] = self.normalize_angle(x_diff[3]) #### Buna bak!!
            Tc = Tc+np.dot(self._weights[i], np.matmul(x_diff.reshape(self.state_dimension,1),
                                                       z_diff.reshape(1,measurment.z_dim)))
        # Kalman gain K;
        K = np.matmul(Tc, np.linalg.inv(S))
        # residual
        z_diff = z - z_pred
        # nis_value = z_diff.T@np.linalg.inv(S)@z_diff #TODO remove temp
        # #self.nis[measurment.sensor_type].append(nis_value)
        # # if nis_value>12.5:
        # #     print("measurment.sensor_type:", measurment.sensor_type, nis_value)
        # #     return
        if measurment.sensor_obj.sensor_type == SensorTypes.radar_polar:
                z_diff[1] = self.normalize_angle(z_diff[1])
        #update state mean and covariance matrix
        z_diff = np.dot(K, z_diff)
        self._x = self._x + z_diff
        self._x[3] = self.normalize_angle(self._x[3])
        self._x[4] = self.normalize_angle(self._x[4])
        self._P = self._P - np.dot(K,np.dot(S,K.T))
        #print(self.last_update_state.update_time - measurment.measurment_time)
        if  measurment.measurment_time -self.last_update_state.update_time <0:
            #print(f"it was {self.x_postion, self.y_postion} after update before it was {self.last_update_state.last_x[0], self.last_update_state.last_x[1]}")
            try:
                self.predict(self.last_update_state.update_time)
                # if self.last_update_state.last_mea: #### If the previous measurment is not good that updating with that values again is not a good idea
                #     self.update(self.last_update_state.last_mea)
            except Exception as e:
                raise e
            #print("now: ",self.x_postion, self.y_postion)
        self.last_update_state.update_state(self._P, self._x, measurment, measurment.measurment_time)
