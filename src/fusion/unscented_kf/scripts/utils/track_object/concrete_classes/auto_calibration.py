import numpy as np

from utils.common.enums import SensorTypes

class CalibrationDataCollector:
    """ Collect set of sensor points with using track object measurments.
        One sensor used as ground truth(recieve measurment in tracking cordinate system),
        Other(target_sensor) row measurments(not converted to trakcing cordinate system) collected. 
    """
    def __init__(self, target_sensor: SensorTypes, sensor_in_track_cor: SensorTypes) -> None:
        self.sensor1 = []
        self.sensor2 = []
        self.based_sensor_mea_times= []#to prevent to use same based value 2 times. e.i if the target sensor hz bigger
        self.target_sensor , self.sensor_in_track_cor = target_sensor, sensor_in_track_cor

    def add_measurment_data(self, detected_by_sensors: dict):
        measurment = detected_by_sensors[self.target_sensor]["last_measurment"]
        measurment_in_track_cor = detected_by_sensors[self.sensor_in_track_cor]["last_measurment"]
        if (measurment and measurment_in_track_cor and measurment_in_track_cor.measurment.measurment_time not in self.based_sensor_mea_times):
            if self.target_sensor == SensorTypes.camera:
                self.sensor1.append([round(measurment.measurment.px,2),round(measurment.measurment.py,2),0])
            else:
                self.sensor1.append([round(measurment.measurment.x,2),round(measurment.measurment.y,2),0])
            self.sensor2.append([round(measurment_in_track_cor.measurment_matrix_in_track_cor_system[0],2), round(measurment_in_track_cor.measurment_matrix_in_track_cor_system[1],2),0])
            self.based_sensor_mea_times.append(measurment_in_track_cor.measurment.measurment_time)

class Calibrator():
    """ Calculate 3D-3D cordinate trasformation between sensors.
        Currently only support one sensor at a time !!!
    """
    def __init__(self, target_sensor_id:int) -> None:
        self.sensor1 = []
        self.sensor2 = []
        self.target_sensor_id = target_sensor_id

    def calculate(self):
        R,t = self.rigid_transform_3D(np.array(self.sensor2).T,
                                      np.array(self.sensor1).T)
        print("R:")
        print(R)
        print("t:")
        print(t)

    @staticmethod
    def rigid_transform_3D(A, B):
        """https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py"""
        assert A.shape == B.shape

        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

        num_rows, num_cols = B.shape
        if num_rows != 3:
            raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # ensure centroids are 3x1
        centroid_A = centroid_A.reshape(-1, 1)
        centroid_B = centroid_B.reshape(-1, 1)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        H = Am @ np.transpose(Bm)

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            print("det(R) < R, reflection detected!, correcting for it ...")
            Vt[2,:] *= -1
            R = Vt.T @ U.T

        t = -R @ centroid_A + centroid_B

        return R, t

if __name__ == '__main__':
    x = Calibrator("foo", "bar")
    x.sensor1 = [[110.4, 6.1, 0], [106.87, 5.57, 0], [103.93, 5.9, 0], [97.0, 5.7, 0], [94.67, 6.23, 0], [89.8, 6.23, 0], [78.33, 5.9, 0], [77.3, 6.1, 0], [74.2, 7.1, 0], [70.0, 7.1, 0], [66.36, 6.1, 0], [62.87, 6.17, 0], [63.4, 6.1, 0], [59.5, 6.3, 0], [54.0, 6.3, 0], [51.27, 6.23, 0], [49.35, 6.15, 0], [45.17, 6.3, 0], [43.6, 6.6, 0], [41.2, 6.9, 0], [37.45, 6.5, 0], [35.67, 6.77, 0], [33.6, 6.81, 0], [31.88, 6.54, 0], [32.0, 6.9, 0], [29.31, 6.44, 0], [26.0, 6.5, 0], [26.2, 6.47, 0], [25.0, 6.38, 0], [23.63, 6.84, 0], [24.3, 6.5, 0], [22.4, 6.9, 0], [22.4, 6.9, 0], [22.4, 6.9, 0], [22.4, 6.9, 0], [22.4, 6.9, 0], [22.4, 6.9, 0], [22.4, 6.9, 0], [22.4, 6.9, 0]]
    x.sensor2 = [[109.25, 6.7, 0], [106.75, 7.25, 0], [103.25, 6.5, 0], [95.75, 6.5, 0], [92.75, 6.75, 0], [89.75, 6.5, 0], [76.75, 6.75, 0], [73.25, 6.75, 0], [70.1, 6.54, 0], [66.57, 6.65, 0], [63.75, 6.75, 0], [60.75, 6.5, 0], [57.25, 6.75, 0], [54.25, 6.75, 0], [51.75, 6.75, 0], [48.75, 7.0, 0], [45.75, 6.75, 0], [43.25, 7.0, 0], [40.25, 6.75, 0], [37.75, 7.0, 0], [35.75, 7.0, 0], [33.25, 6.5, 0], [31.75, 7.0, 0], [29.25, 7.0, 0], [27.25, 7.0, 0], [25.25, 7.0, 0], [24.25, 7.25, 0], [22.25, 7.0, 0], [21.25, 7.25, 0], [19.25, 8.0, 0], [18.25, 7.5, 0], [17.25, 7.75, 0], [16.25, 8.0, 0], [15.25, 8.5, 0], [14.25, 9.0, 0], [13.75, 9.5, 0], [13.75, 9.75, 0], [13.75, 10.0, 0], [13.75, 10.25, 0]]
    x.calculate()
    