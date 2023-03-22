## This file contains rapers for incoming measurments ##

class LazyLoader:
    """ This class is used as a holder for transformed measurements to other coordinate systems.
        It is used to not calculate object location in the cost function coordinate system multiple times
        (Depending on the number of track objects "multiple times" can reach to couple hundreds)
    """
    __slots__ = ("_x_transformed_cor", "_y_transformed_cor", "_z_transformed_cor")
    def __init__(self):
        self._x_transformed_cor: int = None
        self._y_transformed_cor: int = None
        self._z_transformed_cor: int = None
    
    @property
    def transformed_cordinates(self):
        return (self._x_transformed_cor,self._y_transformed_cor)
    
    @property
    def transformed_cordinates_setted(self):
        """ Check if measurment already transform"""
        return self._x_transformed_cor is not None and self._y_transformed_cor is not None
    
    def set_transformed_cordinates(self, transform_points:list): 
        """[x,y] optional [x,y,z]"""
        self._x_transformed_cor = transform_points[0]
        self._y_transformed_cor = transform_points[1]
        if len(transform_points) == 3:
            self._z_transformed_cor = transform_points[2]

class InitialCameraMeasurment(LazyLoader):
    __slots__  = ("px","py", "bbox","obj_name","measurment_time")
    def __init__(self, px:int, py:int, obj_name:str, measurment_time:int, bbox:dict=None) -> None:
        super().__init__()
        self.px:int = px
        self.py:int = py
        self.obj_name:str = obj_name
        self.measurment_time:int = measurment_time
        self.bbox:dict = bbox

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self,key)}' for key in self.__slots__])})"

class InitialRadarMeasurment(LazyLoader):
    __slots__  = ("range","bearing","range_rate","obj_name","measurment_time")
    def __init__(self, range:float, bearing:float, range_rate:float,
                 obj_name:str, measurment_time:int) -> None:
        super().__init__()
        self.range:float = range
        self.bearing:float = bearing
        self.range_rate:float = range_rate
        self.obj_name:str = obj_name
        self.measurment_time:int = measurment_time
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self,key)}' for key in self.__slots__])})"

class InitialCartesianRadarMeasurment(LazyLoader):
    __slots__  = ("x","y","v","obj_name","measurment_time")
    def __init__(self, x:float, y:float, v:float, obj_name:str,
                 measurment_time:int) -> None:
        super().__init__()
        self.x:float = x
        self.y:float = y
        self.v:float = v
        self.obj_name:str = obj_name
        self.measurment_time:int = measurment_time
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self,key)}' for key in self.__slots__])})"

class InitialLidarMeasurment(LazyLoader):
    __slots__  = ("x","y","obj_name","measurment_time", "z")
    def __init__(self, x:float, y:float, obj_name:str, measurment_time:int, z:float=0.0) -> None:
        super().__init__()
        self.x:float = x
        self.y:float = y
        self.obj_name:str = obj_name
        self.measurment_time:int = measurment_time
        self.z:float = z
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self,key)}' for key in self.__slots__])})"

class CameraMeasurmentwithGT(LazyLoader):
    __slots__  = ("px","py","measurment_time","obj_name", "x_gt", "y_gt",
                  "vx_gt","vy_gt","yaw_gt","yawrate_gt", "gt_id", "bbox")
    def __init__(self, px:int, py:int, measurment_time:int, obj_name:str, x_gt:float,
                y_gt:float, vx_gt:float, vy_gt:float, yaw_gt:float, yawrate_gt:float,
                gt_id:int, bbox:dict=None) -> None:
        super().__init__()
        self.px:int = px
        self.py:int = py
        self.measurment_time:int = measurment_time
        self.obj_name:str = obj_name
        self.x_gt:float = x_gt
        self.y_gt:float = y_gt
        self.vx_gt:float = vx_gt
        self.vy_gt:float = vy_gt
        self.yaw_gt:float = yaw_gt
        self.yawrate_gt:float = yawrate_gt
        self.gt_id:int = gt_id
        self.bbox = bbox
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self,key)}' for key in self.__slots__])})"

class RadarMeasurmentwithGT(LazyLoader):
    __slots__  = ("range","bearing","range_rate","measurment_time", "obj_name", "x_gt",
                  "y_gt","vx_gt","vy_gt","yaw_gt", "yawrate_gt", "gt_id")
    def __init__(self, range:float, bearing:float, range_rate:float, measurment_time:int,
                 obj_name:str, x_gt:float, y_gt:float, vx_gt:float, vy_gt:float, yaw_gt:float,
                 yawrate_gt:float, gt_id:int) -> None:
        super().__init__()
        self.range:float = range
        self.bearing:float = bearing
        self.range_rate:float = range_rate
        self.measurment_time:int = measurment_time
        self.obj_name:str = obj_name
        self.x_gt:float = x_gt
        self.y_gt:float = y_gt
        self.vx_gt:float = vx_gt
        self.vy_gt:float = vy_gt
        self.yaw_gt:float = yaw_gt
        self.yawrate_gt:float = yawrate_gt
        self.gt_id:int = gt_id
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self,key)}' for key in self.__slots__])})"

class LidarMeasurmentwithGT(LazyLoader):
    __slots__  = ("x","y","measurment_time","obj_name", "x_gt", "y_gt",
                  "vx_gt","vy_gt","yaw_gt","yawrate_gt", "gt_id", "z")
    def __init__(self, x:float, y:float, measurment_time:int, obj_name:str, x_gt:float,
                 y_gt:float, vx_gt:float, vy_gt:float, yaw_gt:float, yawrate_gt:float,
                 gt_id:int, z:float=0) -> None:
        super().__init__()
        self.x:float = x
        self.y:float = y
        self.measurment_time:int = measurment_time
        self.obj_name:str = obj_name
        self.x_gt:float = x_gt
        self.y_gt:float = y_gt
        self.vx_gt:float = vx_gt
        self.vy_gt:float = vy_gt
        self.yaw_gt:float = yaw_gt
        self.yawrate_gt:float = yawrate_gt
        self.gt_id:int = gt_id
        self.z:float = z
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self,key)}' for key in self.__slots__])})"

class EgoCarOdomMessage:
    __slots__  = ("x","y","timestamp", "yaw", "vx", "vy")
    def __init__(self, x:float, y:float, timestamp:int, yaw:float,
                 vx:float, vy:float) -> None:
        self.x:float = x
        self.y:float = y
        self.timestamp:int = timestamp
        self.yaw:float = yaw
        self.vx:float = vx
        self.vy:float = vy
    def __repr__(self) -> str:
            return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self,key)}' for key in self.__slots__])})"

class TrackedObject:
    """It can be used independed from track object state(object state can have additional values like velocity angle etc.)"""
    __slots__  = ("x","y","z")
    def __init__(self, x:float, y:float, z:float = 0) -> None:
        self.x:float = x
        self.y:float = y
        self.z:float = z
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join([f'{key}={getattr(self,key)}' for key in self.__slots__])})"

#### This is slower than regular class
#from dataclasses import dataclass

# @dataclass(frozen=True)
# class InitialCameraMeasurment:
#     px:int
#     py:int
#     obj_name:str
#     measurment_time:int
#     gt_id:int

# @dataclass(frozen=True)
# class InitialRadarMeasurment:
#     range:float
#     bearing:float
#     range_rate:float
#     obj_name:str
#     measurment_time:int
#     gt_id:int

# @dataclass(frozen=True)
# class InitialCartesianRadarMeasurment:
#     x:float
#     y:float
#     v:float
#     obj_name:str
#     measurment_time:int
#     gt_id:int

# @dataclass(frozen=True)
# class InitialLidarMeasurment:
#     x:float
#     y:float
#     obj_name:str
#     measurment_time:int
#     gt_id:int
#     z:float = 0

# @dataclass(frozen=True)
# class CameraMeasurmentwithGT:
#     px:int
#     py:int
#     measurment_time:int
#     obj_name:str
#     x_gt:float
#     y_gt:float
#     vx_gt:float
#     vy_gt:float
#     yaw_gt:float
#     yawrate_gt:float
#     gt_id:int

# @dataclass(frozen=True)
# class RadarMeasurmentwithGT:
#     range:float
#     bearing:float
#     range_rate:float
#     measurment_time:int
#     obj_name:str
#     x_gt:float
#     y_gt:float
#     vx_gt:float
#     vy_gt:float
#     yaw_gt:float
#     yawrate_gt:float
#     gt_id:int


# @dataclass(frozen=True)
# class LidarMeasurmentwithGT:
#     x:float
#     y:float
#     measurment_time:int
#     obj_name:str
#     x_gt:float
#     y_gt:float
#     vx_gt:float
#     vy_gt:float
#     yaw_gt:float
#     yawrate_gt:float
#     gt_id:int

# @dataclass(frozen=True)
# class EgoCarOdomMessage:
#     x:float
#     y:float
#     timestamp:int
#     yaw:float
#     vx:float
#     vy:float

# @dataclass(frozen=True)
# class TrackedObject:
#     """It can be used independed from track object state(object state can have additional values like velocity angle etc.)"""
#     x:float
#     y:float
#     z:float = 0