class MeasurmentTimeSmaller(Exception):
    """Raise when new measurment timestamp is smaller than old meaurment time stamp"""
    def __init__(self, track_object_id: int=-1) -> None:
        self.track_object_id = track_object_id
        self.obj_memory_address = None
    def __str__(self):
        return "Given measurment time older than last update time ..."
        # You can fill the values as belove
        # if isinstance(exception_type,PMatrixHasNotPossitiveDefine) and exception_type.track_object_id==-1: ## -1 means is not defined
        #             exception_type.track_object_id = track_obj.track_id
        #             exception_type.obj_memory_address = id(track_obj)

class MeasurmentTimeNone(Exception):
    """Raise when new measurment timestamp is None"""
    def __str__(self):
        return "No given measurment time ..."

class PMatrixHasNotPossitiveDefine(Exception):
    """Raise when given matrix doestn cholesky decomposition"""
    def __init__(self, track_object_id: int=-1) -> None:
        self.track_object_id = track_object_id
        ## This is used for deleting objects track_object_id cant be use due to fact that same object id can be exist in track object(car object can have id 5 in sametime person object also can have id 5 depending on id generation function)
        self.obj_memory_address = None 
    def __str__(self):
        return "generate_sigma_points cant find reverse matrix for given P has no positive definite .... track_id: " + str(self.track_object_id)