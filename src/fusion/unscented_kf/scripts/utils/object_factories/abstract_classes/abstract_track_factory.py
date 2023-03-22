from abc import ABC, abstractmethod

#Typing imports
from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment
from utils.track_object.abstract_classes.abstract_tracked_object import AbstractTrack


class AbstractTrackFactory(ABC):
    def __init__(self, generate_id_from_dict: dict) -> None:
        """Generates new track object
        ID assigment options:
            generate_id_from_dict={"car":{"min":100, "max":999}, "person":{"min":1000, "max":1999}} car ids will start from 100 up to 999 and person will start 1000 up to what ever said in max number
            generate_id_from_dict=False will generate increasing ids from 1 to 255 and than returns back to 1.
            generate_id_from_dict=None will make object based id assigment. Ex: car will start from 1 also person will start from one. so there might be same id for both object in sceen in same time.
        """
        self.use_object_type_for_id = True
        self.not_classified_obj_id = 0
        if generate_id_from_dict == False:
            # self.generate_id_from_dict["object_type_indepented_id"]={"min":1, "max":255, "current":1}
            self.use_object_type_for_id = False
            self.current_id = 0
        else:
            self.generate_id_from_dict = generate_id_from_dict if generate_id_from_dict != None else {}
            ## Put current key to dict for increasing with generate_new_track_id
            if len(self.generate_id_from_dict)>0:
                for key in self.generate_id_from_dict.keys():
                    self.generate_id_from_dict[key]["current"] = self.generate_id_from_dict[key]["min"]
    
    def generate_new_track_id(self, object_type: str) -> int:
        if object_type is None:
            self.not_classified_obj_id -= 1
            new_id = self.not_classified_obj_id
            if new_id==-999:
                self.not_classified_obj_id=0
            return new_id
        if self.use_object_type_for_id:
            if object_type in self.generate_id_from_dict:
                new_id = self.generate_id_from_dict[object_type]["current"] + 1
                self.generate_id_from_dict[object_type]["current"] = new_id if new_id != self.generate_id_from_dict[object_type]["max"] else self.generate_id_from_dict[object_type]["min"]
            else:
                self.generate_id_from_dict[object_type] = {"min":1, "max":255, "current":1}
                new_id = 1
        else:
            self.current_id += 1
            new_id = self.current_id
            if new_id==255:
                self.current_id=0
        return new_id

    @abstractmethod
    def generate_track_obj(self) -> AbstractTrack:
        pass

    @abstractmethod
    def reset_tracking_settings(self, tracked_obj:AbstractTrack, object_type: str) -> AbstractTrack:
        pass