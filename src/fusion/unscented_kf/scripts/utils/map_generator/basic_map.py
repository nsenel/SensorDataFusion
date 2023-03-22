import numpy as np
import math
import sys
np.set_printoptions(threshold=sys.maxsize)
## Type imports
from utils.track_object.abstract_classes.abstract_tracked_object import AbstractTrack

class CellInfo:
    def __init__(self) -> None:
        self._object_cnt: int = 0
        self._objects_info: dict[int:AbstractTrack] = dict()
    
    def register(self, tracked_obj:AbstractTrack) ->bool:
        """ Register new object in the cell """
        if not tracked_obj.track_id in self._objects_info.keys():
            self._object_cnt += 1
            self._objects_info[tracked_obj.track_id] = tracked_obj
            return True
        print("MAP: register failed")
        return False
    
    def remove(self, obj_id: int) ->bool:
        """ Remove object in the cell """
        if self._objects_info.get(obj_id, False):
            self._object_cnt -= 1
            del self._objects_info[obj_id]
            return True
        print("MAP: remove failed", obj_id, self._objects_info)
        return False
    
    def pop_obj(self, obj_id:int):
        """ Remove object in the cell and return the object """
        if self._objects_info.get(obj_id, False):
            self._object_cnt -= 1
            return self._objects_info.pop(obj_id)
        return None
    
    def add_obj(self, tracked_obj:AbstractTrack):
        """ Function only used to move existed map object in different location in map.
            First registration of object need to be done in self.register()
        """
        if not tracked_obj.track_id in self._objects_info.keys():
            self._object_cnt += 1
            self._objects_info[tracked_obj.track_id] = tracked_obj
            return True
        print(f"MAP: faild to add_obj track_id:{tracked_obj.obj_id}")
        return False
    
    def contains_defined_obj_type(self):
        """ Checks if there is object in the cell with defined type """
        for obj_info in self._objects_info.values():
            if obj_info.track_object_type:
                return True
        return False
    
    def __repr__(self) -> str:
        return f"Number of objects: {self._object_cnt}"


class BasicMap:
    """ Grid map implementation for reducing search space between track object and new detections. """
    def __init__(self, map_l_min: int, map_l_max: int, map_h_min: int, map_h_max: int, cell_l: int, cell_h: int) -> None:
        self._cell_length: int = cell_l
        self._cell_height: int = cell_h
        map_h, map_l = map_h_max-map_h_min, map_l_max-map_l_min
        self.total_row, self.total_column = (math.floor(map_h/cell_h)+1), math.floor(map_l/cell_l)+1

        self._map: dict[tuple:CellInfo] = dict()
        
        for row in range(math.floor(map_h_min/cell_h),math.floor(map_h_max/cell_h)+1):
            for column in range(math.floor(map_l_min/cell_l),math.floor(map_l_max/cell_l)+1):
                self._map[(row, column)] = CellInfo()

        ### For now just for visialization, real process made based on the self._map dictionary (Can be removed)
        self._np_map = np.zeros((self.total_row,self.total_column)) 
        # if map_h_min<=0:
        #     self.np_map_row_index_scaler = len(set([i[0] for i in self._map.keys() if i[0]>0]))
        # else:
        #     self.np_map_row_index_scaler = len(set([i[0] for i in self._map.keys() if i[0]>(map_h/cell_h)+1]))
        self.np_map_row_index_scaler = max(set([i[0] for i in self._map.keys()]))
        ### In order to scale the indexes as: 0 index of the np_map will be furthers point and last row in array will be x=0 meter or incase map includes neagetif x_loc it will be furtherest point in negative direction
        # step x_loc: 5               step: 10              P.S. index is row in the self._np_map
        # index:0 -> x_loc= 20   or   index:0 -> x_loc=  20
        # index:1 -> x_loc= 15   or   index:1 -> x_loc=  10  
        # index:2 -> x_loc= 10   or   index:2 -> x_loc=  0 
        # index:3 -> x_loc= 5    or   index:3 -> x_loc= -10
        # index:4 -> x_loc= 0    or   index:4 -> x_loc= -20
        self.np_map_column_index_scaler = max(set([i[1] for i in self._map.keys()]))
        ### Column wise 0 row index in np_map will be 0 or incase map includes neagetif y_loc it will be furtherest point in negative direction
        # step y_loc: 5               step: 10              P.S. index is column in the self._np_map
        # index:0 -> y_loc= 20  or  index:0 -> y_loc= 20
        # index:1 -> y_loc= 15  or  index:1 -> y_loc= 10  
        # index:2 -> y_loc= 10  or  index:2 -> y_loc=  0 
        # index:3 -> y_loc= 5   or  index:3 -> y_loc= -10
        # index:4 -> y_loc= 0   or  index:4 -> y_loc= -20
        ###

    def get_key(self, object_loc_x: float, object_loc_y: float):
        """ Find key value using object location """
        row, column = math.floor(object_loc_x/self._cell_height),math.floor(object_loc_y/self._cell_length)
        return self.is_key_exist(row, column), (row, column)

    def is_key_exist(self,row, column):
        return (row,column) in self._map.keys()

    def cell_info(self, object_loc_x: float, object_loc_y: float) -> CellInfo:
        return self._map[math.floor(object_loc_x/self._cell_height),math.floor(object_loc_y/self._cell_length)]
    
    def allow_registiration(self, cell_obj, new_obj_type):
        """ Contains rules about which objects can be registered
            Not register objects will also be deleted from track object list
        """
        if (cell_obj._object_cnt<1 or (new_obj_type and cell_obj.contains_defined_obj_type()==False)):
            return True
        return False

    def register_obj(self, object_loc_x: float, object_loc_y: float, tracked_obj) -> bool:
        """ Registers new objects to map if object conditions(def allow_registiration) satisfactory for registration """
        row, column = math.floor(object_loc_x/self._cell_height),math.floor(object_loc_y/self._cell_length)
        if not self.is_key_exist(row,column): return False ### Object is not in map boundry
        
        valid_obj = self.allow_registiration(self._map[(row, column)], tracked_obj.track_object_type)
        if valid_obj and self._map[(row, column)].register(tracked_obj): 
           self._np_map[self.np_map_row_index_scaler-row, self.np_map_column_index_scaler-column] = self._map[(row,column)]._object_cnt
           return True
        return False

    def remove_obj(self, object_loc_x: float, object_loc_y: float, obj_id: int) -> bool:
        row, column = math.floor(object_loc_x/self._cell_height), math.floor(object_loc_y/self._cell_length)
        if self.is_key_exist(row,column) and self._map[(row, column)].remove(obj_id):
            self._np_map[self.np_map_row_index_scaler-row,self.np_map_column_index_scaler-column] = self._map[(row,column)]._object_cnt
            return True
        print("remove_obj fail:", obj_id, row, column)
        return False
    
    def update_obj_location(self, old_loc_x: float, old_loc_y: float,
                            object_loc_x: float, object_loc_y: float, obj_id: int):
        """ Update object location in the map if new location still in the map other case removes object from the map."""
        row_old, column_old = math.floor(old_loc_x/self._cell_height), math.floor(old_loc_y/self._cell_length)
        row_new, column_new = math.floor(object_loc_x/self._cell_height), math.floor(object_loc_y/self._cell_length)
        in_map_boundry = True if self.is_key_exist(row_new,column_new) else False
        if in_map_boundry:
            if (row_old == row_new and column_old == column_new):
                return in_map_boundry, True
            elif self.is_key_exist(row_old,column_old):
                obj = self._map[(row_old,column_old)].pop_obj(obj_id)
                if obj != None:
                    self._np_map[self.np_map_row_index_scaler-row_old,self.np_map_column_index_scaler-column_old] = self._map[(row_old,column_old)]._object_cnt
                    self._map[(row_new,column_new)].add_obj(obj)
                    self._np_map[self.np_map_row_index_scaler-row_new,self.np_map_column_index_scaler-column_new] = self._map[(row_new,column_new)]._object_cnt
                    return in_map_boundry, True
            return in_map_boundry, False
        else:
            return in_map_boundry, self.remove_obj(old_loc_x, old_loc_y, obj_id)
    
    def print_map(self):
        """
        Prints np.array where elements of array is object counts in the cell
               +y(map_l_max)
                     |
                     |
        -100 ------- 0 ------- 100
        """
        print(np.flip(self._np_map.T,axis=1))# in2lab version

        # #### When _np_map not exist dict object can be used.
        # number_of_rows = sorted(set([i[0] for i in self._map.keys()]),reverse=True)
        # #print("rows",len(number_of_rows),self._np_map.shape)
        # number_of_columns = sorted(set([i[1] for i in self._map.keys()]),reverse=False)
        # for row in number_of_rows:
        #     row_values = [self._map[(row,i)]._object_cnt for i in number_of_columns]
        #     print(row_values,row)
    
    def print_occupied_cells(self):
        """ Print cells where one or more object is registered. """
        number_of_rows = sorted(set([i[0] for i in self._map.keys()]),reverse=True)
        number_of_columns = sorted(set([i[1] for i in self._map.keys()]),reverse=False)
        for row in number_of_rows:
            row_values = [(column,self._map[(row,column)]) for column in number_of_columns if self._map[(row,column)]._object_cnt>0]
            if len(row_values):
                for column,value in row_values:
                    print(f"row: {row}, column: {column}, objects: {value._objects_info}")

if __name__ == '__main__':
    class FakeTraackedObj():
        def __init__(self,track_id,track_object_type) -> None:
            self.track_id = track_id
            self.track_object_type = track_object_type

    test = BasicMap(map_l_min=-3, map_l_max=18, map_h_min=50, map_h_max=100, cell_l=3, cell_h=10)
    test.register_obj(object_loc_x = 50,object_loc_y= -1, tracked_obj=FakeTraackedObj(5,None))
    test.register_obj(object_loc_x = 100,object_loc_y= 0, tracked_obj=FakeTraackedObj(5,None))
    test.print_map()
    test.print_occupied_cells()