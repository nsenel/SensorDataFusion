import unittest
from utils.measurment.concrete_classes.incoming_measurments import EgoCarOdomMessage
from utils.movement_compensation.concrete_classes.movement_compensation_with_global_change import MVWithGlobalPostionChange
from utils.movement_compensation.concrete_classes.movement_compensation_with_speed import MVWithSpeed

class FakeRosTimeObj():
    def __init__(self,nmr) -> None:
        self.nmr = nmr
    def to_sec(self):
        return self.nmr
    def __sub__(self, other):
        if isinstance(other, FakeRosTimeObj):
            return FakeRosTimeObj(self.nmr - other.nmr)
        return (self.nmr-other)

class TestMVWithGlobalPostionChange(unittest.TestCase):
    def setUp(self):
        self.movement_compensator = MVWithGlobalPostionChange()
    
    def test_init_properties(self):
        self.assertEqual(self.movement_compensator._last_odom_message, None)
        self.assertEqual(self.movement_compensator._change_from_last_update, {"x":0, "y":0, "total_yaw_change":0})
    
    def test_update_and_movement_compensation(self):
        self.movement_compensator.update_state(EgoCarOdomMessage(x=9,y=1,timestamp=5,yaw=0,vx=1,vy=0))
        self.movement_compensator.update_state(EgoCarOdomMessage(x=10,y=2,timestamp=6,yaw=0,vx=1,vy=1))
        moved_in_x, moved_in_y= self.movement_compensator.get_movement_compensation(track_x=0, track_y=0, measurment_time=7, last_update_time=6)
        self.assertEqual(moved_in_x, 1)
        self.assertEqual(moved_in_y, 1)
        moved_in_x, moved_in_y= self.movement_compensator.get_movement_compensation(track_x=0, track_y=0, measurment_time=7, last_update_time=7)
        self.assertEqual(moved_in_x, 0)
        self.assertEqual(moved_in_y, 0)
    
    def test_update_and_movement_compensation_with_angle(self):
        self.movement_compensator.update_state(EgoCarOdomMessage(x=9,y=1,timestamp=5,yaw=0,vx=1,vy=0))
        self.movement_compensator.update_state(EgoCarOdomMessage(x=10,y=2,timestamp=6,yaw=0.2,vx=1,vy=1))
        moved_in_x, moved_in_y= self.movement_compensator.get_movement_compensation(track_x=10, track_y=10, measurment_time=7, last_update_time=6)
        self.assertEqual(moved_in_x, 1)
        self.assertEqual(moved_in_y, -1.1860275295381975)

class TestMVWithSpeed(unittest.TestCase):
    def setUp(self):
        self.movement_compensator = MVWithSpeed()
    
    def test_init_properties(self):
        self.assertEqual(self.movement_compensator._last_odom_message, None)
    
    def test_update(self):
        self.movement_compensator.update_state(EgoCarOdomMessage(x=9,y=1,timestamp=FakeRosTimeObj(5),yaw=0,vx=1,vy=0))
        self.movement_compensator.update_state(EgoCarOdomMessage(x=10,y=2,timestamp=FakeRosTimeObj(6),yaw=0,vx=1,vy=1))
        moved_in_x, moved_in_y= self.movement_compensator.get_movement_compensation(track_x=0, track_y=0, measurment_time=7, last_update_time=6)
        self.assertEqual(moved_in_x, 1)
        self.assertEqual(moved_in_y, 1)
        moved_in_x, moved_in_y= self.movement_compensator.get_movement_compensation(track_x=0, track_y=0, measurment_time=7, last_update_time=7)
        self.assertEqual(moved_in_x, 0)
        self.assertEqual(moved_in_y, 0)
    
    def test_update_and_movement_compensation_with_angle(self):
        self.movement_compensator.update_state(EgoCarOdomMessage(x=9,y=1,timestamp=FakeRosTimeObj(5),yaw=0,vx=1,vy=0))
        self.movement_compensator.update_state(EgoCarOdomMessage(x=10,y=2,timestamp=FakeRosTimeObj(6),yaw=0.2,vx=1,vy=1))
        moved_in_x, moved_in_y= self.movement_compensator.get_movement_compensation(track_x=10, track_y=10, measurment_time=FakeRosTimeObj(7), last_update_time=6)
        self.assertEqual(moved_in_x, 1)
        self.assertEqual(moved_in_y, -1.1860275295381975)
