from utils.measurment.abstract_classes.abstract_measurment import AbstractMeasurment

## Typing imports
from utils.sensor.abstract_classes.abstract_sensor import AbstractSensor
import numpy as np

class ParsedMeasurment(AbstractMeasurment):
    def __init__(self, sensor_obj: AbstractSensor, measurment) -> None:
        super().__init__(sensor_obj,measurment)

class ParsedMeasurmentWithGT(AbstractMeasurment):
    def __init__(self, sensor_obj: AbstractSensor, measurment) -> None:
        super().__init__(sensor_obj,measurment)

    @property
    def gt_id(self) -> int:
        return self.measurment.gt_id