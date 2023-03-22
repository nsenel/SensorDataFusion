from abc import ABC, abstractmethod

class AbstractCostCalculator(ABC):
    """ Defines required methods properties of valid cost calculator """
    @abstractmethod
    def calculate_cost(tracks, detections):
        pass