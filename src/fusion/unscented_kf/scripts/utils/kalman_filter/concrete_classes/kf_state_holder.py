
class KFStateHolder():
    def __init__(self, p, x, measurment, update_time):
        self.p = p
        self.x = x
        self.mea = measurment
        self.update_time = update_time

    def update_state(self, p, x, measurment, update_time):
        self.p = p.copy()
        self.x = x.copy()
        self.mea = measurment
        self.update_time = update_time