import enum

class ObjectTypes(enum.Enum):
    """ COCO ids"""
    person = 0
    bicycle = 1
    car = 2
    motorcycle = 3
    bus = 5
    truck = 7

class SensorTypes(enum.Enum):
    camera = 1
    lidar = 2
    radar_cartesian = 3
    radar_polar = 4
    fisheye = 5


if __name__ == "__main__":
    # printing all enum members using loop
    print ("All the enum values are : ")
    for sensor in (SensorTypes):
        print(sensor.value)
        if sensor==SensorTypes.camera:
            print(sensor.name) # If you need to access the type as name
    for obj in (ObjectTypes):
        print(obj.name, obj.value)
    print(ObjectTypes(2))