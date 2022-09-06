import carla 
import math 
import numpy as np 
import glob 
import sys 
import time 
import random 
import cv2 
import os 

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

SHOW_PREVIEW = False 
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10

class env():
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self):
        #init the carla env 
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(1.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library 
        self.model_3 = self.blueprint_library.filter("model3")[0]
    
    def reset(self):
        self.data_dict = dict()
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3,self.transform)
        self.actor_list.append(self.vehicle)

        transform_c = carla.Transform(carla.Location(x=2,z=0.7))
        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb.set_attribute("image_size_x",f"{self.im_width}")
        self.rbg.set_attribute("image_size_y",f"{self.im_height}")
        self.rgb.set_attribute("fov",f"110")
        self.cam_Sensor = self.world.spawn_actor(self.rgb_cam,transform_c,attach_to=self.vehicle)
        self.actor_list.append(self.cam_Sensor)
        self.cam_Sensor.listen(lambda data_c: self.camera_callback(data_c))

        transform_s = carla.Transform(carla.Location(z=2))
        imu_sensor = self.blueprint_library.find("sensor.other.imu")
        self.imu_sensor = self.world.spawn_actor(imu_sensor,transform_s,attach_to = self.vehicle)
        self.actor_list.append(self.imu_sensor)
        self.imu_sensor.listen(lambda data: self.imu_callback(data))
    
    
    def imu_callback(self,data):
        self.data_dict['imu'] = {
            'gyro' : data.gyroscope,
            'accel' : data.accelerometer,
            'compass': data.compass

        }

    def camera_callback(self,data_c):
        i = np.array(data_c.raw_data)
        i2 = i.reshape((self.im_height,self.im_width,4))
        i3 = i2[:,:,:3]
        if self.SHOW_CAM:
            cv2.imshow("",i3)
            cv2.waitKey(1)
        self.front_camera = i3 

            

