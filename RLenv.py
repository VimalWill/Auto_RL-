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
        self.imu_hist = []
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3,self.transform)
        self.actor_list.append(self.vehicle)

        transform_s = carla.Transform(carla.Location(z=2))
        imu_sensor = self.blueprint_library.find("sensor.other.imu")
        self.imu_sensor = self.world.spawn_actor(imu_sensor,transform_s,attach_to = self.vehicle)
        

