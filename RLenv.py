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


class RLenv:
    SHOW_CAM = SHOW_PREVIEW 
    STEER_AMT = 1.0 
    im_width = IM_WIDTH
    im_height = IM_HEIGHT 
    front_camera = None 

    def __init__(self):

        #init the comm with carla env 
        self.client = carla.Client("localhost",2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.mode_3 = self.blueprint_library.filter("model3")[0]

    def reset(self):
        self.action_list = []
        self.gnss_hist = []

        #desinging the environment 
        self.transform = random.choice(self.world.get_map().get_spawn_point())
        self.vehicle = self.world.spwan_actor(self.model_3,self.transform)

        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.blueprint_library.find("sensor.camera.rgb")
        self.rgb.set_attribute("image_size_x",f"{self.im_width}")
        self.rgb.set_attribute("image_size_y",f"{self.im_height}")
        self.rgb.set_attribute("fov",f"110")

        transform = carla.Transform(carla.Location(x=2.5,z=0.7))
        self.sensor = self.spawn_actor(self.rgb_cam,transform,attach_to = self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data:self.process_img(data))

        self.vehicle.apply_control(throttle = 0.0,brake = 0.0)
        time.sleep(4)

        #adding to gnss to the vechile 
        gnss_sen = self.blueprint_library.find("sensor.other.gnss")
        self.gnss_sen = self.world.spawn_actor(gnss_sen,transform,attach_to = self.vehicle)
        self.gnss_sen.listen(lambda event: self.gnss_sen(event))
        
        while self.front_camera is None:
            time.sleep(0.01)
        
        self.episode_start = time.time()
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0,brake=0.0))
        return self.front_camera

    def gnss_data(self,data):
        self.gnss_hist.append(data)

    def process_img(self,image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:,:,:3]
        if self.SHOW_CAM:
            cv2.imshow("",i3)
            cv2.waitkey(1)
        self.front_camera = i3

    def set(self,action):
        if action:
            pass 