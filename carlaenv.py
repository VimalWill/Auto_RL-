import carla 
from gym import Env
from gym.spaces import Box, Discrete
import numpy as np 
import glob 
import sys 
import time 
import random 
import cv2 
import os

import tensorflow as tf 
from tensorflow import keras
from keras import Model
from keras.layers import Dense, Activation, Input
from keras.callbacks import TensorBoard
from keras.applications import ResNet50

from rl.agents import DQNAgent
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

SHOW_PREVIEW = False
ACTIONS  = 3
IM_WIDTH =  640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 60 

class carlaEnv(Env):
    metadata = {'render.modes': ['human']}

    SHOW_CAM = SHOW_PREVIEW
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    STEER_AMT = 1.0
    front_camera = None

    def __init__(self):
        super().__init__()

        #number of actions (Straigth, Left and Right)
        self.action_space = Discrete(ACTIONS)
        self.image_shape = (self.im_height,self.im_width,3)
        self.observation_space = Box(low = 0, high=255,shape=self.image_shape)

        #init the carla 
        self.client = carla.Client('localhost',2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

    def step(self,action):
        
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer = -1*self.STEER_AMT))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer = 0))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0,steer= 1*self.STEER_AMT))
        
        #calculating the acceleration 
        accel = self.data_dict['imu']['accel'] - carla.Vector3D(x=0,y=0,z=9.81)
        accel_mag = accel.length()
        
        #computing reward 
        if len(self.collision_hist) == 0 and (accel_mag < 50 and accel_mag > 35):
            Done = False 
            Reward = 5

        elif self.episode_start + SECONDS_PER_EPISODE < time.time():
            Done = True 
            Reward = 2

        else:
            Done = False 
            Reward = -10

        info = {}

        return self.front_camera,Reward, Done, info #observation, reward, status, extra information 
        

    def render(self, mode='human'):
        return #nothing

    def reset(self):
        self.data_dict = dict()
        self.actor_list = []
        self.collision_hist  = []

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

        self.vehicle.apply_control(carla.Vehiclecontrol(throttle=0.0,brake = 0.0))
        time.sleep(4.0)

        collision_sensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor,transform_c,to_attach = self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.collision_callback(event))

        transform_s = carla.Transform(carla.Location(z=2))
        imu_sensor = self.blueprint_library.find("sensor.other.imu")
        self.imu_sensor = self.world.spawn_actor(imu_sensor,transform_s,attach_to = self.vehicle)
        self.actor_list.append(self.imu_sensor)
        self.imu_sensor.listen(lambda data: self.imu_callback(data))

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()
        self.vehicle.apply_control(carla.Vehiclecontrol(throttle=0.0,brake=0.0))

        return self.front_camera #observation 

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
    
    def collision_callback(self,event):
        self.collision_hist.append(event)

class ModifiedTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, name)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

class DqlModel():
    def __init__(self):
        self.state = carlaEnv().observation_space.shape 
        self.action = carlaEnv().action_space.n 
        self.tensorboard = ModifiedTensorBoard(log_dir="./logs")

    def cnn_model(self):

        #using Resnet pretrained model 
        self.base_model = ResNet50(input_tensor=self.state,weights="imagenet",include_top=False)
        self.x = self.base_model.output
        self.x = Dense(64, Activation = "relu")(self.x)
        self.x = Dense(32, Activation = "relu")(self.x)
        self.x = Dense(64, Activation = "relu")(self.x)

        self.output_layer = Dense(self.action,Activation='linear')(self.x)
        self.model = Model(input = self.base_model.input, output = self.output_layer)

    def DqnModel(self):
        self.policy = BoltzmannGumbelQPolicy()
        self.memory = SequentialMemory(limit = 50000,window_length = 1)
        
        dqn = DQNAgent(model = self.model, memory = self.memory,policy=self.policy, 
        nb_actions = self.action,nb_steps_warmup=10,target_model_update = 1e-2)

        return dqn
        
     

    