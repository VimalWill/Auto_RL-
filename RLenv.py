import glob 
import carla 
import numpy 
import time 
import sys 
import os 

SHOW_PREVIEW = False
IM_WIDTH = 640 
IM_HIEGHT = 480 

class wl_env:
    SHOW_CAM = SHOW_PREVIEW 
    STEER_AMT = 1.0 
    im_width = IM_WIDTH
    im_height = IM_HIEGHT
    front_camera = None 
    
    def __init__(self):
        self.client = carla.client("localhost",2000)
        self.client.set_timeout()
        
