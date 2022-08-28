#importing the requirenments
from airsim import CarClient, CarControls, ImageRequest, ImageType
from configparser import ConfigParser
import numpy as np 
from numpy.linalg import norm
from os.path import dirname, abspath, join

#initalization of the class 
class WC_Agent():
    #connect to the AirSim Simulator 
    super().__init__()
    super().confirmConnection()
    super().enableApiControl(True)

    config = ConfigParser()
    config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))
    print("testing")

