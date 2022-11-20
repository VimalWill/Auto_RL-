from CarlaEnv import Carlaeni
from keras.optimizers import Adam
import tensorflow as tf
from tensorflow import keras
from keras import Model
from tensorboard import ModifiedTensorBoard
from keras.layers import Dense, Activation, Input, GlobalAveragePooling2D
from keras.applications import ResNet50
import cv2 

from rl.agents import DQNAgent
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory



state = Carlaeni().observation_space.shape
action = Carlaeni().action_space.n
        
def cnn_model(state,action):
    base_model = ResNet50(
    input_shape=state, weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(64, activation="relu")(x)
    output_layer = Dense(action, activation='softmax')(x)
    model = Model(inputs=base_model.input,outputs=output_layer)

    return model 

model = cnn_model(state,action)
    
def main():
    
    """
    -> import the env and DQL agent
    -> compile the DQL model 
    -> fit the env and actions  
    -> test the with n_esp = 50k, 100k and 150k

    """       

    #init the carla env 
    env = carlaEnv()
    #ensuring the working 
    current_state = env.reset()
    cv2.imshow(f'Agent - preview', current_state)
    cv2.waitKey(2)


if __name__ == "__main__":
    main()

