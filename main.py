from tabnanny import verbose
from carlaenv import carlaEnv, DqlModel
from keras.optimizers import Adam

def main():
    
    """
    -> import the env and DQL agent
    -> compile the DQL model 
    -> fit the env and actions  

    """       

    #init the carla env 
    env = carlaEnv()

    #calling the DQL model 
    dql = DqlModel.DqnModel()

    dql.compile(Adam(lr=1e-3),metrics=['mae'])
    dql.fit(env,nb_steps=50000,visualize = True, verbose=1)

    

if __name__ == "__main__":
    main()

