from logging import Handler
import numpy as np


class KF:

    def __init__(self, init_v, init_pose=0.0, dt = 0.1):
        self.state = np.array([[init_pose], [init_v]])
        self.P = np.array([[1, 0], [0, 1]])
        self.dt = dt
        self.noise = {'velocity' : 0.2, 'lidar' : 0.5}


    def predict(self, motion):


        return pred_state, pred_p
    

    def update(self, measurement, pred):


        return update_state, update_P
    

    def calc(self, state_vector):


        return self.state[0][0]
