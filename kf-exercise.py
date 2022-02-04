import math
import matplotlib.pyplot as plt
import numpy as np 
from kf import *

COEFF_FRIC = 0.5

def sim(init_v=15):
    pose = 0.0
    lidar = 200 - pose
    v = init_v
    dt = 0.1

    kf = KF(v, pose, dt)

    x_true = [0.0]
    x_measured = [0.0]
    x_pred = [0.0]
    v_arr = [v]

    while v > 0:
        v -= COEFF_FRIC
        pose = pose + v*dt
        lidar = 200 - x_measured[-1]
        
        x_true.append(pose)
        pose_w_noise = pose + np.random.normal(loc=0.0, scale=0.5)
        x_measured.append(pose_w_noise)

        v_w_noise = v + np.random.normal(loc=0.0, scale=0.2)
        state_vector = [lidar, v_w_noise]

        x_pred.append(kf.calc(state_vector))

        v_arr.append(v_w_noise )

    return x_true, x_measured, x_pred, v_arr

if __name__ == '__main__':
    true, measured, pred, v_arr = sim()

    x = np.arange(len(true))

    plt.plot(x, true)
    plt.plot(x, measured)
    plt.plot(np.array(x), pred)
    plt.legend(("True", "Measured", "Pred"))
    plt.show()

    # plt.plot(x, v_arr)
    # plt.show()
