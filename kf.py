from logging import Handler
import numpy as np

class KF: 
    def __init__(self, init_v, init_pose=0.0, dt = 0.1):
        self.state = [[init_pose], [init_v]]
        self.P = [[0.1, 0], [0, 0.1]]
        self.dt = dt
        self.noise = {'velocity' : 0.2, 'lidar' : 0.5}

    def predict(self, motion):
        fric_coeff = 0.5
        state = self.state
        B = [[0, 0], [0, - fric_coeff/motion]]
        A = [[1, self.dt], [0, 1]]
        
        vel_correction = np.matmul(B, state)
        state += vel_correction
        state[1][0] += np.random.normal(loc=0.0, scale=self.noise['velocity']) # Perhaps add noise to correct

        pred_state = A*state
        #print(motion - pred_state[1][[0]])
        pred_p = np.matmul(np.matmul(A, self.P), np.transpose(A))
        #pred_p = A*self.P*list(np.transpose(A))
        #print(type(self.P))
        pred_p += np.random.normal(loc=0.0, scale=self.noise['velocity'])
        return pred_state, pred_p

    def update(self, measurement, pred): 
        pred_state, pred_p = pred
        measured_pose = 200 - measurement
        measured_pose = [[measured_pose, 0], [0, 0]]
        I = 1

        H = [[1, 0]]
        print(np.transpose(np.matmul(np.array(H), np.array(pred_state))))
        #print(H.shape, pred_state.shape)
        y_k = measured_pose - H*pred_state
        #y_k = measured_pose - np.transpose(np.matmul(np.array(H), np.array(pred_state)))
        S_k = H*pred_p*np.transpose(H) + np.random.normal(loc=0.0, scale=self.noise['lidar'])
        K = pred_p*np.transpose(H)*np.linalg.inv(S_k)

        #print(np.matmul(K, y_k))        
        #print(y_k)

        update_state = pred_state + K*y_k
        update_P = (I - K*H)*pred_p

        #print(update_state)

        return update_state, update_P
        

    def calc(self, state_vector): 
        lidar, v = state_vector
        pred_state, pred_p = self.predict(v)
        pred_input = [pred_state, pred_p]
        self.state, self.P = self.update(lidar, pred_input)
        #print(self.P,  "\n")
        # print(self.state[1][0])

        return self.state[0][0] 