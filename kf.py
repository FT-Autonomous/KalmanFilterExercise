from logging import Handler
import numpy as np

class KF: 
    def __init__(self, init_v, init_pose=0.0, dt = 0.1):
        self.state = np.array([[init_pose], [init_v]])
        self.P = np.array([[0.1, 0], [0, 0.1]])
        self.dt = dt
        self.noise = {'velocity' : 0.2, 'lidar' : 0.5}

    def predict(self, motion):
        fric_coeff = 0.5
        state = self.state
        #print('self.state', self.state)
        B = np.array([[0, 0], [0, - fric_coeff/motion]])
        A = np.array([[1, self.dt], [0, 1]])
        
        vel_correction = np.matmul(B, state)
        state += vel_correction
        state[1][0] += np.random.normal(loc=0.0, scale=self.noise['velocity']) # Perhaps add noise to correct

        pred_state = np.matmul(A,state)
        #print('pred_state', pred_state)
        #print(motion - pred_state[1][[0]])
        pred_p = np.matmul(np.matmul(A, self.P), np.transpose(A))
        #pred_p = A*self.P*list(np.transpose(A))
        #print(type(self.P))
        pred_p += np.random.normal(loc=0.0, scale=self.noise['velocity'])
        return pred_state, self.P
        return pred_state, pred_p

    def update(self, measurement, pred): 
        pred_state, pred_p = pred
        measured_pose = 200 - measurement
        measured_pose = np.array([[measured_pose]])

        H = np.array([[1, 0]])
        #print(np.transpose(np.matmul(np.array(H), np.array(pred_state))))
        #print(H.shape, pred_state.shape)
        y_k = measured_pose - np.matmul(H, pred_state)
        #y_k = measured_pose - np.transpose(np.matmul(np.array(H), np.array(pred_state)))
        S_k = np.matmul(np.matmul(H, pred_p), np.transpose(H)) + np.random.normal(loc=0.0, scale=self.noise['lidar'])
        K = np.matmul(np.matmul(pred_p, np.transpose(H)), np.linalg.inv(S_k))

        #K = [[0], [0]]
        #print(np.matmul(K, y_k))        
        #print(y_k)

        update_state = pred_state + np.matmul(K, y_k)
        update_P = np.matmul((np.identity(2) - np.matmul(K, H)), pred_p)

        print('K:', K)
        print('pred_state:', pred_state)
        print('measured_posed', measured_pose)
        print('y_k:', y_k)
        print('update_state:', update_state)

        return update_state, update_P
        

    def calc(self, state_vector): 
        lidar, v = state_vector
        pred_state, pred_p = self.predict(v)
        #self.state, self.P = self.predict(v)
        pred_input = [pred_state, pred_p]
        self.state, self.P = self.update(lidar, pred_input)
        #print(self.P,  "\n")
        # print(self.state[1][0])

        return self.state[0][0] 