from logging import Handler
import numpy as np

class KF: 
    def __init__(self, init_v, init_pose=0.0, dt = 0.1):
        self.state = np.array([[init_pose], [init_v]])
        self.P = np.array([[1, 0], [0, 1]])
        self.dt = dt
        self.noise = {'velocity' : 0.2, 'lidar' : 0.5}

    def predict(self, motion):
        fric_coeff = 0.5
        state = self.state
        B = np.array([[0, 0], [0, - fric_coeff/motion]])
        A = np.array([[1, self.dt], [0, 1]])
        
        vel_correction = np.matmul(B, state)
        state += vel_correction
        state[1][0] += np.random.normal(loc=0.0, scale=self.noise['velocity']) # Perhaps add noise to correct

        pred_state = np.matmul(A, state)
        # print("before pred_p")
        # print("np.matmul(A, self.P):", np.matmul(A, self.P))
        pred_p = np.matmul(np.matmul(A, self.P), np.transpose(A))
        #print("after pred_p...")
        #pred_p += np.random.normal(loc=0.0, scale=self.noise['velocity'])
        #print('pred_p : ', pred_p)
        return pred_state, pred_p

    def update(self, measurement, pred): 
        pred_state, pred_p = pred
        measured_pose = 200 - measurement
        measured_pose = np.array([[measured_pose]])

        H = np.array([[1, 0]])
        y_k = measured_pose - np.matmul(H, pred_state)

        #print("np.matmul(H, pred_p): ", (np.matmul(H, pred_p)))
        #print(np.transpose(H))

        print(H.shape, pred_p.shape)

        S_k = np.matmul(np.matmul(H, pred_p), np.transpose(H)) + abs(np.random.normal(loc=0.0, scale=self.noise['lidar']))
        print("S_k: ", S_k)
        print("top_bit:", np.matmul(pred_p, np.transpose(H)))
        K = np.matmul(pred_p, np.transpose(H))@np.linalg.inv(S_k)
        print("K:", K)

        #av = np.average(np.array([self.noise["velocity"], self.noise['lidar']]))
        #K = [[av], [av]]
        print("-------------------------")

        update_state = pred_state + np.matmul(K, y_k)
        update_P = np.matmul((np.identity(2) - np.matmul(K, H)), pred_p)

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