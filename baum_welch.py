import numpy as np
from numpy import ndarray, zeros
'''
The Baum–Welch Algorithm
'''

class BaumWeltch():
    """
    S: State transition probability matrix
    O: Output emission probability matrix
    pi: Initial state probability vector
    --------------------------------------
    obs_seq: list of observations
    num_obs: number of observations
    num_state: number of states
    """

    def __init__(self, S, O, start_value=1, obs_seq=None):
        self.S = S
        self.O = O
        self.start_value = start_value
        self.obs_seq = obs_seq
        self.num_state = self.S.shape[0]
        self.num_obs = 9 #self.O.shape[0]

        print('here')

    def forward(self):
        print('Forward Alg')

        # matrix -> num_state x num_obs
        a = zeros((self.num_obs+2, self.num_state))

        # Initializes a[*][START] to self.start_value
        a[:, 0] = self.start_value

        print(a)
        for i in range(1, self.num_obs):
            for k in range(self.num_state):
                # probability_objs = self.O[k, self.obs_seq[i]]
                a[i,k] += np.dot(a[i-1, :], (self.S[:,k])) # * p_objs

        print(a)
        return a

    def backward(self):
        print('Backward Alg')


transition = [[.7,.2,.1], [.15,.8,.05], [.6,.35,.05]]
emission = [[.7,.2,.05,.05], [.2,.6,.1,.1]]
observations = []

BaumWeltch(np.array(transition), emission).forward()
# BaumWeltch(np.ndarray(transition), np.ndarray(emission)).forward()