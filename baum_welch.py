import numpy as np
from numpy import array, zeros

'''
The Baum–Welch Algorithm
'''
class BaumWeltch():
    """
    S: State transition probability matrix
    O: Output emission probability matrix
    start_value: start state value
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
        self.num_obs = len(self.O)

    # Forward algorithm
    def forward(self):
        print('Computing Forward Algorithm...')

        # matrix -> num_state x num_obs
        a = zeros((self.num_obs, self.num_state))
        # Initializes a[*][START] to self.start_value
        a[:, 0] = self.start_value

        for i in range(1, self.num_obs):
            for k in range(self.num_state):
                obs_probability = self.O[k, self.obs_seq[i]]
                a[i,k] += np.dot(a[i-1, :], (self.S[:, k])) * obs_probability

        print(a)
        return a

    # Backward algorithm
    def backward(self):
        print('Computing Backward Algorithm...')

        # matrix -> num_state x num_obs
        a = zeros((self.num_obs, self.num_state))
        # Initializes a[*][END] to self.start_value
        a[:, -1] = self.start_value

        for i in reversed(range(self.num_obs-1)):
            for k in range(self.num_state):
                obs_probability = self.O[:, self.obs_seq[i+1]]
                a[i, k] = np.sum(a[i+1, :] * self.S[k, :]) * obs_probability
        return a


transition = [[.7,.2,.1], [.15,.8,.05], [.6,.35,.05]]
emission = [[.7,.2,.05,.05], [.2,.6,.1,.1]]
observations = []

BaumWeltch(array(transition), array(emission)).forward()
# BaumWeltch(np.ndarray(transition), np.ndarray(emission)).forward()