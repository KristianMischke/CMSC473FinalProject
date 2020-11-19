import numpy as np

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

    def __init__(self, S, O, pi):
        self.pi = pi
        self.S = S
        self.O = O
        self.num_obs = 0
        self.num_state = 0

        print('here')

    def forward(self):
        print('Forward Alg')

    def backward(self):
        print('Backward Alg')


BaumWeltch()