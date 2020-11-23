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

        # matrix -> num_state+2 x num_obs
        a = zeros((self.num_obs+2, self.num_state))
        # Initializes a[*][START] to self.start_value
        a[0][0] = self.start_value

        for i in range(1, self.num_obs+1):
            for k in range(self.num_state):
                obs_probability = self.O[k, self.obs_seq[i]]
                for old in range(self.num_state):
                    move_probability = self.S[old, k]
                    a[i, k] += a[i-1, old] * move_probability * obs_probability

        'Utilizes the power of numpy -> same result'
        'Σ 𝛼(𝑖-1,𝑠′) ∗ 𝑝(𝑠|𝑠′) ∗ 𝑝(obs[𝑖]|𝑠′)'
        # for i in range(1, self.num_obs+1):
        #     for k in range(self.num_state):
        #         obs_probability = self.O[k, self.obs_seq[i]]
        #         a[i,k] += np.dot(a[i-1, :], (self.S[:, k])) * obs_probability

        return a


    # Backward algorithm
    def backward(self):
        print('Computing Backward Algorithm...')

        # matrix -> num_state+2 x num_obs
        b = zeros((self.num_obs+2, self.num_state))
        b[self.num_obs+1][-1] = self.start_value

        for i in range(self.num_obs, -1, -1):
            for next in range(self.num_state):
                obs_probability = self.O[next, self.obs_seq[i+1]]
                for k in range(self.num_state):
                    move_probability = self.S[k, next]
                    b[i, k] += b[i+1, next] * obs_probability * move_probability

        'Utilizes the power of numpy -> Working on this...'
        'Σ 𝛽(𝑖+1,𝑠′) ∗ 𝑝(𝑠′|𝑠) ∗ 𝑝(obs[𝑖+1]|𝑠′)'
        # for i in range(self.num_obs, -1, -1):
        #     for k in range(self.num_state):
        #         obs_probability = self.O[:, self.obs_seq[i + 1]]
        #         b[i, k] = np.sum(b[i + 1, :] * self.S[k, :] * obs_probability)

        return b

    # computes expectation maximization
    def expectation_maximization(self):
        a = self.forward()
        b = self.backward()
        c = zeros((self.num_obs, self.num_state))
        l = a[-1][-1]

        for i in reversed(range(self.num_obs - 1)):
            for next in range(self.num_state):
                c[next, self.obs_seq[i+1]] += a[i+1][next] * b[i+1][next]/l
                for k in range(self.num_state):
                    u = 'obs_prob(obsi+1 | next)' * self.S[k, next]
                    c[k, next] += a[i, k] * u * b[i+1][next]/l


if __name__ == "__main__":
    pass
    # transition = [[.7,.2,.1], [.15,.8,.05], [.6,.35,.05]]
    # emission = [[.7,.2,.05,.05], [.2,.6,.1,.1]]
    # obs_sequence = [0,1]
    #
    # BaumWeltch(array(transition), array(emission), obs_seq=obs_sequence).forward()
    # BaumWeltch(np.ndarray(transition), np.ndarray(emission)).forward()