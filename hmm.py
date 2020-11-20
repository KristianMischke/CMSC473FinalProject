import numpy as np


class HMM:
    """
    Hidden Markov Model base class

    member variables:
        -hidden            The hidden states labelled y in the paper
        -observed          The observed states labelled x in the paper
        -transitions       Probabilities for transitioning between hidden states y_i -> y_i+1
        -emissions         Probabilities for producing observed states from hidden states y_i -> x_i
        -emissions_joint   Probabilities for joint exact value of joint probability used in PRLGs y_i,y_i+1 -> x_i
        -start_state       Starting state used in the model
        -end_state         Ending state used in the model

    NOTE: for our purposes 'observed' states are word tokens
    """

    # initializes the states and tokens
    def __init__(self, num_hidden, num_observed, start_state, end_state):
        self.num_hidden = num_hidden
        self.num_observed = num_observed
        self.transitions = np.zeros((num_hidden, num_hidden))
        self.emissions = np.zeros((num_hidden, num_observed))
        self.emissions_joint = np.zeros((num_hidden, num_hidden, num_observed))
        self.start_state = start_state
        self.end_state = end_state

    # assign probability to the transition y_n -> y_n+1
    def set_transition(self, from_state, to_state, prob):
        self.transitions[from_state][to_state] = prob

    # assign probability to the emission y_n -> x_n
    def set_emission(self, from_state, emission_token, prob):
        self.emissions[from_state][emission_token] = prob

    # assign probability to the joint emission y_n,y_n+1 -> x_n
    def set_emission_joint(self, from_state, next_state, emission_token, prob):
        self.emissions_joint[from_state][next_state][emission_token] = prob

    # helper function for loading emissions & transitions from existing data
    def load_sequences(self, hidden_sequences, observed_sequences):
        if len(hidden_sequences) != len(observed_sequences):
            raise Exception(
                f"[HMM] load_sequences: sequence lengths must match ({len(hidden_sequences)} != {len(observed_sequences)})")

        hidden_counts = np.zeros(self.num_hidden, dtype=int)
        transition_counts = np.zeros((self.num_hidden, self.num_hidden), dtype=int)
        emission_counts = np.zeros((self.num_hidden, self.num_observed), dtype=int)
        emission_joint_counts = np.zeros((self.num_hidden, self.num_hidden, self.num_observed), dtype=int)

        # loop over all given sequences
        for i in range(len(hidden_sequences)):
            hidden_seq = hidden_sequences[i]
            observed_seq = observed_sequences[i]

            if len(hidden_seq) != len(observed_seq):
                raise Exception(
                    f"[HMM] load_sequences: sequence[{i}] lengths must match ({len(hidden_seq)} != {len(observed_seq)})")

            # loop over the tokens/states in the sequence
            for j in range(len(hidden_seq)):
                state = hidden_seq[j]
                next_state = self.end_state if j + 1 >= len(hidden_seq) else hidden_seq[j + 1]
                observed = observed_seq[j]

                # update counts
                hidden_counts[state] += 1
                transition_counts[state][next_state] += 1
                emission_joint_counts[state][next_state][observed] += 1
                emission_counts[state][observed] += 1

        # update transition probabilities
        for from_state in range(self.num_hidden):
            for to_state in range(self.num_hidden):
                if hidden_counts[from_state] == 0:
                    prob = 0
                else:
                    prob = transition_counts[from_state][to_state] / hidden_counts[from_state]
                self.set_transition(from_state, to_state, prob)

        # update emission probabilities
        for hidden_state in range(self.num_hidden):
            for observed in range(self.num_observed):
                if hidden_counts[hidden_state] == 0:
                    prob = 0
                else:
                    prob = emission_counts[hidden_state][observed] / hidden_counts[hidden_state]
                self.set_emission(hidden_state, observed, prob)

        # update joint emission probabilities
        for from_state in range(self.num_hidden):
            for to_state in range(self.num_hidden):
                for observed in range(self.num_observed):
                    if transition_counts[from_state][to_state] == 0:
                        prob = 0
                    else:
                        prob = emission_joint_counts[from_state][to_state][observed] / transition_counts[from_state][to_state]
                    self.set_emission_joint(from_state, to_state, observed, prob)

    # runs tests to ensure all the probability distributions are proper
    # TODO: verify states in the transition & emission dicts are in the hidden and observed sets
    def test_validity(self):
        valid = True

        # verify any given node has a sum = 1 of it's transitions
        for from_state in range(self.num_hidden):
            total = 0
            for to_state in range(self.num_hidden):
                total += self.transitions[from_state][to_state]
            if abs(total - 1) > 0.00001:
                valid = False
                print(f"[HMM] Validation failed: transitions in {from_state} sum to {total} which is !~ 1")

        # verify any given node has a sum = 1 of it's emissions
        for from_state in range(self.num_hidden):
            total = 0
            for to_state in range(self.num_observed):
                total += self.emissions[from_state][to_state]
            if abs(total - 1) > 0.00001:
                valid = False
                print(f"[HMM] Validation failed: emissions in {from_state} sum to {total} which is !~ 1")

        if valid:
            print("[HMM] Validation Successful!")

    # HMM forward algorithm
    def p(self, observed_sequence):
        n = len(observed_sequence)
        alpha = np.zeros((n+2, self.num_hidden))
        alpha[0][self.start_state] = 1

        for i in range(1, n + 1):
            for state in range(self.num_hidden):
                for old in range(self.num_hidden):
                    alpha[i][state] += alpha[i - 1][old] * self.p_joint(old, observed_sequence[i - 1], state)

        return alpha[n][self.end_state]

    # P(y_n -> x_n y_n+1) Probability that the current state y_n emits observed state x_n
    # AND produced the next hidden state y_n+1
    def p_joint(self, from_hidden, observed, next_hidden):
        return self.p_emission_independent(observed, from_hidden) * self.p_transition(next_hidden, from_hidden)

    # P(x_n|y_n) Probability that the hidden state emits the observed state
    def p_emission_independent(self, observed, hidden):
        return self.emissions[hidden][observed]

    # P(y_n+1|y_n) Probability that the hidden state transitions to the next_hidden state
    def p_transition(self, next_hidden, hidden):
        return self.transitions[hidden][next_hidden]

    # TODO: @Min find the most likely sequence of hidden states given an observed sequence
    def viterbi(self, observed_sequence):
        pass


class PRLG(HMM):
    """
    Child class that redefines the probability to be an exact representation without the independence assumption
    referred to as PRLG (Probabilistic Right Linear Grammar) in the paper
    """

    # P(y_n -> x_n y_n+1) Probability that the current state y_n emits observed state x_n
    # AND produced the next hidden state y_n+1
    def p_joint(self, hidden, observed, next_hidden):
        return self.p_emission(observed, next_hidden, hidden) * self.p_transition(next_hidden, hidden)

    # P(x_n|y_n,y_n+1) Probability of an observed state given that the hidden state transitions to the next_hidden state
    def p_emission(self, observed, next_hidden, hidden):
        return self.emissions_joint[hidden][next_hidden][observed]
