# Hidden Markov Model base class
#
# member variables:
# -hidden            The hidden states labelled y in the paper
# -observed          The observed states labelled x in the paper
# -transitions       Probabilities for transitioning between hidden states y_i -> y_i+1
# -emissions         Probabilities for producing observed states from hidden states y_i -> x_i
# -emissions_joint   Probabilities for joint exact value of joint probability used in PRLGs y_i,y_i+1 -> x_i
# -start_state       Starting state used in the model
# -end_state         Ending state used in the model
#
# NOTE: for our purposes 'observed' states are word tokens
class HMM():

    # initializes the states and tokens
    def __init__(self, hidden, observed, start_state, end_state):
        self.hidden = set(hidden)
        self.observed = set(observed)
        self.transitions = {}
        self.emissions = {}
        self.emissions_joint = {}
        self.start_state = start_state
        self.end_state = end_state

    # assign probability to the transition y_n -> y_n+1
    def set_transition(self, from_state, to_state, prob):
        if from_state not in self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][to_state] = prob

    # assign probability to the emission y_n -> x_n
    def set_emission(self, from_state, emission_token, prob):
        if from_state not in self.emissions:
            self.emissions[from_state] = {}
        self.emissions[from_state][emission_token] = prob

    # assign probability to the joint emission y_n,y_n+1 -> x_n
    def set_emission_joint(self, from_state, next_state, emission_token, prob):
        if from_state not in self.emissions_joint:
            self.emissions_joint[from_state] = {}
        if next_state not in self.emissions_joint[from_state]:
            self.emissions_joint[from_state][next_state] = {}
        self.emissions_joint[from_state][next_state][emission_token] = prob

    # helper function for loading emissions & transitions from existing data
    def load_sequences(self, hidden_sequences, observed_sequences):
        if len(hidden_sequences) != len(observed_sequences):
            raise Exception(
                f"[HMM] load_sequences: sequence lengths must match ({len(hidden_sequences)} != {len(observed_sequences)})")

        hidden_counts = {}
        transition_counts = {}
        emission_counts = {}
        emission_joint_counts = {}

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

                # ensure keys in dictionaries
                if state not in hidden_counts:
                    hidden_counts[state] = 0
                if state not in emission_counts:
                    emission_counts[state] = {}
                if observed not in emission_counts[state]:
                    emission_counts[state][observed] = 0
                if state not in transition_counts:
                    transition_counts[state] = {}
                if next_state not in transition_counts[state]:
                    transition_counts[state][next_state] = 0

                if state not in emission_joint_counts:
                    emission_joint_counts[state] = {}
                if next_state not in emission_joint_counts[state]:
                    emission_joint_counts[state][next_state] = {}
                if observed not in emission_joint_counts[state][next_state]:
                    emission_joint_counts[state][next_state][observed] = 0

                # update counts
                hidden_counts[state] += 1
                transition_counts[state][next_state] += 1
                emission_joint_counts[state][next_state][observed] += 1
                emission_counts[state][observed] += 1

        # update transition probabilities
        for from_state in transition_counts:
            for to_state in transition_counts[from_state]:
                self.set_transition(from_state, to_state,
                                    transition_counts[from_state][to_state] / hidden_counts[from_state])

        # update emission probabilities
        for hidden_state in emission_counts:
            for observed in emission_counts[hidden_state]:
                self.set_emission(hidden_state, observed,
                                  emission_counts[hidden_state][observed] / hidden_counts[hidden_state])

        # update joint emission probabilities
        for from_state in emission_joint_counts:
            for to_state in emission_joint_counts[from_state]:
                for observed in emission_joint_counts[from_state][to_state]:
                    self.set_emission_joint(from_state, to_state, observed,
                                            emission_joint_counts[from_state][to_state][observed] /
                                            transition_counts[from_state][to_state])

    # runs tests to ensure all the probability distributions are proper
    # TODO: verify states in the transition & emission dicts are in the hidden and observed sets
    def test_validity(self):
        valid = True

        # verify any given node has a sum = 1 of it's transitions
        for from_state in self.transitions:
            sum = 0
            for to_state in self.transitions[from_state]:
                sum += self.transitions[from_state][to_state]
            if abs(sum - 1) > 0.00001:
                valid = False
                print(f"[HMM] Validation failed: transitions in {from_state} sum to {sum} which is !~ 1")

        # verify any given node has a sum = 1 of it's emissions
        for from_state in self.emissions:
            sum = 0
            for to_state in self.emissions[from_state]:
                sum += self.emissions[from_state][to_state]
            if abs(sum - 1) > 0.00001:
                valid = False
                print(f"[HMM] Validation failed: emissions in {from_state} sum to {sum} which is !~ 1")

        if valid:
            print("[HMM] Validation Successful!")

    # HMM forward algorithm
    def P(self, observed_sequence):
        N = len(observed_sequence)
        alpha = []
        for i in range(N + 2):
            alpha.append({})
            for state in self.hidden:
                alpha[i][state] = 0

        alpha[0][self.start_state] = 1

        for i in range(1, N + 1):
            for state in self.hidden:
                for old in self.hidden:
                    alpha[i][state] += alpha[i - 1][old] * self.P_joint(old, observed_sequence[i - 1], state)

        return alpha[N][self.end_state]

    # P(y_n -> x_n y_n+1) Probability that the current state y_n emmits observed state x_n
    # AND produced the next hidden state y_n+1
    def P_joint(self, from_hidden, observed, next_hidden):
        return self.P_emission_independent(observed, from_hidden) * self.P_transition(next_hidden, from_hidden)

    # P(x_n|y_n) Probability that the hidden state emits the observed state
    def P_emission_independent(self, observed, hidden):
        if hidden not in self.emissions:
            raise Exception(f"[HMM] P_emission_independent: hidden state '{hidden}' not found!")
        if observed not in self.emissions[hidden]:
            return 0
        return self.emissions[hidden][observed]

    # P(y_n+1|y_n) Probability that the hidden state transitions to the next_hidden state
    def P_transition(self, next_hidden, hidden):
        if hidden not in self.transitions:
            raise Exception(f"[HMM] P_transition: hidden state '{hidden}' not found!")
        if next_hidden not in self.transitions[hidden]:
            return 0
        return self.transitions[hidden][next_hidden]

    # TODO: @Min find the most likely sequence of hidden states given an observed sequence
    def viterbi(self, observed_sequence):
        pass


# child class that redefines the probability to be an exact representation without the independence assumption
# referred to as PRLG (Probabilistic Right Linear Grammar) in the paper
class PRLG(HMM):
    # P(y_n -> x_n y_n+1) Probability that the current state y_n emmits observed state x_n
    # AND produced the next hidden state y_n+1
    def P_joint(self, hidden, observed, next_hidden):
        return self.P_emission(observed, next_hidden, hidden) * self.P_transition(next_hidden, hidden)

    # P(x_n|y_n,y_n+1) Probability of an observed state given that the hidden state transitions to the next_hidden state
    def P_emission(self, observed, next_hidden, hidden):
        if hidden not in self.emissions_joint:
            raise Exception(f"[HMM] P_transition: hidden state '{hidden}' not found!")
        if next_hidden not in self.emissions_joint[hidden]:
            return 0
        if observed not in self.emissions_joint[hidden][next_hidden]:
            return 0
        return self.emissions_joint[hidden][next_hidden][observed]
