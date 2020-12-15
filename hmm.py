from baum_welch import BaumWeltch
import numpy as np
import random

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
            for j in range(len(hidden_seq) - 1):
                state = hidden_seq[j]
                next_state = hidden_seq[j + 1]
                observed = observed_seq[j]
                next_observed = observed_seq[j + 1]

                # update counts
                hidden_counts[state] += 1
                transition_counts[state][next_state] += 1
                emission_joint_counts[state][next_state][next_observed] += 1
                emission_counts[state][observed] += 1
            hidden_counts[hidden_seq[-1]] += 1
            emission_counts[hidden_seq[-1]][observed_seq[-1]] += 1

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
                        prob = emission_joint_counts[from_state][to_state][observed] / \
                               transition_counts[from_state][to_state]
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
            if abs(total - 1) > 0.00001 and from_state is not self.end_state:
                valid = False
                print(f"[HMM] Validation failed: transitions in {from_state} sum to {total} which is !~ 1")

        # verify any given node has a sum = 1 of it's emissions
        for from_state in range(self.num_hidden):
            total = 0
            for to_state in range(self.num_observed):
                total += self.emissions[from_state][to_state]
            if abs(total - 1) > 0.00001 and from_state is not self.end_state:
                valid = False
                print(f"[HMM] Validation failed: emissions in {from_state} sum to {total} which is !~ 1")

        if valid:
            print("[HMM] Validation Successful!")

    # forward probability
    def p(self, observed_sequence):
        t = len(observed_sequence)
        return self.forward(observed_sequence)[t - 1, self.end_state]

    # Baum-Welch Forward Algorithm
    # assumes obs_seq has already been tokenized (and thus contains the start and end tokens)
    def forward(self, obs_seq, log_space=False):
        t = len(obs_seq)

        # matrix -> num_state x num_obs
        # Matrix Initialization

        if log_space:
            a = np.full((t, self.num_hidden), -np.inf)
            a[0][0] = 0
        else:
            a = np.zeros((t, self.num_hidden))
            a[0][0] = 1.0

        for i in range(1, t):
            for k in range(self.num_hidden):
                for old in range(self.num_hidden):
                    if log_space:
                        a[i, k] = np.logaddexp2(a[i, k], a[i - 1, old] + self.p_joint(old, obs_seq[i], k, log_space=True))
                    else:
                        a[i, k] += a[i - 1, old] * self.p_joint(old, obs_seq[i], k)
        return a

    # Baum-Welch Backward Algorithm
    # assumes obs_seq has already been tokenized (and thus contains the start and end tokens)
    def backward(self, obs_seq, log_space=False):
        t = len(obs_seq)

        # matrix -> num_state x num_obs
        if log_space:
            b = np.full((t, self.num_hidden), -np.inf)
            b[-1][self.end_state] = 0
        else:
            b = np.zeros((t, self.num_hidden))
            b[-1][self.end_state] = 1

        for i in range(t - 2, -1, -1):
            for next in range(self.num_hidden):
                for k in range(self.num_hidden):
                    if log_space:
                        b[i, k] = np.logaddexp2(b[i, k], b[i + 1, next] + self.p_joint(k, obs_seq[i+1], next, log_space=True))
                    else:
                        b[i, k] += b[i + 1, next] * self.p_joint(k, obs_seq[i + 1], next)
        return b

    # Baum-Welch Expectation Maximization Algorithm
    def expectation_maximization(self, obs_seq, log_space=False):
        t = len(obs_seq)
        a = self.forward(obs_seq, log_space)
        b = self.backward(obs_seq, log_space)
        if log_space:
            c_obs = np.full((self.num_hidden, self.num_observed), -np.inf)
            c_trans = np.full((self.num_hidden, self.num_hidden), -np.inf)
            c_prlg = np.full((self.num_hidden, self.num_hidden, self.num_observed), -np.inf)

            c_obs[0, obs_seq[-1]] = 0
            c_trans[0, 0] = 0
            c_prlg[0, 0, obs_seq[-1]] = 0
        else:
            c_obs = np.zeros((self.num_hidden, self.num_observed))
            c_trans = np.zeros((self.num_hidden, self.num_hidden))
            c_prlg = np.zeros((self.num_hidden, self.num_hidden, self.num_observed))
        l = a[-1][self.end_state]

        for i in range(t - 2, -1, -1):
            for next in range(self.num_hidden):
                if log_space:
                    c_obs[next, obs_seq[i + 1]] = np.logaddexp2(c_obs[next, obs_seq[i + 1]], a[i + 1][next] + b[i + 1][next] - l)
                else:
                    c_obs[next, obs_seq[i + 1]] += a[i + 1][next] * b[i + 1][next] / l
                for k in range(self.num_hidden):
                    u = self.p_joint(k, obs_seq[i + 1], next, log_space)
                    if log_space:
                        c_trans[k, next] = np.logaddexp2(c_trans[k, next], a[i, k] + u + b[i + 1][next] - l)
                        c_prlg[k, next, obs_seq[i + 1]] = np.logaddexp2(c_prlg[k, next, obs_seq[i + 1]], a[i, k] + u + b[i + 1][next] - l)
                    else:
                        c_trans[k, next] += a[i, k] * u * b[i + 1][next] / l
                        c_prlg[k, next, obs_seq[i + 1]] += a[i, k] * u * b[i + 1][next] / l

        return c_obs, c_trans, c_prlg

    # Computes the perplexity of a sentence
    def compute_sent_perplexity(self, obs_seq):
        N = len(obs_seq)
        log_marginal_likelihood = self.forward(obs_seq, log_space=True)[-1, self.end_state]
        return np.exp2((-1 / N) * log_marginal_likelihood)

    # Computes the perplexity of a corpus
    def compute_corpus_perplexity(self, sentences):
        N = 0
        joint_log_prob = 0
        for sent in sentences:
            log_marginal_likelihood = self.forward(sent, log_space=True)[-1, self.end_state]
            joint_log_prob += log_marginal_likelihood
            N += len(sent)
        return np.exp2((-1 / N) * joint_log_prob)

    def em_update(self, obs_seq_batch):
        c_obs = np.full((self.num_hidden, self.num_observed), -np.inf)
        c_trans = np.full((self.num_hidden, self.num_hidden), -np.inf)
        c_prlg = np.full((self.num_hidden, self.num_hidden, self.num_observed), -np.inf)

        for obs_seq in obs_seq_batch:
            c_obs_temp, c_trans_temp, c_prlg_temp = self.expectation_maximization(obs_seq, log_space=True)
            c_obs = np.logaddexp2(c_obs, c_obs_temp)
            c_trans = np.logaddexp2(c_trans, c_trans_temp)
            c_prlg = np.logaddexp2(c_prlg, c_prlg_temp)

        # update emission and transition probabilities based on EM results
        for i in range(self.num_hidden):
            self.emissions[i, :] += np.exp2(c_obs[i, :])
            total = sum(self.emissions[i, :])
            if total != 0:
                self.emissions[i, :] /= total

        for i in range(self.num_hidden):
            self.transitions[i, :] += np.exp2(c_trans[i, :])
            total = sum(self.transitions[i, :])
            if total != 0:
                self.transitions[i, :] /= total

        for i in range(self.num_hidden):
            for j in range(self.num_hidden):
                self.emissions_joint[i, j, :] += np.exp2(c_prlg[i, j, :])
                total = sum(self.emissions_joint[i, j, :])
                if total != 0:
                    self.emissions_joint[i, j, :] /= total

    # P(y_n -> x_n y_n+1) Probability that the current state y_n emits observed state x_n
    # AND produced the next hidden state y_n+1
    def p_joint(self, from_hidden, observed, next_hidden, log_space=False):
        obs_probability = self.p_emission_independent(observed, next_hidden)
        move_probability = self.p_transition(next_hidden, from_hidden)
        if log_space:
            obs_probability = -np.inf if obs_probability == 0 else np.log2(obs_probability)
            move_probability = -np.inf if move_probability == 0 else np.log2(move_probability)
            return obs_probability + move_probability
        return obs_probability * move_probability

    # P(x_n|y_n) Probability that the hidden state emits the observed state
    def p_emission_independent(self, observed, hidden):
        return self.emissions[hidden][observed]

    # P(y_n+1|y_n) Probability that the hidden state transitions to the next_hidden state
    def p_transition(self, next_hidden, hidden):
        return self.transitions[hidden][next_hidden]

    # https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
    # Test in viterbi_test.py:
    #   If I pass in each sequence from our observed_sequences in temp, I should be getting the hidden_sequence
    #   as the most probable sequence of hidden states.
    def viterbi(self, observed_sequence, debug=False):
        # The spaces are just indices because we converted each unique token to its own unique index using our
        # tokenizer. 

        obs_space = [x for x in range(self.num_observed)]
        state_space = [x for x in range(self.num_hidden)]

        # v_table contains every state at every observation, and the probability is calculated
        #   at each state given the path prior (from obs=o to obs=i-1 where i is the current observation) and
        #   the probability of this state following its previous.
        # v_table sized: T x K:
        #   T is the total number of observations in the given sequence
        #   K is the total number of states
        # v_table[i][j]
        #   i: index of observation
        #   j: index of state in state space
        v_table = [{}]
        
        # Initialize every state at the first observation in the sequence
        #   "prob": The probability of a state occurring at first step 
        #           and the probability of that observation occurring given that state
        #   "prev": None, there is no previous state at first observation
        for state in state_space:
            v_table[0][state] = {
                "prob": 1 if state is self.start_state else 0,
                "prev": None
            }

        # In the wiki, t was used instead of "obs" here to indicate "t"ime of occurrence
        # Instead, refer to "obs" for observations in the order that they occur in the sequence
        for obs in range(1, len(observed_sequence)):
            v_table.append({})
            for state in state_space:
                # Begin finding the maximum probability path of a state given all possible previous states

                # Initialize "max" probability by beginning at the probability of the 
                #   previous path and first state.
                # Calculate the joint probability
                max_prev_state = state_space[0]
                max_path_prob = v_table[obs - 1][max_prev_state]["prob"] * self.p_joint(max_prev_state, observed_sequence[obs], state)

                # Start from second state since we initialized max from first
                for prev_state in state_space[1:]:
                    # Calculate the probability of the current state given the previous path and its last state
                    path_prob = v_table[obs - 1][prev_state]["prob"] * self.p_joint(prev_state, observed_sequence[obs], state)
                    if path_prob > max_path_prob:
                        max_path_prob = path_prob
                        max_prev_state = prev_state

                # Finally, store our max probability at this observation
                v_table[obs][state] = {
                    "prob": max_path_prob,
                    "prev": max_prev_state
                }

        if debug:
            for line in self.dptable(v_table):
                print(line)
        
        best_path = []
        max_prob = 0.0
        prev = None
        best_state = None
        
        # Grab the best (max probability) path by finding it in the last index of v_table
        # This is the probability that this result is reached by the end of the Viterbi Algorithm
        for state, data in v_table[-1].items():
            if debug:
                print("state: {}".format(state))
                print("data: {}".format(data))
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_state = state

        if best_state is not None:
            # Back-track from our best path starting at the end
            best_path.append(best_state)
            prev = best_state

            # Start from second-to-last index since we found the end of our best path (len() - 1 is last index)
            # End at -1 (exclusive)
            # Re-build our path from the left (inserting at 0)
            for t in range(len(v_table) - 2, -1, -1):
                best_path.insert(0, v_table[t + 1][prev]["prev"])
                prev = v_table[t + 1][prev]["prev"]

            # TODO: move print to calling function??
            if debug:
                print('The steps of states are ' + ' '.join([str(x) for x in best_path]) + ' with highest probability of {}'.format(str(max_prob)))
            return best_path
        return None

    # Taken from the Python example in the wiki, adjusted for better output spacing
    def dptable(self, v_table):
        # Print a table of steps from dictionary
        yield "".ljust(4) + " ".join((str(i).ljust(7)) for i in range(len(v_table)))
        for state in v_table[0]:
            yield ("{}: ".format(state)).ljust(4) + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in v_table)


class PRLG(HMM):
    """
    Child class that redefines the probability to be an exact representation without the independence assumption
    referred to as PRLG (Probabilistic Right Linear Grammar) in the paper
    """

    # P(y_n -> x_n y_n+1) Probability that the current state y_n emits observed state x_n
    # AND produced the next hidden state y_n+1
    def p_joint(self, hidden, observed, next_hidden, log_space=False):
        obs_probability = self.p_emission(observed, next_hidden, hidden)
        move_probability = self.p_transition(next_hidden, hidden)
        if log_space:
            obs_probability = -np.inf if obs_probability == 0 else np.log2(obs_probability)
            move_probability = -np.inf if move_probability == 0 else np.log2(move_probability)
            return obs_probability + move_probability
        return obs_probability * move_probability

    # P(x_n+1|y_n,y_n+1) Probability of an observed state given that the hidden state transitions to the next_hidden state
    def p_emission(self, observed, next_hidden, hidden):
        return self.emissions_joint[hidden][next_hidden][observed]
