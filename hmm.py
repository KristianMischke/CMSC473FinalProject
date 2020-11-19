
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
      self.transitions[from_state][to_state] = prob

   # assign probability to the emission y_n -> x_n
   def set_emission(self, from_state, emission_token, prob):
      self.emissions[from_state][emission_token] = prob

   # assign probability to the joint emission y_n,y_n+1 -> x_n
   def set_emission_joint(self, from_state, next_state, emission_token, prob):
      self.emissions_joint[from_state][next_state][emission_token] = prob

   # runs tests to ensure all the probability distributions are proper
   # TODO: verify states in the transition & emission dicts are in the hidden and observed sets
   def test_validity(self):
      valid = True

      # verify any given node has a sum = 1 of it's transitions
      for from_state in self.transitions:
         sum = 0
         for to_state in from_state:
            sum += to_state
         if abs(sum - 1) > 0.00001:
            valid = False
            print(f"[HMM] Validation failed: transitions in {from_state} sum to {sum} which is !~ 1")

      # verify any given node has a sum = 1 of it's emissions
      for from_state in self.emissions:
         sum = 0
         for to_state in from_state:
            sum += to_state
         if abs(sum - 1) > 0.00001:
            valid = False
            print(f"[HMM] Validation failed: emissions in {from_state} sum to {sum} which is !~ 1")
      
      if valid:
         print("[HMM] Validation Successful!")
   
   # HMM forward algorithm
   def P(self, observed_sequence):
      N = len(observed_sequence)
      alpha = []*(N+2)

      for state in self.hidden:
         alpha[0][state] = 0
      alpha[0][self.start_state] = 1

      for i in range(1, N+1):
         for state in self.hidden:
            p_obs = self.P_emission_independent(observed_sequence[i], state)  # note: this caching is better than using the joint probability
                                                                              # in the 3rd nested loop, but it unfortunately unavoidable in PRLG
            for old in self.hidden:
               p_move = self.P_transition(state, old)
               alpha[i][state] += alpha[i-1][old] * p_obs * p_move

      return alpha[N+1][self.end_state]

   # P(y_n -> x_n y_n+1) Probability that the current state y_n emmits observed state x_n
   # AND produced the next hidden state y_n+1
   def P_joint(self, from_hidden, observed, next_hidden):
      return self.P_emission_independent(observed, from_hidden)*self.P_transition(next_hidden, from_hidden)

   # P(x_n|y_n) Probability that the hidden state emits the observed state
   def P_emission_independent(self, observed, hidden):
      return self.emissions[hidden][observed]

   # P(y_n+1|y_n) Probability that the hidden state transitions to the next_hidden state
   def P_transition(self, next_hidden, hidden):
      return self.transitions[hidden][next_hidden]

   # TODO: @Min find the most likely sequence of hidden states given an observed sequence
   def viterbi(self, observed_sequence):
      pass


# child class that redefines the probability to be an exact representation without the independence assumption
# refered to as PRLG (Probabalistic Right Linear Grammar) in the paper
class PRLG(HMM):

   # PRLG forward algorithm
   def P(self, observed_sequence):
      N = len(observed_sequence)
      alpha = []*(N+2)

      for state in self.hidden:
         alpha[0][state] = 0
      alpha[0][self.start_state] = 1

      for i in range(1, N+1):
         for state in self.hidden:
            for old in self.hidden:
               alpha[i][state] += alpha[i-1][old] * self.P_joint(old, observed_sequence[i], state) # PRLG uses exact definition of joint probability

      return alpha[N+1][self.end_state]

   # P(y_n -> x_n y_n+1) Probability that the current state y_n emmits observed state x_n
   # AND produced the next hidden state y_n+1
   def P_joint(self, hidden, observed, next_hidden):
      return self.P_emission(observed, next_hidden, hidden)*self.P_transition(next_hidden, hidden)

   # P(x_n|y_n,y_n+1) Probability of an observed state given that the hidden state transitions to the next_hidden state
   def P_emission(self, observed, next_hidden, hidden):
      return self.emissions_joint[hidden][next_hidden][observed]