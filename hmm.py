
# Hidden Markov Model base class
#
# member variables:
# -hidden            The hidden states labelled y in the paper
# -observed          The observed states labelled x in the paper
# -transitions       Probabilities for transitioning between hidden states y_i -> y_i+1
# -emissions         Probabilities for producing observed states from hidden states y_i -> x_i
#
# NOTE: for our purposes 'observed' states are word tokens
class HMM():

   # initializes the states and tokens
   def __init__(self, hidden, observed, start_state):
      self.hidden = set(hidden)
      self.observed = set(observed)
      self.start_state = start_state

   # assign probability to the transition y_n -> y_n+1
   def set_transition(self, from_state, to_state, prob):
      self.transitions[from_state][to_state] = prob

   # assign probability to the emission y_n -> x_n
   def set_emission(self, from_state, emission_token, prob):
      self.emissions[from_state][emission_token] = prob

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
   
   # P(y_n -> x_n y_n+1) Probability that the current state y_n emmits observed state x_n
   # AND produced the next hidden state y_n+1
   def P(self, from_hidden, observed, next_hidden):
      return self.P_emission_independent(observed, from_hidden)*self.P_transition(next_hidden, from_hidden)

   # TODO: implement probabilities

   # P(x_n|y_n) Probability that the hidden state emits the observed state
   def P_emission_independent(self, observed, hidden):
      pass

   # P(y_n+1|y_n) Probability that the hidden state transitions to the next_hidden state
   def P_transition(self, next_hidden, hidden):
      pass

   # P(x_n|y_n,y_n+1) Probability of an observed state given that the hidden state transitions to the next_hidden state
   def P_emission(self, observed, next_hidden, hidden):
      pass


# child class that redefines the probability to be an exact representation without the independence assumption
# refered to as PRLG (Probabalistic Right Linear Grammar) in the paper
class PRLG(HMM):
   def P(self, hidden, observed, next_hidden):
      return self.P_emission(observed, next_hidden, hidden)*self.P_transition(next_hidden, hidden)