from hmm import HMM, PRLG

observed = [
   "the fluffy cat chased the mouse .".split(),
   "the cat chased the mouse .".split(),
   "the cat ate the mouse .".split(),
   "the cat ate the speedy mouse .".split(),
   "the super fluffy cat ate the speedy mouse .".split()
]

hidden = [
   "B I I O B I STOP".split(),
   "B I O B I STOP".split(),
   "B I O B I STOP".split(),
   "B I O B I I STOP".split(),
   "B I I I O B I I STOP".split()
]

hidden_states = "B I O STOP".split()
word_types = "the fluffy cat chased mouse . ate super speedy".split()

hmm_model = HMM(hidden_states, word_types, "B", "STOP")
prlg_model = PRLG(hidden_states, word_types, "B", "STOP")

hmm_model.load_sequences(hidden, observed)
prlg_model.load_sequences(hidden, observed)

hmm_model.test_validity()
prlg_model.test_validity()

for state in hidden_states:
   for t in word_types:
      print(f"P_emission_ind({t}, {state})", prlg_model.P_emission_independent(t, state))

print()
print()

for state in hidden_states:
   for next_state in hidden_states:
      print(f"P_transition({state} -> {next_state})", prlg_model.P_transition(next_state, state))

print()
print()

for state in hidden_states:
   for next_state in hidden_states:
      for t in word_types:
         print(f"P_joint({state},{next_state} -> {t})", prlg_model.P_joint(state, t, next_state))

print()
print()

for i in range(len(observed)):
   print(f"hmm.P({observed[i]}) = {hmm_model.P(observed[i])}")
   print(f"prlg.P({observed[i]}) = {prlg_model.P(observed[i])}")
   print()

novel_sentence = "the cat .".split()
print(f"hmm.P({novel_sentence}) = {hmm_model.P(novel_sentence)}")
print(f"prlg.P({novel_sentence}) = {prlg_model.P(novel_sentence)}")

novel_sentence = "the mouse ate the speedy cat .".split()
print(f"hmm.P({novel_sentence}) = {hmm_model.P(novel_sentence)}")
print(f"prlg.P({novel_sentence}) = {prlg_model.P(novel_sentence)}")