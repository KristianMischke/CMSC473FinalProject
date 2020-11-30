from hmm import HMM, PRLG
from cascade_parser import CascadeParse
import tokenizer
import numpy as np

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

observed_sequences, observed_translations, observed_token_lookup = tokenizer.convert_token_sequences_to_ids(observed, "BOS", "EOS")
hidden_sequences, hidden_translations, hidden_token_lookup = tokenizer.convert_token_sequences_to_ids(hidden, "BOS", "EOS")

hmm_model = HMM(len(hidden_token_lookup), len(observed_token_lookup), hidden_translations["BOS"], hidden_translations["EOS"])
prlg_model = PRLG(len(hidden_token_lookup), len(observed_token_lookup), hidden_translations["BOS"], hidden_translations["EOS"])

hmm_model.load_sequences(hidden_sequences, observed_sequences)
prlg_model.load_sequences(hidden_sequences, observed_sequences)

hmm_model.test_validity()
prlg_model.test_validity()

# for h_i in range(len(hidden_token_lookup)):
#    for t_i in range(len(observed_token_lookup)):
#       print(f"P_emission_ind({observed_token_lookup[t_i]}|{hidden_token_lookup[h_i]})", prlg_model.p_emission_independent(t_i, h_i))
#
# print()
# print()
#
# for h_i in range(len(hidden_token_lookup)):
#    for h_i2 in range(len(hidden_token_lookup)):
#       print(f"P_transition({hidden_token_lookup[h_i]} -> {hidden_token_lookup[h_i2]})", prlg_model.p_transition(h_i2, h_i))
#
# print()
# print()
#
# for h_i in range(len(hidden_token_lookup)):
#    for h_i2 in range(len(hidden_token_lookup)):
#       for t_i in range(len(observed_token_lookup)):
#          print(f"P_joint({hidden_token_lookup[h_i]},{hidden_token_lookup[h_i2]} -> {observed_token_lookup[t_i]})", prlg_model.p_joint(h_i, t_i, h_i2))
#
# print()
# print()

for i in range(len(observed_sequences)):
   print(f"hmm.P({observed[i]}) = {hmm_model.p(observed_sequences[i])}")
   print(f"prlg.P({observed[i]}) = {prlg_model.p(observed_sequences[i])}")
   print(f"baum-welch.forward({observed[i]}) =\n {hmm_model.forward(observed_sequences[i])}")
   print(f"baum-welch.backward({observed[i]}) =\n {hmm_model.backward(observed_sequences[i])}")
   print()

novel_sentence = "the cat .".split()
novel_sentence_ids = tokenizer.convert_token_sequence_to_ids(novel_sentence, observed_translations, "BOS", "EOS")
print(f"hmm.P({novel_sentence}) = {hmm_model.p(novel_sentence_ids)}")
print(f"prlg.P({novel_sentence}) = {prlg_model.p(novel_sentence_ids)}")

novel_sentence = "the mouse .".split()
novel_sentence_ids = tokenizer.convert_token_sequence_to_ids(novel_sentence, observed_translations, "BOS", "EOS")
print(f"hmm.P({novel_sentence}) = {hmm_model.p(novel_sentence_ids)}")
print(f"prlg.P({novel_sentence}) = {prlg_model.p(novel_sentence_ids)}")

novel_sentence = "the mouse ate the speedy cat .".split()
novel_sentence_ids = tokenizer.convert_token_sequence_to_ids(novel_sentence, observed_translations, "BOS", "EOS")
print(f"hmm.P({novel_sentence}) = {hmm_model.p(novel_sentence_ids)}")
print(f"prlg.P({novel_sentence}) = {prlg_model.p(novel_sentence_ids)}")

cascade_parser = CascadeParse(hmm_model, tokenizer.get_frequencies_from_sequences(observed_sequences), hidden_translations["B"], hidden_translations["I"])
print("Cascaded Parse Parenthesization:")
for token in cascade_parser.parse(novel_sentence_ids):
   print(" ", end="")
   token.print(True)
input()

print("\n\n\n")

print(f"baum-welch.forward({novel_sentence}) =\n {hmm_model.forward(novel_sentence_ids)}")
print(f"baum-welch.backward({novel_sentence}) =\n {hmm_model.backward(novel_sentence_ids)}")
c_obs, c_trans = hmm_model.expectation_maximization(novel_sentence_ids)
print(f"baum-welch.ME({novel_sentence}) =")
print(c_obs)
print(c_trans)



print("----")
print('Exepectation Matrix test')
emission = [[.0, .0, .0, .0], [.0, .7, .2, .1], [.0, .1, .2, .7], [1.0, .0, .0, .0]]
transition = [[.0, .5, .5, .0], [.0, .8, .1, .1], [.0, .1, .8, .1], [.0, .0, .0, .0]]
obs_sequence = [0, 2, 3, 3, 2, 0]

test_hmm = HMM(4, 4, 0, 3)
test_hmm.emissions = np.array(emission)
test_hmm.transitions = np.array(transition)

a = test_hmm.forward(obs_sequence)
res = a[len(obs_sequence)-1][3]
print(a)
print(res)

print()
b = test_hmm.backward(obs_sequence)
print(b)

c_obs, c_trans = test_hmm.expectation_maximization(obs_sequence)
print(c_obs)
print(c_trans)