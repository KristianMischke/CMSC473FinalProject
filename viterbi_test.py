from hmm import HMM, PRLG
import tokenizer

OBSERVED = [
   "the fluffy cat chased the mouse .".split(),
   "the cat chased the mouse .".split(),
   "the cat ate the mouse .".split(),
   "the cat ate the speedy mouse .".split(),
   "the super fluffy cat ate the speedy mouse .".split()
]
HIDDEN = [
   "B I I O B I STOP".split(),
   "B I O B I STOP".split(),
   "B I O B I STOP".split(),
   "B I O B I I STOP".split(),
   "B I I I O B I I STOP".split()
]

DOC_OBSERVED_SPACE = ("normal", "cold", "dizzy")
DOC_STATE_SPACE = ("Healthy", "Fever")
DOC_INIT_PROBS = {"Healthy": 0.6, "Fever": 0.4}
DOC_TRANSITIONS = [
    [0.7, 0.3],
    [0.4, 0.6]
]
DOC_EMISSIONS = [
    [0.5, 0.4, 0.1],
    [0.1, 0.3, 0.6]
]

if __name__ == "__main__":
    observed_sequences, observed_translations, observed_token_lookup = tokenizer.convert_token_sequences_to_ids(OBSERVED)
    hidden_sequences, hidden_translations, hidden_token_lookup = tokenizer.convert_token_sequences_to_ids(HIDDEN)

    # Test the "doctor" example from the wiki
    len_states = 2
    len_obs = 3
    doc_model = HMM(len_states, len_obs, None, None)
    for i in range(len_states):
        for j in range(len_states):
            doc_model.transitions[i][j] = DOC_TRANSITIONS[i][j]
        for j in range(len_obs):
            doc_model.emissions[i][j] = DOC_EMISSIONS[i][j]
    doc_model.viterbi([0,1,2], {0:0.6,1:0.4})

    print("\n\n\n")

    # Test our examples
    test_model = HMM(len(hidden_token_lookup), len(observed_token_lookup), hidden_translations["B"], hidden_translations["STOP"])
    test_model.load_sequences(hidden_sequences, observed_sequences)
    for i in range(len(observed_sequences)):
        test_model.viterbi(observed_sequences[i])
        print(hidden_sequences[i])
        input() # Pause to observe the viterbi results with our hidden_sequences
