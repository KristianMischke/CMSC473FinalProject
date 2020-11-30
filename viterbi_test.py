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

DOC_OBSERVED_SPACE = ["special", "normal", "cold", "dizzy"]
DOC_STATE_SPACE = ["START", "Healthy", "Fever"]
DOC_INIT_PROBS = {"Healthy": 0.6, "Fever": 0.4}
DOC_TRANSITIONS = [
    [0., 0.6, 0.4], # this row is START -> Healthy, Fever (same as DOC_INIT_PROBS)
    [0., 0.7, 0.3],
    [0., 0.4, 0.6]
]
DOC_EMISSIONS = [
    [0., 0., 0., 0.], # 0th item is special
    [0., 0.5, 0.4, 0.1],
    [0., 0.1, 0.3, 0.6]
]

if __name__ == "__main__":
    observed_sequences, observed_translations, observed_token_lookup = tokenizer.convert_token_sequences_to_ids(OBSERVED, "BOS", "EOS")
    hidden_sequences, hidden_translations, hidden_token_lookup = tokenizer.convert_token_sequences_to_ids(HIDDEN, "BOS", "EOS")

    # Test the "doctor" example from the wiki
    len_states = len(DOC_STATE_SPACE)
    len_obs = len(DOC_OBSERVED_SPACE)
    doc_model = HMM(len_states, len_obs, 0, 0)
    for i in range(len_states):
        for j in range(len_states):
            doc_model.transitions[i][j] = DOC_TRANSITIONS[i][j]
        for j in range(len_obs):
            doc_model.emissions[i][j] = DOC_EMISSIONS[i][j]
    doc_model.viterbi([0,1,2,3])

    print("\n\n\n")

    # Test our examples
    hmm_model = HMM(len(hidden_token_lookup), len(observed_token_lookup), hidden_translations["BOS"], hidden_translations["EOS"])
    hmm_model.load_sequences(hidden_sequences, observed_sequences)
    for i in range(len(observed_sequences)):
        hmm_model.viterbi(observed_sequences[i])
        print(hidden_sequences[i])
        input() # Pause to observe the viterbi results with our hidden_sequences

    print("\n\n\n")
    print("PRLG TEST")
    # Test our examples
    prlg_model = PRLG(len(hidden_token_lookup), len(observed_token_lookup), hidden_translations["BOS"], hidden_translations["EOS"])
    prlg_model.load_sequences(hidden_sequences, observed_sequences)
    for i in range(len(observed_sequences)):
        prlg_model.viterbi(observed_sequences[i])
        print(hidden_sequences[i])
        input() # Pause to observe the viterbi results with our hidden_sequences

    # test novel sentences on both models
    novel_sentence = "the mouse ate the speedy cat .".split()
    novel_sentence_ids = tokenizer.convert_token_sequence_to_ids(novel_sentence, observed_translations, "BOS", "EOS")
    hmm_model.viterbi(novel_sentence_ids)
    prlg_model.viterbi(novel_sentence_ids)