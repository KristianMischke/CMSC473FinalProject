

# convert the tokens in a sequences to indices based on given translation table
def convert_token_sequence_to_ids(sequence, translation_table, bos=None, eos=None, oov=None):
    new_sequence = []
    if bos is not None:
        new_sequence.append(translation_table[bos])
    for token in sequence:
        if token in translation_table:
            new_sequence.append(translation_table[token])
        elif oov is not None:
            new_sequence.append(translation_table[oov])
    if eos is not None:
        new_sequence.append(translation_table[eos])
    return new_sequence


# convert the tokens in an array of sequences to indices
# returns tuple with translated sequences, translation table (token -> id), and lookup table (id -> token)
def convert_token_sequences_to_ids(sequences, bos=None, eos=None, oov_thresh=0, oov=None):
    translation_table = {}
    token_lookup = []
    out_sequences = []
    if bos is not None:
        translation_table[bos] = len(translation_table)
        token_lookup.append(bos)
    if eos is not None and eos != bos:
        translation_table[eos] = len(translation_table)
        token_lookup.append(eos)
    if oov is not None:
        translation_table[oov] = len(translation_table)
        token_lookup.append(oov)

    # first get counts to be used in OOV calculation
    token_counts = {}
    for sequence in sequences:
        for token in sequence:
            if token in token_counts:
                token_counts[token] += 1
            else:
                token_counts[token] = 1

    # convert the sequence to ids, generating new ids when new tokens are encountered
    for sequence in sequences:
        new_sequence = []
        if bos is not None:
            new_sequence.append(translation_table[bos])
        for token in sequence:
            if token_counts[token] <= oov_thresh:
                new_sequence.append(translation_table[oov])
            else:
                if token not in translation_table:
                    translation_table[token] = len(translation_table)
                    token_lookup.append(token)
                new_sequence.append(translation_table[token])
        if eos is not None:
            new_sequence.append(translation_table[eos])
        out_sequences.append(new_sequence)
    return out_sequences, translation_table, token_lookup


def convert_id_sequence_to_tokens(sequence, token_lookup):
    new_sequence = []
    for index in sequence:
        new_sequence.append(token_lookup[index])
    return new_sequence


def get_frequencies_from_sequences(sequences):
    frequency_dict = {}

    for sequence in sequences:
        for token in sequence:
            if token not in frequency_dict:
                frequency_dict[token] = 1
            else:
                frequency_dict[token] += 1

    return frequency_dict
