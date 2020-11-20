
# convert the tokens in a sequences to indices based on given translation table
def convert_token_sequence_to_ids(sequence, translation_table, oov=None):
    new_sequence = []
    for token in sequence:
        if token in translation_table:
            new_sequence.append(translation_table[token])
        elif oov is not None:
            new_sequence.append(oov)
    return new_sequence

# convert the tokens in an array of sequences to indices
# returns tuple with translated sequences, translation table (token -> id), and lookup table (id -> token)
def convert_token_sequences_to_ids(sequences):
    translation_table = {}
    token_lookup = []
    out_sequences = []
    for sequence in sequences:
        new_sequence = []
        for token in sequence:
            if token not in translation_table:
                translation_table[token] = len(translation_table)
                token_lookup.append(token)
            new_sequence.append(translation_table[token])
        out_sequences.append(new_sequence)
    return out_sequences, translation_table, token_lookup

def convert_id_sequence_to_tokens(sequence, token_lookup):
    new_sequence = []
    for index in sequence:
        new_sequence.append(token_lookup[index])
    return new_sequence
