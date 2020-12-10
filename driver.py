from ast import literal_eval
from hmm import HMM, PRLG
from cascade_parser import CascadeParse
import tokenizer, preprocessor
import random
import numpy as np


def init_probabilities(model, hidden_translation, observed_translation, use_stop_state: bool):
    b_state = hidden_translation["B"]
    i_state = hidden_translation["I"]
    bos_state = hidden_translation["BOS"]
    eos_state = hidden_translation["EOS"]
    stop_state = hidden_translation["STOP"] if use_stop_state else None

    bos_token = observed_translation["BOS"]
    eos_token = observed_translation["EOS"]
    punct_strs = [".", ",", ";", "!", "?", "`"]    # phrasal boundary tokens
    punct_tokens = [observed_translation[x] for x in punct_strs if x in observed_translation]

    # draw emission and transition probabilities from a uniform probability distribution
    for prev in range(model.num_hidden):
        for state in range(model.num_hidden):
            # assign initial uniform with random offset counts
            model.transitions[prev, state] = 10 + random.random()
            for token in range(model.num_observed):
                model.emissions[state, token] = 10 + random.random()
                model.emissions_joint[prev, state, token] = 10 + random.random()

                # clear any values that are not allowed
                invalid = False
                if state == eos_state and token != eos_token:  # EOS must emit EOS
                    invalid = True
                if state == bos_state and token != bos_token:  # BOS must emit BOS
                    invalid = True
                if use_stop_state and token in punct_tokens:  # only STOP state can emit punctuation
                    invalid = state != stop_state

                if invalid:
                    model.emissions[state, token] = 0
                    model.emissions_joint[prev, state, token] = 0

            # clear any values that are not allowed
            invalid = False
            if prev == b_state and state != i_state:    # B can exclusively go to I, so clear value if not i
                invalid = True
            if state == i_state and prev != i_state and prev != b_state:   # only B and I can go to I
                invalid = True
            if prev == bos_state and state == bos_state:  # BOS state cannot transition to itself
                invalid = True
            if state == bos_state:  # BOS state cannot be transitioned to
                invalid = True
            if prev == eos_state:  # EOS state has no transitions
                invalid = True

            if invalid:
                model.transitions[prev, state] = 0
                model.emissions_joint[prev, state, :] = 0

            # normalize values to achieve probability distribution
            total_emit = sum(model.emissions[state, :])
            if total_emit != 0:
                model.emissions[state, :] /= total_emit
            total_joint = sum(model.emissions_joint[prev, state, :])
            if total_joint != 0:
                model.emissions_joint[prev, state, :] /= total_joint

        total_trans = sum(model.transitions[prev, :])
        if total_trans != 0:
            model.transitions[prev, :] /= total_trans


def get_data_tuples(folder, file):
    result = []
    with open("datasets/" + folder + "/" + file + ".txt", encoding='utf-8') as data_file:
        line = data_file.readline()
        while line.strip() != "":
            result.append(literal_eval(line))   # parse line as python tuple
            line = data_file.readline()
    return result


def process_data(data_tuples: list, replace_this: bool, replace_num: bool):
    sequences = []
    for data_tuple in data_tuples:
        card_name = data_tuple[0] if replace_this else ""
        card_text = data_tuple[1]
        tokens = preprocessor.tokenizer(card_name, card_text)
        processed_tokens = preprocessor.token_replacer(card_name, tokens, replace_this, replace_num)
        sequences.append(processed_tokens)
    return sequences


def run_project_variant(dataset: str, epochs: int, use_prlg: bool, use_dev: bool, replace_this: bool, replace_num: bool, use_stop_state: bool):
    # select cards to load (either individual dataset or all)
    def load_dataset_or_all(file):
        if dataset == "all":
            temp = get_data_tuples("mtg", file)
            temp.extend(get_data_tuples("hearthstone", file))
            temp.extend(get_data_tuples("keyforge", file))
            temp.extend(get_data_tuples("yugioh", file))
        else:
            temp = get_data_tuples(dataset, file)
        return temp

    dev_or_train_file = "dev" if use_dev else "train"
    data = load_dataset_or_all(dev_or_train_file)
    test_data = load_dataset_or_all("test")

    # process the tokens and convert to sequences of ids and corresponding lookup tables
    str_token_sequences = process_data(data, replace_this, replace_num)
    str_test_sequences = process_data(test_data, replace_this, replace_num)
    token_sequences, token_translations, token_lookup = tokenizer.convert_token_sequences_to_ids(str_token_sequences, "BOS", "EOS")
    hidden_state = ["B", "I", "O"]
    if use_stop_state:
        hidden_state.append("STOP")
    hidden_sequences, hidden_translations, hidden_lookup = tokenizer.convert_token_sequences_to_ids([hidden_state], "BOS", "EOS")
    token_frequencies = tokenizer.get_frequencies_from_sequences(token_sequences)

    if use_prlg:
        model = PRLG(len(hidden_lookup), len(token_lookup), hidden_translations["BOS"], hidden_translations["EOS"])
    else:
        model = HMM(len(hidden_lookup), len(token_lookup), hidden_translations["BOS"], hidden_translations["EOS"])

    init_probabilities(model, hidden_translations, token_translations, use_stop_state)
    model.test_validity()
    cascade_parser = CascadeParse(model, token_frequencies, hidden_translations["B"], hidden_translations["I"])

    for e in range(epochs):
        print("epoch", e)
        # TODO: re-randomize sequences
        for i in range(len(token_sequences)):
            #print(token_sequences[i])
            #print(i, tokenizer.convert_id_sequence_to_tokens(token_sequences[i], token_lookup))
            model.em_update(token_sequences[i])

        # TODO: report perplexity

        #rand_test_sequence = str_test_sequences[random.randint(0, len(str_test_sequences))]
        #rand_token_sequence = tokenizer.convert_token_sequence_to_ids(rand_test_sequence, token_translations, "BOS", "EOS")
        rand_token_sequence = token_sequences[random.randint(0, len(token_sequences)-1)]  # TODO: use test dataset
        root = cascade_parser.parse(rand_token_sequence)    # TODO: OOV
        root.print(True)
        print()
        root.print(True, token_lookup)
        print()

        most_likely_sequence = model.viterbi(rand_token_sequence)
        print(most_likely_sequence)
        if most_likely_sequence:
            print(tokenizer.convert_id_sequence_to_tokens(most_likely_sequence, hidden_lookup))

    # TODO: if not replace_this, run accuracy, F1, etc on <this> chunking


run_project_variant("keyforge", 100, use_prlg=True, use_dev=True, replace_this=True, replace_num=True, use_stop_state=False)
