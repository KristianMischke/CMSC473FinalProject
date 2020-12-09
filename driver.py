from ast import literal_eval
from hmm import HMM, PRLG
from cascade_parser import CascadeParse
import tokenizer, preprocessor
import random
import numpy as np


def init_probabilities(model, hidden_translation, observed_translation):
    b_state = hidden_translation["B"]
    i_state = hidden_translation["B"]
    bos_state = hidden_translation["BOS"]
    eos_state = hidden_translation["EOS"]

    bos_token = observed_translation["BOS"]
    eos_token = observed_translation["EOS"]

    # draw emission and transition probabilities from a uniform probability distribution
    for i in range(model.num_hidden):
        for j in range(model.num_observed):
            model.emissions[i][j] = 10 + random.random()
            if j == eos_token and i != eos_state:   # EOS must emit EOS
                model.emissions[i][j] = 0
            if j == bos_token and i != bos_state:   # BOS must emit BOS
                model.emissions[i][j] = 0
        total = sum(model.emissions[i][:])
        model.emissions[i][:] /= total

    for i in range(model.num_hidden):
        for j in range(model.num_hidden):
            model.transitions[i][j] = 10 + random.random()
            if i == b_state and j != i_state:   # B can only go to I
                model.emissions[i][j] = 0
            if j == i_state and (i != i_state or i != b_state):  # only B and I can go to I
                model.emissions[i][j] = 0
        total = sum(model.transitions[i][:])
        model.transitions[i][:] /= total

    # TODO: figure out correct initializing probabilities
    for i in range(model.num_hidden):
        for j in range(model.num_hidden):
            for k in range(model.num_observed):
                model.emissions_joint[i][j][k] = 10 + random.random()
            total = sum(model.emissions_joint[i][j][:])
            model.emissions_joint[i][j][:] /= total


def get_data_tuples(folder, file):
    result = []
    with open("datasets/" + folder + "/" + file + ".txt", encoding='utf-8') as data_file:
        line = data_file.readline()
        while line != "":
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


def run_project_variant(dataset: str, epochs: int, use_prlg: bool, use_dev: bool, replace_this: bool, replace_num: bool):
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
    hidden_sequences, hidden_translations, hidden_lookup = tokenizer.convert_token_sequences_to_ids([["B", "I", "O", "STOP"]], "BOS", "EOS")
    token_frequencies = tokenizer.get_frequencies_from_sequences(token_sequences)

    if use_prlg:
        model = PRLG(len(hidden_lookup), len(token_lookup), hidden_translations["BOS"], hidden_translations["EOS"])
    else:
        model = HMM(len(hidden_lookup), len(token_lookup), hidden_translations["BOS"], hidden_translations["EOS"])

    init_probabilities(model, hidden_translations, token_translations)
    model.test_validity()
    cascade_parser = CascadeParse(model, token_frequencies, hidden_translations["B"], hidden_translations["I"])

    for e in range(epochs):
        print("epoch", e)
        # TODO: re-randomize sequences
        for i in range(len(token_sequences)):
            #print(token_sequences[i])
            print(i, tokenizer.convert_id_sequence_to_tokens(token_sequences[i], token_lookup))
            model.em_update(token_sequences[i])
        # TODO: report perplexity

        #rand_test_sequence = str_test_sequences[random.randint(0, len(str_test_sequences))]
        #rand_token_sequence = tokenizer.convert_token_sequence_to_ids(rand_test_sequence, token_translations, "BOS", "EOS")
        rand_token_sequence = token_sequences[random.randint(0, len(token_sequences))]  # TODO: use test dataset
        root = cascade_parser.parse(rand_token_sequence)    # TODO: OOV
        root.print(True)
        print()
        root.print(True, token_lookup)
        print()

        print(model.viterbi(rand_token_sequence))

    # TODO: if not replace_this, run accuracy, F1, etc on <this> chunking


run_project_variant("keyforge", 20, False, True, True, True)
