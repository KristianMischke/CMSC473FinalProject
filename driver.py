import os
import sys
import pickle
import random
import time
from ast import literal_eval
from typing import Union

import preprocessor
import tokenizer
from cascade_parser import CascadeParse
from hmm import HMM, PRLG


def init_probabilities(model, hidden_translation, observed_translation, use_stop_state: bool):
    b_state = hidden_translation["B"]
    i_state = hidden_translation["I"]
    bos_state = hidden_translation["BOS"]
    eos_state = hidden_translation["EOS"]
    stop_state = hidden_translation["STOP"] if use_stop_state else None

    bos_token = observed_translation["BOS"]
    eos_token = observed_translation["EOS"]
    punct_strs = [".", ",", ";", "!", "?", "`"]  # phrasal boundary tokens
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
                if use_stop_state and (token in punct_tokens):  # only STOP state can emit punctuation
                    invalid = state != stop_state

                if invalid:
                    model.emissions[state, token] = 0
                    model.emissions_joint[prev, state, token] = 0

            # clear any values that are not allowed
            invalid = False
            if prev == b_state and state != i_state:  # B can exclusively go to I, so clear value if not i
                invalid = True
            if state == i_state and prev != i_state and prev != b_state:  # only B and I can go to I
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
            result.append(literal_eval(line))  # parse line as python tuple
            line = data_file.readline()
    return result


def get_treebank_data(treebank_raw_path, data_division, use_lowercase: bool):
    all_lines = []

    for subdir, dirs, files in os.walk(treebank_raw_path):
        try:
            if (data_division == "train" and int(os.path.basename(subdir)) <= 22) or \
               (data_division == "test" and int(os.path.basename(subdir)) == 23) or \
               (data_division == "dev" and int(os.path.basename(subdir)) == 24):
                for file in files:
                    with open(os.path.join(subdir, file), encoding='utf-8') as data_file:
                        all_lines.extend(data_file.readlines())
        except ValueError:
            pass

    result = []
    current = ""
    for i in range(len(all_lines)):
        if (all_lines[i].strip() == "" or all_lines[i].strip() == ".START") and len(current) > 0:
            if use_lowercase:
                result.append(current.lower())
            else:
                result.append(current)
            current = ""
        else:
            current += all_lines[i]

    return result


def process_data(data_tuples: list, replace_this: bool, replace_num: bool):
    sequences = []
    for data_tuple in data_tuples:
        if isinstance(data_tuple, str):
            card_name = ""
            card_text = data_tuple
        else:
            card_name = data_tuple[0] if replace_this else ""
            card_text = data_tuple[1]
        tokens = preprocessor.tokenizer(card_name, card_text)
        processed_tokens = preprocessor.token_replacer(card_name, tokens, replace_this, replace_num)
        sequences.append(processed_tokens)
    return sequences


class ProjectSave:
    def __init__(self,
                 dataset: str,
                 oov_thresh: int,
                 use_lowercase: bool,
                 epochs: int,
                 use_prlg: bool,
                 use_dev: bool,
                 replace_this: bool,
                 replace_num: bool,
                 use_stop_state: bool,
                 save_every_x: int,

                 current_epoch: int,
                 model: HMM,
                 dev_perplexity_table: list,
                 test_perplexity_table: list):
        self.dataset = dataset
        self.oov_thresh = oov_thresh
        self.use_lowercase = use_lowercase
        self.epochs = epochs
        self.use_prlg = use_prlg
        self.use_dev = use_dev
        self.replace_this = replace_this
        self.replace_num = replace_num
        self.use_stop_state = use_stop_state
        self.save_every_x = save_every_x

        self.current_epoch = current_epoch
        self.model = model
        self.dev_perplexity_table = dev_perplexity_table
        self.test_perplexity_table = test_perplexity_table


def run_project_variant(dataset: str,
                        oov_thresh: int,
                        use_lowercase: bool,
                        epochs: int,
                        use_prlg: bool,
                        use_dev: bool,
                        replace_this: bool,
                        replace_num: bool,
                        use_stop_state: bool,
                        save_every_x: int,
                        load_model_path: Union[str, None],
                        save_model_dir: str,
                        tree_banks_raw_path="D:\\Documents\\treebank_3\\raw\\wsj"):
    if not os.path.isabs(save_model_dir):
        save_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_model_dir)

    start_epoch = 0
    model = None
    dev_perplexity_table = []
    test_perplexity_table = []

    # init from saved model if path specified
    if load_model_path is not None and len(load_model_path) > 0:
        if not os.path.isabs(load_model_path):
            load_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), load_model_path)

        resume_model = pickle.load(open(load_model_path, "rb"))
        dataset = resume_model.dataset
        oov_thresh = resume_model.oov_thresh
        use_lowercase = resume_model.use_lowercase
        if epochs is None or epochs <= 0:   # allow override epochs if it is a non-negative number
            epochs = resume_model.epochs
        use_prlg = resume_model.use_prlg
        use_dev = resume_model.use_dev
        replace_this = resume_model.replace_this
        replace_num = resume_model.replace_num
        use_stop_state = resume_model.use_stop_state
        save_every_x = resume_model.save_every_x
        save_model_dir = os.path.dirname(load_model_path)
        start_epoch = resume_model.current_epoch
        model = resume_model.model
        dev_perplexity_table = resume_model.dev_perplexity_table
        test_perplexity_table = resume_model.test_perplexity_table

    # select cards to load (either individual dataset or all)
    def load_dataset_or_all(file):
        if dataset == "treebank_3":
            temp = get_treebank_data(tree_banks_raw_path, file, use_lowercase)
        else:
            if dataset == "all":
                temp = get_data_tuples("mtg", file)
                temp.extend(get_data_tuples("hearthstone", file))
                temp.extend(get_data_tuples("keyforge", file))
                temp.extend(get_data_tuples("yugioh", file))
            else:
                temp = get_data_tuples(dataset, file)

            if use_lowercase and dataset != "treebank_3":
                for j in range(len(temp)):
                    temp[j] = (temp[j][0].lower(), temp[j][1].lower())

        return temp

    dev_or_train_file = "dev" if use_dev else "train"
    data = load_dataset_or_all(dev_or_train_file)
    test_data = load_dataset_or_all("test")

    # process the tokens and convert to sequences of ids and corresponding lookup tables
    str_token_sequences = process_data(data, replace_this, replace_num)
    str_test_sequences = process_data(test_data, replace_this, replace_num)
    token_sequences, token_translations, token_lookup = tokenizer.convert_token_sequences_to_ids(str_token_sequences,
                                                                                                 "BOS", "EOS", oov_thresh, "OOV")
    # convert test dataset (with oov)
    test_sequences = []
    for s in str_test_sequences:
        test_sequences.append(tokenizer.convert_token_sequence_to_ids(s, token_translations, "BOS", "EOS", "OOV"))

    hidden_state = ["B", "I", "O"]
    if use_stop_state:
        hidden_state.append("STOP")
    hidden_sequences, hidden_translations, hidden_lookup = tokenizer.convert_token_sequences_to_ids([hidden_state],
                                                                                                    "BOS", "EOS")
    token_frequencies = tokenizer.get_frequencies_from_sequences(token_sequences)

    if model is None:
        # if we didn't load a model, create a new one and init the probabilities
        if use_prlg:
            model = PRLG(len(hidden_lookup), len(token_lookup), hidden_translations["BOS"], hidden_translations["EOS"])
        else:
            model = HMM(len(hidden_lookup), len(token_lookup), hidden_translations["BOS"], hidden_translations["EOS"])
        init_probabilities(model, hidden_translations, token_translations, use_stop_state)

    model.test_validity()
    cascade_parser = CascadeParse(model, token_frequencies, hidden_translations["B"], hidden_translations["I"])

    for e in range(start_epoch, epochs):
        iteration_start_time = time.time()
        print("epoch", e)
        model.em_update(token_sequences)
        print("elapsed time:", time.time() - iteration_start_time)

        dev_perplexity = model.compute_corpus_perplexity(token_sequences)
        test_perplexity = model.compute_corpus_perplexity(test_sequences)
        dev_perplexity_table.append((e, dev_perplexity))
        test_perplexity_table.append((e, test_perplexity))
        print("dev perplexity:", dev_perplexity)
        print("test perplexity:", test_perplexity)

        i = random.randint(0, len(str_test_sequences))
        rand_test_sequence = str_test_sequences[i]
        rand_token_sequence = test_sequences[i]
        #rand_token_sequence = token_sequences[random.randint(0, len(token_sequences) - 1)]  # TODO: use test dataset
        print(rand_test_sequence)
        print(tokenizer.convert_id_sequence_to_tokens(rand_token_sequence, token_lookup))
        root = cascade_parser.parse(rand_token_sequence)
        root.print(True)
        print()
        root.print(True, token_lookup)
        print()

        most_likely_sequence = model.viterbi(rand_token_sequence)
        print(most_likely_sequence)
        if most_likely_sequence:
            print(tokenizer.convert_id_sequence_to_tokens(most_likely_sequence, hidden_lookup))

        # save model at end of epoch
        if save_every_x >= 1 and e % save_every_x == 0 and save_model_dir is not None:
            save_obj = ProjectSave(dataset,
                                   oov_thresh,
                                   use_lowercase,
                                   epochs,
                                   use_prlg,
                                   use_dev,
                                   replace_this,
                                   replace_num,
                                   use_stop_state,
                                   save_every_x,
                                   e+1,     # resume when loading at epoch + 1
                                   model,
                                   dev_perplexity_table,
                                   test_perplexity_table)
            if not os.path.exists(save_model_dir):
                os.mkdir(save_model_dir)
            path = os.path.join(save_model_dir, f"model_{str(e).zfill(3)}.p")
            pickle.dump(save_obj, open(path, "wb"))

    if save_model_dir is not None:
        with open(os.path.join(save_model_dir, "dev_perplexity_history.csv"), 'w', encoding='utf-8') as f:
            f.write(f"epoch,perplexity\n")
            for e, perplexity in dev_perplexity_table:
                f.write(f"{str(e)},{str(perplexity)}\n")
        with open(os.path.join(save_model_dir, "test_perplexity_history.csv"), 'w', encoding='utf-8') as f:
            f.write(f"epoch,perplexity\n")
            for e, perplexity in test_perplexity_table:
                f.write(f"{str(e)},{str(perplexity)}\n")


# TODO: @Min, command line arguments for each of the parameters of this function
# maybe like: -dataset=mtg --use_prlg etc...
# if load_model_path is assigned then you don't need to specify the other arguments, but user can override epochs=
# otherwise set epochs to zero and it will use the one loaded from the file
# run_project_variant(dataset="keyforge",
#                     oov_thresh=1,
#                     use_lowercase=True,
#                     epochs=100,
#                     use_prlg=True,
#                     use_dev=True,
#                     replace_this=True,
#                     replace_num=True,
#                     use_stop_state=True,
#                     save_every_x=10,
#                     load_model_path=None,  # "saved_models/keyforge_prlg_r_dev/model_090.p",
#                     save_model_dir="saved_models/keyforge_prlg_r_dev"
#                     )

# TEMP args for training
if "treebank" in sys.argv:
    use_prlg = "prlg" in sys.argv
    run_project_variant(dataset="treebank_3",
                        oov_thresh=1,
                        use_lowercase=True,
                        epochs=100,
                        use_prlg=use_prlg,
                        use_dev=False,
                        replace_this=False,
                        replace_num=False,
                        use_stop_state=False,
                        save_every_x=2,
                        load_model_path=None,
                        save_model_dir=f"saved_models/treebank{'_prlg' if use_prlg else ''}"
                        )
    run_project_variant(dataset="treebank_3",
                        oov_thresh=1,
                        use_lowercase=True,
                        epochs=100,
                        use_prlg=use_prlg,
                        use_dev=False,
                        replace_this=False,
                        replace_num=False,
                        use_stop_state=True,
                        save_every_x=2,
                        load_model_path=None,
                        save_model_dir=f"saved_models/treebank{'_prlg' if use_prlg else ''}_stop"
                        )
else:
    valid = ["all", "mtg", "hearthstone", "yugioh", "keyforge"]

    for iteration in sys.argv:
        if iteration in valid:
            run_project_variant(dataset=iteration,
                                oov_thresh=1,
                                use_lowercase=True,
                                epochs=100,
                                use_prlg=True,
                                use_dev=False,
                                replace_this=False,
                                replace_num=False,
                                use_stop_state=True,
                                save_every_x=2,
                                load_model_path=None,
                                save_model_dir=f"saved_models/{iteration}_stop"
                                )
            run_project_variant(dataset=iteration,
                                oov_thresh=1,
                                use_lowercase=True,
                                epochs=100,
                                use_prlg=True,
                                use_dev=False,
                                replace_this=True,
                                replace_num=True,
                                use_stop_state=True,
                                save_every_x=2,
                                load_model_path=None,
                                save_model_dir=f"saved_models/{iteration}_stop_replace"
                                )
