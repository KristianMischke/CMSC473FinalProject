# 

## Installation

Ensure you have Python 3 installed: https://www.python.org/downloads/

1. (Optional) Create a Python virtual environment: https://docs.python.org/3/tutorial/venv.html

2. `python3 -m pip install -r requirements.txt` Install Python packages

3. `python3 driver.py [flags]` Run the model from `driver.py`.


## Usage

- `-d [dataset_name]`, `--dataset [dataset_name]` - Specify a dataset to build the model upon.
- `--oov_thresh [threshold]` - Specify the (inclusive) threshold of appearances required of a  token in a set before converting it to Out Of Vocabulary (OOV).
- `-e [epochs]`, `--epochs [epochs]` - Number of epochs/iterations to run.
- `-i [save_interval]`, `--interval [save_interval]` - Specify an epoch interval for saving the model.
- `--load_model_path [path]` - Load a saved model from an existing model at the path provided.
- `-s [path]`, `--save_model_dir [path]` - Specify the directory path to save models to.
- `--treebank_path [path]` - Specify the path to raw Treebank-3 data

Boolean flags
- `--use_lowercase` - Convert all tokens to lowercase before training.
- `--prlg` - Use PRLG instead of HMM.
- `--dev` - Use the dev data. (If not used, train file is used instead)
- `--replace_this` - Replace the names of cards with "<this>" in rules-text during preprocessing.
- `--replace_num` - Replace numerical tokens with "<number>" during preprocessing.
- `--use_stop_state` - Use the STOP state.
- `--perplexity_test` - Run perplexity test.
- `--parse_test_set` - Generate file with parse trees of the test set.
- `--all_true` - Mark all boolean flags as True (shortcut)

**Example run:** `python3 driver.py -d mtg -e 50 -i 2 --all_true`