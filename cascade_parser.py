from typing import Union
from hmm import HMM, PRLG
import tokenizer


class CascadeParse:
    class CollapsedToken:
        def __init__(self, token: int, children: Union[list, type(None)]):
            self.token = token
            self.children = children

        def print(self, show_pseudo: bool, token_lookup: Union[dict, type(None)] = None):
            str_token = str(self.token)
            if token_lookup is not None:
                str_token = token_lookup[self.token]

            if self.children is None or len(self.children) == 0:
                print(str_token, end="")
            else:
                print("(", end="")
                if show_pseudo:
                    print(f"{str_token}: ", end="")
                for x in self.children:
                    x.print(show_pseudo, token_lookup)
                    print(" ", end="")
                print(")", end="")

    def __init__(self, model: HMM, token_frequencies: dict, begin_chunk_state, continue_chunk_state):
        self.model = model
        self.token_frequencies = token_frequencies
        self.begin_chunk_state = begin_chunk_state
        self.continue_chunk_state = continue_chunk_state

    def parse(self, sequence_ids):
        # init result as flat array of ids
        result = []
        result_ids = []
        for token_id in sequence_ids:
            result.append(self.CollapsedToken(token_id, None))
            result_ids.append(token_id)

        # cascaded chunking loop
        done = False
        while not done:
            best_match = self.model.viterbi(result_ids)
            iteration_result = []
            if best_match is not None:
                # viterbi successful, so do next iteration

                num_chunks = 0
                highest_freq = -1
                highest_freq_chunk_token = -1
                current_chunk = []
                for i in range(len(best_match)):
                    added_to_chunk = False
                    if len(current_chunk) > 0:
                        # inside chunk
                        if best_match[i] == self.continue_chunk_state:
                            # add token to chunk
                            current_chunk.append(result[i])
                            added_to_chunk = True

                            # update highest frequency token in chunk
                            test_freq = self.token_frequencies[result_ids[i]]
                            if test_freq > highest_freq:
                                highest_freq = test_freq
                                highest_freq_chunk_token = result_ids[i]
                        else:
                            # chunk ended, so append to iteration result, and reset current chunk variables
                            iteration_result.append(self.CollapsedToken(highest_freq_chunk_token, current_chunk))
                            current_chunk = []
                            highest_freq_chunk_token = -1
                            highest_freq = -1

                    if best_match[i] == self.begin_chunk_state:
                        # chunk began
                        current_chunk.append(result[i])
                        highest_freq = self.token_frequencies[result_ids[i]]
                        highest_freq_chunk_token = result_ids[i]
                        num_chunks += 1
                    elif not added_to_chunk:
                        # not in chunk, so append previous result to iteration result
                        iteration_result.append(result[i])
                result = iteration_result
                result_ids = [x.token for x in result]
                if num_chunks == 0:
                    done = True
            else:
                done = True

        return self.CollapsedToken(result_ids[0], result)
