
# given a conllu chunk sentence, returns the result as a chunked array
def chunk_conllu_sentence(sentence):
   result = []
   for item in sentence:
      if item[2][0] == 'I':            # if item is intermediate, append token to last item in result
         result[-1].append(item[0])
      else:                            # else just add the current item
         result.append([item[0]])
   return result

# input: path to a conllu chunk file with format as described in https://www.clips.uantwerpen.be/conll2000/chunking/
# assumes: file is readable and in correct format
# result: array of sentences (which is an array of the tokens (which is an array of columns in the format))
def get_conllu_chunk_sentences(conllu_path):
   file = open(conllu_path, "r", encoding="utf-8")

   sentences = []

   current_sentence = []
   lines = file.readlines()
   for line in lines:
      columns = line.split()
      if len(columns) == 3:                  # add token to the current sentence
         current_sentence.append(columns)
      else:                                  # sentence is terminated, so add to list, and start a new one
         sentences.append(current_sentence)
         current_sentence = []

   if len(current_sentence) != 0:
      sentences.append(current_sentence)

   file.close()
   return sentences


# testing sanity of function
S = get_conllu_chunk_sentences("./datasets/conllu2000/train.txt")
print("full sentence tokens for S[0]:")
print(S[0])
print()
print("chunked sentence tokens for S[0]:")
print(chunk_conllu_sentence(S[0]))