
import copy

IGNORED_CHARS = [
    "'"
]

CONTRACTIONS = [
    "'s",
    "n't",
    "'ve",
    "'m",
    "'re",
    "'d",
    "'ll"
]

END_PUNCT = [
    "!",
    ".",
    "?"
]

ESCAPE_CHARS = [
    #"t", # Tab
    #"n", # Newline
    "'", # Apostrophe
]


def token_replacer(card_name: str, tokenized_card_text: list, replace_this: bool, replace_number: bool) -> str:
    """
    :param card_name: Name of the card being processed
    :param tokenized_card_text: Tokenized card text
    :param replace_this: Boolean determining whether to replace tokens with card name with "<this>"
    :param replace_number: Boolean determining whether to replace numerical values with "<number>"
    :return replaced_tokens: Newly replaced tokenized card text
    """
    replaced_tokens = copy.deepcopy(tokenized_card_text)

    for i in range(len(replaced_tokens)):
        token = replaced_tokens[i]
        if token == card_name and replace_this:
            # Remove the name token and replace with "<this>"
            del replaced_tokens[i]
            replaced_tokens.insert(i, "<this>")

        elif token.isnumeric() and replace_number:
            # Remove number and replace with "<number>"
            del replaced_tokens[i]
            replaced_tokens.insert(i, "<number>")
    
    return replaced_tokens
                

def tokenizer(card_name: str, card_text: str) -> str:
    """
    :param card_name: Name of the card being processed
    :param card_text: Description of the card being processed
    :return split_Text: Newly tokenized card text
    """
    name_in_text = card_name in card_text

    # Split the card text by space initially.
    split_text = card_text.split(" ")

    # Combine any tokens that were split from multi-part names
    if name_in_text:
        name_token_len = len(card_name.split(" "))
        # Check up to the last index possible where card name is split as well
        i = 0
        while i <= len(split_text) - name_token_len:
            # Since we only split by spaces initially, we can accurately find
            # potential name locations by simply joining `name_token_len` tokens from split_text
            potential_name = " ".join(split_text[i:i+name_token_len])
            if card_name in potential_name:
                # Remove the associated tokens. We can rebuild the tokens using potential_name
                del split_text[i:i+name_token_len]

                # Find the indexes of the beginning and end.
                # Before loc and after end_loc are our "extra" strings.
                loc = potential_name.index(card_name)
                end_loc = loc + len(card_name)

                # To keep the same order, insert ending extras first
                if len(card_name) < len(potential_name) - loc:
                    re_insert = potential_name[end_loc:]
                    re_insert = re_insert.split(" ")
                    split_text[i:i] = re_insert
                
                # Insert the card name
                split_text.insert(i, potential_name[loc:end_loc])

                # Complete the new split by adding the beginning extras
                if loc > 0:
                    re_insert = potential_name[:loc]
                    re_insert = re_insert.split(" ")
                    split_text[i:i] = re_insert

            i += 1

    # Process the initial token list backwards so that any new tokens
    #   do not affect our ending target index of 0.
    tokens_left = len(split_text)
    while tokens_left > 0:
        # Current token index being looked at
        i = tokens_left - 1

        # Strip out token if it isn't a card name
        if split_text[i] != card_name:

            # Escape sequences
            # Start here, to convert the token into a regular word before continuing if necessary.
            if "\\" in split_text[i]:
                token = split_text[i]
                # Only look for backslashes before the last char (escape characters are 2 long)
                loc = token.index("\\", 0, len(token)-1)
                if token[loc+1] == "'":
                    split_text[i] = token[:loc] + token[loc+1:]
                
                    # Check current once again in case there are more escaped characters
                    if split_text[i]:
                        tokens_left += 1
                    else:
                        split_text.pop(i)
                
                elif token[loc+1] == "n" or token[loc+1] == "t":
                    split_text[i] = token[:loc]

                    count = 1  # We are counting how many times backwards we re-check.

                    # There's a string after escape character, insert first.
                    if token[loc+2:]:
                        count += 1  # Another string means another space to check
                        split_text.insert(i+1, token[loc+2:])

                    # Insert the string after the escape character
                    split_text.insert(i+1, "\\{}".format(token[loc+1]))
                
                    # Check starting from last string once again in case there are more escaped characters
                    if split_text[i]:
                        tokens_left += 1 + count
                    else:
                        # Case: "\n" alone would have a count of 1, causing an infinite loop.
                        if count > 1:
                            tokens_left += count
                        split_text.pop(i)

            # Current token ends in punctuation (STRICTLY sentence enders)
            elif split_text[i][-1:] in END_PUNCT:
                ending = split_text[i][-1:]
                if ending in END_PUNCT:
                    # Strip punctuation off of token
                    split_text[i] = split_text[i][:-1]

                    # Removed punctuation ending, append new token after current
                    split_text.insert(i+1, ending)
                    # Check current once again if current still contains characters
                    if split_text[i]:
                        tokens_left += 1
                    else:
                        split_text.pop(i)

            # Filter out numbers (GROUPED)
            elif any(c.isnumeric() for c in split_text[i]):
                # Store 'divided' strings in a list. [Ex: "123", "sometext", "456"]
                div = []
                token = split_text[i]
                start = 0  # First index of our string before non-alphanumeric character
                storing_alpha = True  # Storing alphabet strings
                for j, letter in enumerate(token):
                    # XOR, our boolean acts as a "switch" to know what we're storing currently
                    if ((not letter.isnumeric() and storing_alpha) 
                            or (letter.isnumeric() and not storing_alpha)):
                        if start != j:  # Prevent empty string
                            div.append(token[start:j])
                        start = j
                        storing_alpha = not(storing_alpha)

                # Append rest of unchecked string once concluded
                if token[start:]:
                    div.append(token[start:])
                
                # Insert (extend) list at index
                split_text.pop(i)
                split_text[i:i] = div

                # Run the check through our newly splitted tokens 
                # (if we find a number alone, don't re-check it)
                if len(div) > 1:
                    tokens_left += len(div)

            # Existence of an apostrophe may imply contractions
            elif "'" in split_text[i]: 
                # Cover contractions and possessions ('s, n't, etc.)
                for c in CONTRACTIONS:
                    if c == split_text[i][-len(c):]:
                        split_text.insert(i+1, split_text[i][-len(c):])
                        split_text[i] = split_text[i][:-len(c)]
                        # Check current once again if current still contains characters
                        if split_text[i]:
                            tokens_left += 1
                        else:
                            split_text.pop(i)

            # Filter out non-alphanumeric strings (INDIVIDUALLY)
            elif any(not c.isalnum() and c not in IGNORED_CHARS for c in split_text[i]):
                # Store 'divided' strings in a list. [Ex: "(", "sometext", ")"]
                div = []
                token = split_text[i]
                start = 0  # First index of our string before non-alphanumeric character
                storing_string = True
                for j, letter in enumerate(token):
                    # If non-letter, and we are currently storing string, save string then continue.
                    # If letter, and we are not storing a string, save non-letter, 
                    #       then mark that we are now storing a string
                    if ((not letter.isalnum() and storing_string) or 
                            (letter.isalnum() and not storing_string)):
                        if start != j:  # Prevent empty string
                            div.append(token[start:j])
                        storing_string = not(storing_string)
                        start = j
                    
                    # If non-letter, and we are not storing a string, save the character alone
                    elif not letter.isalnum() and not storing_string:
                        if start != j:  # Prevent empty string
                            div.append(token[start:j])
                        start = j
                        
                # Append rest of unchecked string once concluded
                if token[start:]:
                    div.append(token[start:])
                
                # Insert (extend) list at index
                split_text.pop(i)
                split_text[i:i] = div

        # Continue to next token
        tokens_left -= 1

    return split_text


if __name__ == "__main__":
    # Premade test
    user_input = "Morph {W} (You may cast this card face down as a 2/2 creature for {3}. Turn it face up any time for its morph cost.)\nWhen Daru Mender is turned face up, regenerate target creature."

    tokenized = tokenizer("Daru Mender", user_input)
    print()
    print(" ".join(tokenized))
    test = "Morph { W } ( You may cast this card face down as a 2 / 2 creature for { 3 } . Turn it face up any time for its morph cost . ) \n When Daru Mender is turned face up , regenerate target creature ."
    print(test)
    print("Matches:", " ".join(tokenized) == test, "\n")

    replaced_tokens = token_replacer("Daru Mender", tokenized, True, True)
    print(" ".join(replaced_tokens))
    test = "Morph { W } ( You may cast this card face down as a <number> / <number> creature for { <number> } . Turn it face up any time for its morph cost . ) \n When <this> is turned face up , regenerate target creature ."
    print(test)
    print("Matches:", " ".join(replaced_tokens) == test, "\n")

    # User input test
    while user_input != "quit":
        user_name = input("Enter a name: ")
        user_input = input("Enter a string: ")
        tokenized = tokenizer(user_name, user_input)
        replaced_tokens = token_replacer(user_name, tokenized, True, True)
        print(user_name, user_input)
        print(tokenized)
        print(replaced_tokens)