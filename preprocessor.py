
import copy

WRAPPERS = [
    "{", "}",
    "[", "]",
    "(", ")",
    "\"", "\""
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

    if replace_this:
        name_tokens = card_name.split()
        for i in range(len(replaced_tokens) - len(name_tokens)):
            token = " ".join(replaced_tokens[i:i+len(name_tokens)])
            if token == card_name:
                # Remove the name tokens and replace with "<this>"
                del replaced_tokens[i:i+len(name_tokens)]
                replaced_tokens.insert(i, "<this>")

    if replace_number:
        for i in range(len(replaced_tokens)):
            token = replaced_tokens[i]
            if token.isnumeric():
                del replaced_tokens[i]
                replaced_tokens.insert(i, "<number>")
    
    return replaced_tokens
                


def tokenizer(card_name: str, card_text: str) -> str:
    """
    :param card_name: Name of the card being processed
    :param card_text: Description of the card being processed
    :return split_Text: Newly tokenized card text
    """
    # Split the card text by space initially.
    split_text = card_text.split(" ")

    # Process the initial token list backwards so that any new tokens
    #   do not affect our ending target index of 0.
    tokens_left = len(split_text)
    while tokens_left > 0:
        # Current token index being looked at
        i = tokens_left - 1

        # Filter out numbers (GROUPED)
        if any(c.isnumeric() for c in split_text[i]):
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

        # Filter out non-alphanumeric strings (INDIVIDUALLY)
        elif any(not c.isalnum() for c in split_text[i]):
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

    tokenized = tokenizer("", user_input)
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
        tokenized = tokenizer("", user_input)
        replaced_tokens = token_replacer("", tokenized, True, True)
        user_input = input("Enter a string: ")