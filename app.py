# stores card name under "faceName". type=String
# also can get card name using "asciiName". type=String
#                           ^card name no special unicode characters
import json


f = open('AllPrintings.json',)
data = json.load(f)
nameList = []
ruleList = []

for i in data['data']['10E']['cards']:
    # gets the name of the card
    # print("NAME: " + i['name'])
    nameList.append(i['name'])
    try:
        # gets the text of the card, if that card element does not
        # have a text element, we print nothing (in the except KeyError)
        # print("RULE TEXT: " + i['text'])
        ruleList.append(i['text'])
    except KeyError:
        print("")

# changes our names and rules from unicode to string.
# if you want unicode, then remove these 2 lines
# allows us to print the names and rules without u' prefix
nameList = [str(r) for r in nameList]
ruleList = [str(r) for r in ruleList]

# create list of tuples in the format of <cardname, rules>
tupleList = list(zip(nameList, ruleList))

for i in tupleList:
    print i
