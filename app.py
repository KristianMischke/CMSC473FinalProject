# stores card name under "faceName". type=String
# also can get card name using "asciiName". type=String
#                           ^card name no special unicode characters
import json
import numpy as np
import glob


f = open('AllPrintings.json',)
data = json.load(f)
nameList = []
ruleList = []
star = "★"


def writeToFile(list, fileName):
    with open(fileName, 'w') as f:
        for item in list:
            f.write("%s\n" % item)


# print(data['data'].keys())
for key in data['data'].keys():
    for i in data['data'][key]['cards']:
        # ignore cards with stars in the number element, and name duplicates
        if star not in i['number']:
            if i['name'] not in nameList:
                # get the name of the given card
                nameList.append(i['name'])
                try:
                    # gets the text of the card, if that card element does not
                    # have a text element, we print nothing (in the except KeyError)
                    # print("RULE TEXT: " + i['text'])
                    ruleList.append(i['text'])
                except KeyError:
                    ruleList.append("")

# print(len(nameList))


# # changes our names and rules from unicode to string.
# # if you want unicode, then remove these 2 lines
# # allows us to print the names and rules without u' prefix
# nameList = [str(r) for r in nameList]
# ruleList = [str(r) for r in ruleList]

# create list of tuples in the format of <cardname, rules>
tupleList = list(zip(nameList, ruleList))

# for i in tupleList:
#     print(i)

# split list into 60-20-20
train, validate, test = np.split(
    tupleList, [int(len(tupleList)*0.6), int(len(tupleList)*0.8)])

writeToFile(train, "train.txt")
writeToFile(validate, "dev.txt")
writeToFile(test, "test.txt")
