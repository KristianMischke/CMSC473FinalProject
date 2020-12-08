# stores card name under "faceName". type=String
# also can get card name using "asciiName". type=String
#                           ^card name no special unicode characters
import json
import numpy as np
import glob
import random


def writeToFile(list, fileName):
    with open(fileName, 'w', encoding='utf-8') as f:
        for item in list:
            f.write("%s\n" % item)


def tupleAndWriteToFiles(nameList, ruleList, fileName):
    tupleList = list(zip(nameList, ruleList))

    # randomize the order of the tupleList
    random.shuffle(tupleList)

    # split list into 60-20-20
    train, dev, test = np.split(
        tupleList, [int(len(tupleList)*0.6), int(len(tupleList)*0.8)])

    writeToFile(train, "datasets/" + fileName + "/train.txt")
    writeToFile(dev, "datasets/" + fileName + "/dev.txt")
    writeToFile(test, "datasets/" + fileName + "/test.txt")


def getDataFromJson(jsonFile, name, rules, fileName):
    #f = open('HearthstoneCards.json', encoding='utf-8')
    with open(jsonFile) as data_file:
        jsonObject = json.load(data_file)

    nameList = []
    ruleList = []

    for key in jsonObject:
        nameList.append(key[name])
        try:
            ruleList.append(key[rules])
        except KeyError:
            ruleList.append("")

    tupleAndWriteToFiles(nameList, ruleList, fileName)


def magicData():
    f = open('AllPrintings.json', encoding='utf-8')
    data = json.load(f)
    nameList = []
    ruleList = []
    star = "★"

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

    # create list of tuples in the format of <cardname, rules>
    tupleList = list(zip(nameList, ruleList))

    # split list into 60-20-20
    train, dev, test = np.split(
        tupleList, [int(len(tupleList)*0.6), int(len(tupleList)*0.8)])

    writeToFile(train, "datasets/mtg/train.txt")
    writeToFile(dev, "datasets/mtg/dev.txt")
    writeToFile(test, "datasets/mtg/test.txt")


def hearthstone():

    getDataFromJson("HearthstoneCards.json", "name", "text", "hearthstone")


def keyforge():
    getDataFromJson("KeyforgeCards.json", "card_title",
                    "card_text", "keyforge")


def yugioh():
    with open('YugiohCards.php') as data_file:
        jsonObject = json.load(data_file)

    nameList = []
    ruleList = []

    for key in jsonObject["data"]:
        nameList.append(key["name"])
        try:
            ruleList.append(key["desc"])
        except KeyError:
            ruleList.append("")

    tupleAndWriteToFiles(nameList, ruleList, "yugioh")


def main():
    magicData()
    keyforge()
    hearthstone()
    yugioh()


if __name__ == "__main__":
    main()
