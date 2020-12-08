# stores card name under "faceName". type=String
# also can get card name using "asciiName". type=String
#                           ^card name no special unicode characters
import json
import numpy as np
import glob


def writeToFile(list, fileName):
    with open(fileName, 'w', encoding='utf-8') as f:
        for item in list:
            f.write("%s\n" % item)


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

    writeToFile(train, "dataset/mtg/train.txt")
    writeToFile(dev, "dataset/mtg/dev.txt")
    writeToFile(test, "dataset/mtg/test.txt")


def hearthstone():
    #f = open('HearthstoneCards.json', encoding='utf-8')
    with open('HearthstoneCards.json') as data_file:
        jsonObject = json.load(data_file)

    nameList = []
    ruleList = []

    for key in jsonObject:
        nameList.append(key["name"])
        try:
            ruleList.append(key["text"])
        except KeyError:
            ruleList.append("")

    tupleList = list(zip(nameList, ruleList))

    # split list into 60-20-20
    train, dev, test = np.split(
        tupleList, [int(len(tupleList)*0.6), int(len(tupleList)*0.8)])

    writeToFile(train, "dataset/hearthstone/train.txt")
    writeToFile(dev, "dataset/hearthstone/dev.txt")
    writeToFile(test, "dataset/hearthstone/test.txt")


def keyforge():
    with open('KeyforgeCards.json') as data_file:
        jsonObject = json.load(data_file)

    nameList = []
    ruleList = []

    for key in jsonObject:
        nameList.append(key["card_title"])
        try:
            ruleList.append(key["card_text"])
        except KeyError:
            ruleList.append("")

    tupleList = list(zip(nameList, ruleList))

    # split list into 60-20-20
    train, dev, test = np.split(
        tupleList, [int(len(tupleList)*0.6), int(len(tupleList)*0.8)])

    writeToFile(train, "dataset/keyforge/train.txt")
    writeToFile(dev, "dataset/keyforge/dev.txt")
    writeToFile(test, "dataset/keyforge/test.txt")


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

    tupleList = list(zip(nameList, ruleList))

    # split list into 60-20-20
    train, dev, test = np.split(
        tupleList, [int(len(tupleList)*0.6), int(len(tupleList)*0.8)])

    writeToFile(train, "dataset/yugioh/train.txt")
    writeToFile(dev, "dataset/yugioh/dev.txt")
    writeToFile(test, "dataset/yugioh/test.txt")


def main():
    # magicData()
    # keyforge()
    # hearthstone()
    yugioh()


if __name__ == "__main__":
    main()
