import pandas as pd
import numpy as np
from histograms import Dictogram
import random
from collections import deque
import re
import matplotlib.pyplot as plt

#IMPORT CSV
df = pd.read_csv('20180320161401.22251.events.csv')
EventCodeLookup = pd.read_csv('EventCodeLookup.csv')

#get EventCodes size
numRows = df['EventCode'].shape[0]

#create eventcode list and markov chain dictionary
eventCodeList = list()
markovDict = {}
goldsteinNums = list(df['GoldsteinScale'])

#create eventCodeList 
for i in range(numRows-1):
    tup = (df['EventCode'].iloc[i], df['EventCode'].iloc[i+1])
    eventCodeList.append(tup)


# for tup in eventCodeList:
#     if tup[0] in markovDict:
#         # append the new number to the existing array at this slot
#         markovDict[tup[0]].append(tup[1])
#     else:
#         # create a new array in this slot
#         markovDict[tup[0]] = [tup[1]]


# with open('MarkovChainDict.csv', 'w') as f:
#     [f.write('{0},{1}\n'.format(key, value)) for key, value in markovDict.items()]

test = list(df['EventCode'])
test.append("END")

def make_markov_model(data):
    markov_model = dict()

    for i in range(0, len(data)-1):
        if data[i] in markov_model:
            # We have to just append to the existing histogram
            markov_model[data[i]].update([data[i+1]])
        else:
            markov_model[data[i]] = Dictogram([data[i+1]])
    return markov_model

def generate_random_start(model):
    # To just generate any starting word uncomment line:
    # return random.choice(model.keys())

    # To generate a "valid" starting word use:
    # Valid starting words are words that started a sentence in the corpus
    if 'END' in model:
        seed_word = 'END'
        while seed_word == 'END':
            seed_word = model['END'].return_weighted_random_word()
        return seed_word
    return random.choice(list(model.keys()))


def generate_random_sentence(length, markov_model):
    current_word = generate_random_start(markov_model)
    sentence = [current_word]
    for i in range(0, length):
        current_dictogram = markov_model[current_word]
        random_weighted_word = current_dictogram.return_weighted_random_word()
        current_word = random_weighted_word
        sentence.append(current_word)
    #sentence[0] = sentence[0]
    #return ' '.join(str(sentence)) + '.'
    return sentence

#print(generate_random_start(make_markov_model(test)))
MarkovRandomEvents = generate_random_sentence(5, make_markov_model(test))
pd.to_numeric(EventCodeLookup["Code"])


Events = list()
for code in MarkovRandomEvents:
    temp = list(EventCodeLookup['Description'].loc[EventCodeLookup['Code'] == code])[0]
    Events.append(temp)

#print(Events)


GoldsteinCusum = list()
cusum = 0
for num in goldsteinNums:
    cusum += num
    GoldsteinCusum.append(cusum)

plt.plot(GoldsteinCusum)
plt.title("Cumulative Goldstein Scale between US and Iraq 2003-2004")
plt.show()