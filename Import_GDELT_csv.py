import pandas as pd
import numpy as np
from histograms import Dictogram
import random
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import json



############################################## FUNCTIONS ################################################
def make_markov_model(data):
    markov_model = dict()
    for i in range(0, len(data)-1):
        if data[i] in markov_model:
            # We have to just append to the existing histogram
            markov_model[data[i]].update([data[i+1]])
        else:
            markov_model[data[i]] = Dictogram([data[i+1]])
    #print(markov_model)
    return markov_model

def generate_random_start(model):
    # To just generate any starting event uncomment line:
    # return random.choice(model.keys())
    if 'END' in model:
        seed_event = 'END'
        while seed_event == 'END':
            seed_event = model['END'].return_weighted_random_word()
        return seed_event
    return random.choice(list(model.keys()))


def generate_random_events(length, markov_model):
    current_event = generate_random_start(markov_model)
    event_list = [current_event]
    for i in range(0, length):
        current_dictogram = markov_model[current_event]
        random_weighted_event = current_dictogram.return_weighted_random_word()
        current_event = random_weighted_event
        event_list.append(current_event)
    #event_list[0] = event_list[0]
    #return ' '.join(str(event_list)) + '.'
    return event_list

def make_higher_order_markov_model(order, data):
    markov_model = dict()
    for i in range(0, len(data)-order):
        # Create the window
        window = tuple(data[i: i+order])
        # Add to the dictionary
        if window in markov_model:
            # We have to just append to the existing Dictogram
            markov_model[window].update([data[i+order]])
        else:
            markov_model[window] = Dictogram([data[i+order]])
    return markov_model


def main():
    #IMPORT CSV
    EVENTFILE = 'russia.csv' #Replace this string with your event file name. The file should be located in the same directory as the script.
    df = pd.read_csv(EVENTFILE)
    df['EVENTCODE'] = df.EVENTCODE.astype(str)
    EventCodeLookup = pd.read_csv('EventCodeLookup.csv')
    pd.to_numeric(EventCodeLookup["Code"])

    #get EventCodes size
    numRows = df['EVENTCODE'].shape[0]

    #create eventcode list and markov chain dictionary
    eventCodeList = list()
    #markovDict = defaultdict()
    markovDict = {}
    numOfEvents = 20 #This is the number of predicted events.
    trainPercentage = 0.8
    trainrows = int(numRows * trainPercentage)
    train = list(df['EVENTCODE'])
    train = train[:trainrows]

    #CREATE EVENTCODELIST
    for i in range(numRows-1):
        tup = (df['EVENTCODE'].iloc[i], df['EVENTCODE'].iloc[i+1])
        eventCodeList.append(tup)


    #CREATE MARKOV CHAIN MODEL
    markovDict = make_markov_model(train)
    markovDict_prob = {}
    markovDict_2nd_prob = {}
    #t_df = pd.DataFrame(np.zeros((298, 298)), columns=EventCodeLookup['Code'], index=EventCodeLookup['Code'])
    states = list(EventCodeLookup['Code'])
    markovDict_2nd = make_higher_order_markov_model(2, train)
    #print(markovDict)

    #CREATE TRANSITION MATRIX FOR 1ST-ORDER MC
    for k, v in markovDict.items():
        #print(k ,type(k))
        temptot = sum(v.values())
        tempprobdict = {}
        for kt, vt in v.items():
            tempprobdict[kt] = vt / temptot
            #t_df.loc[k, kt] = vt / temptot
        markovDict_prob[k] = tempprobdict
    #print(markovDict_prob) #Uncomment to see what the MC Model looks like

    #CREATE TRANSITION MATRIX FOR 2ND-ORDER MC
    for k, v in markovDict_2nd.items():
        temptot = sum(v.values())
        tempprobdict = {}
        for kt, vt in v.items():
            tempprobdict[kt] = vt / temptot
            #t_df.loc[k, kt] = vt / temptot
        markovDict_2nd_prob[k] = tempprobdict
    #print(markovDict_2nd_prob) #Uncomment to see what the 2nd order MC Model looks like

    print('total number of possible events in the given dataset (1st order): ' len(markovDict.keys()))
    print('total number of possible events in the given dataset (2nd order): ' len(markovDict_2nd.keys()))
    #results = [prediction == truth for prediction, truth in zip(y_predicted, y)]
    #accuracy = float(results.count(True)) / float(len(results))

    #This will generate list of events with 'numOfEvents' length from Markov Model.
    MarkovRandomEvents = generate_random_events(numOfEvents, make_markov_model(train))
    Events = list()
    for code in MarkovRandomEvents:
        temp = list(EventCodeLookup['Description'].loc[EventCodeLookup['Code'] == code])[0]
        Events.append(temp)
    #print(Events) #This prints the forecasted events.


    ######################################### OUTPUT TO JSON FILE ##########################################
    mc_data = list()

    for k,v in markovDict_prob.items():
        for kt, kv in v.items():
            tempdict = {}
            tempdict['source'] = str(k)
            tempdict['target'] = str(kt)
            tempdict['value'] = str(kv)
            mc_data.append(tempdict)

    JSONFILE = 'MC_2nd_json.json'
    with open(JSONFILE, 'w') as f:
       #json_data = json.dump(mc_data, f)
       json.dump(mc_data, f)


if __name__ == "__main__":
    main()