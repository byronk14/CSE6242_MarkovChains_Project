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

def generate_random_events_with_fixed_start(length, markov_model, startEvent):
    current_event = startEvent
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


if __name__ == "__main__":
    #IMPORT CSV
    #EVENTFILE = 'results-20180419-222454.csv' #Replace this string with your event file name. The file should be located in the same directory as the script.
    EVENTFILE = 'russiaordered.csv'
    df = pd.read_csv(EVENTFILE)
    df['code'] = df.code.astype(str)
    EventCodeLookup = pd.read_csv('EventCodeLookup.csv')
    pd.to_numeric(EventCodeLookup["Code"])

    #get EventCodes size
    numRows = df['code'].shape[0]

    #create eventcode list and markov chain dictionary
    eventCodeList = list()
    #markovDict = defaultdict()
    markovDict = {}
    numOfEvents = 20 #This is the number of predicted events.
    trainPercentage = 0.55
    trainrows = int(numRows * trainPercentage)
    train = list(df['code'])
    train = train[:trainrows]

    #CREATE EVENTCODELIST
    for i in range(numRows-1):
        tup = (df['code'].iloc[i], df['code'].iloc[i+1])
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

    print('total number of possible events in the given dataset (1st order): ', len(markovDict.keys()))
    print('total number of possible events in the given dataset (2nd order): ', len(markovDict_2nd.keys()))
    #results = [prediction == truth for prediction, truth in zip(y_predicted, y)]
    #accuracy = float(results.count(True)) / float(len(results))

    #This will generate list of events with 'numOfEvents' length from Markov Model.
    MarkovRandomEvents = generate_random_events(numOfEvents, markovDict)
    Events = list()
    for code in MarkovRandomEvents:
        temp = list(EventCodeLookup['Description'].loc[EventCodeLookup['Code'] == int(code)])[0]
        Events.append(temp)
    #print(Events) #This prints the forecasted events.
    
    #Distance Metric
    #init_event = generate_random_start(markovDict)
    init_event = '31'
    sub_event_dict = markovDict[init_event]
    distance_list = list()
    avg_distance_dict = {}
    event_generation_ct = 20

    for k in sub_event_dict.keys():
        for i in range(event_generation_ct):
            randomEvent = generate_random_events_with_fixed_start(numOfEvents, markovDict, init_event)
            distance = randomEvent.index(k) if k in randomEvent else None
            if distance is not None:
                distance_list.append(distance)
        avg_distance_dict[k] = sum(distance_list) / len(distance_list)
    print(avg_distance_dict)
    
    import plotly as py
    import plotly.graph_objs as go
    
    data = [go.Bar(
        x=['60%', '65%', '70%', '75%', '80%', '85%', '90%'],
        y=[9.041, 9.101, 8.844, 8.762, 8.337, 8.401, 8.395],
        opacity=0.6
    )]
    
    xaxis = go.XAxis(title="Train %")
    yaxis = go.YAxis(title='Distance')
    
    py.offline.plot({
    "data": data,
    "layout": go.Layout(title="Distance per Training Split", xaxis=xaxis, yaxis=yaxis)
    })
    
    
# =============================================================================
#     trace2 = go.Bar(
#         x=[0, 1, 2, 3, 4, 5],
#         y=[1, 0.5, 0.7, -1.2, 0.3, 0.4]
#     )
# =============================================================================

        
    #train 0.9, avg distance= 9.126
    #train 0.85, avg distance= 9.208
    #train 0.8, avg distance= 8.337
    #train 0.75, avg distance= 8.762
    #train 0.7, avg distance= 8.944
    #train 0.65, avg distance= 8.335
    #train 0.6, avg distance= 8.668
    #train 0.55, avg distance= 8.809
    #train 0.5, avg distance= 8.837
    
    ######################################### OUTPUT TO JSON FILE ##########################################
# =============================================================================
#     mc_data = list()
# 
#     for k,v in markovDict_prob.items():
#         for kt, kv in v.items():
#             tempdict = {}
#             tempdict['source'] = str(k)
#             tempdict['target'] = str(kt)
#             tempdict['value'] = str(kv)
#             mc_data.append(tempdict)
# 
#     JSONFILE = 'MC_2nd_json.json'
#     with open(JSONFILE, 'w') as f:
#        #json_data = json.dump(mc_data, f)
#        json.dump(mc_data, f)
# =============================================================================
