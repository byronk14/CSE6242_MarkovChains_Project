import pandas as pd
import numpy as np
from histograms import Dictogram
import random
from collections import deque
import re
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import networkx as nx
from pprint import pprint
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


############################################ START HERE ###############################################
#IMPORT CSV
EVENTFILE = '20180320161401.22251.events.csv' #Replace this string with your event file name. The file should be located in the same directory as the script.
df = pd.read_csv(EVENTFILE)
EventCodeLookup = pd.read_csv('EventCodeLookup.csv')
pd.to_numeric(EventCodeLookup["Code"])

#get EventCodes size
numRows = df['EventCode'].shape[0]

#create eventcode list and markov chain dictionary
eventCodeList = list()
markovDict = {}
numOfEvents = 20 #This is the number of predicted events.
trainPercentage = 0.8
trainrows = int(numRows * trainPercentage)
train = list(df['EventCode'])
train = train[:trainrows]

#CREATE EVENTCODELIST
for i in range(numRows-1):
    tup = (df['EventCode'].iloc[i], df['EventCode'].iloc[i+1])
    eventCodeList.append(tup)


#CREATE MARKOV CHAIN MODEL
markovDict = make_markov_model(train)
markovDict_prob = {}
t_df = pd.DataFrame(np.zeros((298, 298)), columns=EventCodeLookup['Code'], index=EventCodeLookup['Code'])
states = list(EventCodeLookup['Code'])


#CREATE TRANSITION MATRIX
for k, v in markovDict.items():
    temptot = sum(v.values())
    tempprobdict = {}
    for kt, vt in v.items():
        tempprobdict[kt] = vt / temptot
        t_df.loc[k, kt] = vt / temptot
    markovDict_prob[k] = tempprobdict
#print(markovDict_prob) #Uncomment to see what the MC Model looks like



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
print(mc_data)
JSONFILE = 'MC_json.json'
with open(JSONFILE, 'w') as f:
    json_data = json.dump(mc_data, f)



############################################### IGNORE #################################################
# create a function that maps transition probability dataframe 
# to markov edges and weights

# def _get_markov_edges(Q):
#     edges = {}
#     for col in Q.columns:
#         for idx in Q.index:
#             edges[(idx,col)] = Q.loc[idx,col]
#     return edges

# edges_wts = _get_markov_edges(t_df)
#pprint(edges_wts)

# create graph object
# G = nx.MultiDiGraph()
# # nodes correspond to states
# G.add_nodes_from(states)
# #print(f'Nodes:\n{G.nodes()}\n')
# # edges represent transition probabilities
# for k, v in edges_wts.items():
#     tmp_origin, tmp_destination = k[0], k[1]
#     G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
# #print(f'Edges:')
# pprint(G.edges(data=True))    
# #pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
# nx.draw_networkx(G, pos)
# # create edge labels for jupyter plot but is not necessary
# edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
# nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
# nx.drawing.nx_pydot.write_dot(G, 'US_Iraq_03_04_markov.dot')