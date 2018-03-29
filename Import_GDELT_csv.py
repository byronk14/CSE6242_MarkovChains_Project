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


#IMPORT CSV
df = pd.read_csv('20180320161401.22251.events.csv')
EventCodeLookup = pd.read_csv('EventCodeLookup.csv')
pd.to_numeric(EventCodeLookup["Code"])

#get EventCodes size
numRows = df['EventCode'].shape[0]

#create eventcode list and markov chain dictionary
eventCodeList = list()
markovDict = {}
numOfEvents = 20
trainPercentage = 0.8
trainrows = int(numRows * trainPercentage)
# goldsteinNums = list(df['GoldsteinScale'])
# lat = list(df['ActionGeo_Lat'])
# lon = list(df['ActionGeo_Long'])

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



train = list(df['EventCode'])
train = train[:trainrows]
#train.append("END")

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
#print(markovDict_prob)


G = nx.MultiDiGraph()

# nodes correspond to states
G.add_nodes_from(states)
print(f'Nodes:\n{G.nodes()}\n')

labels={}
edge_labels={}
for i, origin_state in enumerate(states):
    for j, destination_state in enumerate(states):
        rate = t_df.loc[origin_state,destination_state]
        if rate > 0:
            G.add_edge(origin_state,
                       destination_state,
                       weight=rate,
                       label="{:.02f}".format(rate))
            edge_labels[(origin_state, destination_state)] = label="{:.02f}".format(rate)

#print(f'Edges:')
#pprint(G.edges(data=True)) 

plt.figure(figsize=(14,7))
node_size = 200
pos = {state: state for state in states}
nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
nx.draw_networkx_labels(G, pos, font_weight=2)
nx.draw_networkx_edge_labels(G, pos, edge_labels)
plt.axis('off')
nx.draw(G)
plt.show()

#Insert probabilities into transition matrix
# for k, v in markovDict_prob.items():
#     for kt, vt in v.items():
#         t_df.loc[k, kt] = vt


#from pprint import pprint 

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





# MarkovRandomEvents = generate_random_sentence(numOfEvents, make_markov_model(train))
# Events = list()
# for code in MarkovRandomEvents:
#     temp = list(EventCodeLookup['Description'].loc[EventCodeLookup['Code'] == code])[0]
#     Events.append(temp)
# print(Events)







##VISUALIZATIONS
# GoldsteinCusum = list()
# cusum = 0
# for num in goldsteinNums:
#     cusum += num
#     GoldsteinCusum.append(cusum)

# plt.plot(GoldsteinCusum)
# plt.title("Cumulative Goldstein Scale between US and Iraq 2003-2004")
# plt.show()

# Define the projection, scale, the corners of the map, and the resolution.
# m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
#             llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
# # Draw the coastlines
# m.drawcoastlines()
# # Color the continents
# m.fillcontinents(color='coral',lake_color='aqua')
# # draw parallels and meridians.
# m.drawparallels(np.arange(-90.,91.,30.))
# m.drawmeridians(np.arange(-180.,181.,60.))
# # fill in the oceans
# m.drawmapboundary(fill_color='aqua')
# plt.title("Mercator Projection")
# x,y = m(lon, lat)                          
# m.plot(x,y, 'bo', markersize=5)
# plt.show()