import pickle

import networkx as nx
import numpy as np
import torch


def load():
    graphs = pickle.load(open("./enzymes_s.pkl", 'rb'))
    features = []
    adj_list = []
    labels = []
    for graph in graphs:
        feats = nx.get_node_attributes(graph, 'feat').values()
        features.append(np.stack(feats))
        adj_list.append(nx.to_numpy_matrix(graph))
        labels.append(graph.graph['label'])
    return adj_list, features, np.array(labels)


def load_graphs():
    graphs = pickle.load(open("/data/jinjing/example/graph-pooling/enzymes_s.pkl", 'rb'))
    return graphs


