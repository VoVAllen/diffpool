from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from graphsage import GraphSAGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adj = np.load("../data/enzymes/graph.npy")
features = np.load("../data/enzymes/one_hot.npy")
labels = np.load("../data/enzymes/labels.npy")
labels = torch.from_numpy(labels).to(device)

a = torch.from_numpy(adj[0]).float()
f = torch.from_numpy(features[0]).float()

model = GraphSAGE(3, 8)
model(f, a)
