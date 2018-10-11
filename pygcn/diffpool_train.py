from __future__ import division
from __future__ import print_function

import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from pygcn.models import Classifier, GraphSAGEModel, Model

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj = np.load("../data/enzymes/graph.npy")
# features = np.load("../data/enzymes/one_hot.npy")
# labels = np.load("../data/enzymes/labels.npy")
ds = pickle.load(open("/data/jinjing/example/diffpool/enzymes.pkl", "rb"))
adj, features, labels = ds['adj'], ds['node_attr'], ds['graph_label']
labels = torch.from_numpy(labels).to(device)

idx_train, idx_test = train_test_split(np.arange(600), train_size=0.8, random_state=9)

model = Model().to(device)
model.train()
classifier = Classifier().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-7)
for e in tqdm(range(args.epochs)):
    pred_labels = []
    for i, idx in enumerate(idx_train):
        adj_train = torch.from_numpy(adj[idx]).to(device).float()
        features_train = torch.from_numpy(features[idx]).to(device).float()
        labels_train = labels[idx].view(-1).long()
        graph_feat = model(features_train, adj_train)
        output = classifier(graph_feat)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels_train)
        # print(labels_train)
        loss.backward()
        pred_labels.append(output.argmax())
        a = list(model.named_parameters())
        if torch.isnan(a[4][1].grad).sum() != 0:
            print(f"{i}:{torch.isnan(a[4][1].grad).sum()}")
        # else:
    optimizer.step()
    optimizer.zero_grad()
    pred_labels = torch.stack(pred_labels, dim=0)
    acc = (pred_labels.long() == labels[idx_train].long()).float().mean()
    tqdm.write(str(acc))
    print(acc)

# Model and optimizer
