from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import load_data
from models import Model

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--link-pred', action='store_true', default=False,
                    help='Enable Link Prediction Loss')

args = parser.parse_args()
device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if device == 'cuda':
#     torch.cuda.manual_seed(args.seed)

adj, features, labels = load_data.load()
max_num_nodes = max([g.shape[0] for g in adj])
labels = torch.from_numpy(labels).to(device)

idx = np.arange(600)
np.random.RandomState(seed=124).shuffle(idx)
idx_train, idx_test = idx[:480], idx[480:]

model = Model(pool_size=int(max_num_nodes * 0.25), device=device).to(device)
model.train()
# optimizer = optim.SGD(model.parameters(), lr=1e-5)
optimizer = optim.Adam(model.parameters())
for e in tqdm(range(args.epochs)):
    pred_labels = []
    for i, idx in enumerate(idx_train):
        adj_train = torch.from_numpy(adj[idx]).to(device).float()
        features_train = torch.from_numpy(features[idx]).to(device).float()
        labels_train = labels[idx].view(-1).long()
        graph_feat = model(features_train, adj_train)
        output = model.classifier(graph_feat)
        criterion = nn.CrossEntropyLoss()
        if args.link_pred:
            loss = criterion(output, labels_train) + model.link_pred_loss()
        else:
            loss = criterion(output, labels_train)
        loss.backward()
        pred_labels.append(output.argmax())
        if i % 32 == 0:
            optimizer.step()
            optimizer.zero_grad()

    optimizer.step()
    optimizer.zero_grad()
    pred_labels = torch.stack(pred_labels, dim=0)
    acc = (pred_labels.long() == labels[idx_train].long()).float().mean()
    tqdm.write(f"Epoch:{e}  \t train_acc:{acc:.2f}")

    val_list = []
    for i, idx in enumerate(idx_test):
        with torch.no_grad():
            adj_test = torch.from_numpy(adj[idx]).to(device).float()
            features_test = torch.from_numpy(features[idx]).to(device).float()
            labels_test = labels[idx].view(-1).long()
            graph_feat = model(features_test, adj_test)
            output = model.classifier(graph_feat)
            val_list.append(output.argmax())
    val_acc = (torch.stack(val_list) == labels[idx_test]).float().mean()
    tqdm.write(f"Epoch:{e}  \t val_acc:{val_acc:.2f}")
