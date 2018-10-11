import torch
import torch.nn as nn
import torch.nn.functional as F

from graphsage import GraphSAGE
from layers import GraphConvolution


class DiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, is_final=False):
        super(DiffPool, self).__init__()
        self.is_final = is_final
        self.embed = GraphConvolution(nfeat, nhid)
        self.pool = GraphConvolution(nfeat, nnext)

    def forward(self, x, adj):
        z_l = self.embed(x, adj)
        if self.is_final:
            s_l = torch.ones(adj.size(0), 1).to("cuda")
        else:
            s_l = F.softmax(self.pool(x, adj), dim=1)
        xnext = torch.mm(s_l.t(), z_l)
        anext = s_l.t().mm(adj).mm(s_l)
        return xnext, anext


class GraphSAGEModel(nn.Module):
    def __init__(self):
        super(GraphSAGEModel, self).__init__()
        self.GNNlayers = nn.ModuleList([
            GraphSAGE(18, 16),
            GraphSAGE(16, 16),
            GraphSAGE(16, 8),
        ])
        self.BNlayers = nn.ModuleList([
            nn.BatchNorm1d(16),
            nn.BatchNorm1d(16),
            nn.BatchNorm1d(8),
        ])

    def forward(self, x, adj):
        for i in range(3):
            x = self.GNNlayers[i](x, adj)
            x = self.BNlayers[i](x)
        return x.mean(dim=0, keepdim=True)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dps = nn.ModuleList([
            GraphSAGE(18, 16),
            GraphSAGE(16, 16),
            DiffPool(16, 20, 8),
            GraphSAGE(8, 8),
            DiffPool(8, 1, 8, is_final=True)
        ])
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.batchnorm2 = nn.BatchNorm1d(16)
        self.batchnorm3 = nn.BatchNorm1d(8)

    def forward(self, x, adj):
        x = self.dps[0](x, adj)
        x = self.batchnorm1(x)
        x = self.dps[1](x, adj)
        x = self.batchnorm2(x)
        x, adj = self.dps[2](x, adj)
        x = self.dps[3](x, adj)
        x = self.batchnorm3(x)
        x, adj = self.dps[4](x, adj)
        return x


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(8, 6),
                                        nn.ReLU(),
                                        nn.Linear(6, 6))

    def forward(self, x):
        return self.classifier(x)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
