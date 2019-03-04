import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import scipy


class BatchedGraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, device='cpu', use_bn=True, mean=False, add_self=False):
        super().__init__()
        self.add_self = add_self
        self.use_bn = use_bn
        self.device = device
        self.mean = mean
        self.W = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform_(self.W.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x, adj, mask=None):
        if self.add_self:
            adj = adj + torch.eye(adj.size(0)).to(self.device)

        if self.mean:
            adj = adj / adj.sum(1, keepdim=True)

        h_k_N = torch.matmul(adj, x)
        h_k = self.W(h_k_N)
        h_k = F.normalize(h_k, dim=2, p=2)
        h_k = F.relu(h_k)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(h_k.size(1)).to(self.device)
            h_k = self.bn(h_k)
        if mask is not None:
            h_k = h_k * mask.unsqueeze(2).expand_as(h_k)
        return h_k


class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, is_final=False, device='cpu', link_pred=False):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.device = device
        self.is_final = is_final
        self.embed = BatchedGraphSAGE(nfeat, nhid, device=self.device, use_bn=True)
        self.assign_mat = BatchedGraphSAGE(nfeat, nnext, device=self.device, use_bn=True)
        self.link_pred_loss = 0

    def forward(self, x, adj):
        z_l = self.embed(x, adj)
        s_l = F.softmax(self.assign_mat(x, adj), dim=-1)
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
        if self.link_pred:
            self.link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=1)
        return xnext, anext


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(30, 50),
                                        nn.ReLU(),
                                        nn.Linear(50, 6))

    def forward(self, x):
        return self.classifier(x)


class BatchedModel(nn.Module):
    def __init__(self, pool_size, device, link_pred=False):
        super().__init__()
        self.link_pred = link_pred
        self.device = device
        self.layers = nn.ModuleList([
            BatchedGraphSAGE(18, 30, device=self.device),
            BatchedGraphSAGE(30, 30, device=self.device),
            BatchedDiffPool(30, pool_size, 30, device=self.device, link_pred=link_pred),
            BatchedGraphSAGE(30, 30, device=self.device),
            BatchedGraphSAGE(30, 30, device=self.device),
            # BatchedDiffPool(30, 1, 30, is_final=True, device=self.device)
        ])
        self.classifier = Classifier()

    def forward(self, x, adj, mask):
        for layer in self.layers:
            if isinstance(layer, BatchedGraphSAGE):
                if mask.shape[1] == x.shape[1]:
                    x = layer(x, adj, mask)
                else:
                    x = layer(x, adj)
            elif isinstance(layer, BatchedDiffPool):
                x, adj = layer(x, adj)

        # x = x * mask
        readout_x = x.sum(dim=1)
        return readout_x

    def loss(self, output, labels):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)
        if self.link_pred:
            for layer in self.layers:
                if isinstance(layer, BatchedDiffPool):
                    loss += layer.link_pred_loss

        return loss
