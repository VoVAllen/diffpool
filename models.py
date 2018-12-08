import torch
import torch.nn as nn
import torch.nn.functional as F

from graphsage import GraphSAGE


class DiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, is_final=False, device='cpu'):
        super(DiffPool, self).__init__()
        self.device = device
        self.is_final = is_final
        self.embed = GraphSAGE(nfeat, nhid, device=self.device)
        self.assign_mat = GraphSAGE(nfeat, nnext, device=self.device)
        self.link_pred_loss = 0

    def forward(self, x, adj):
        z_l = self.embed(x, adj)
        if self.is_final:
            s_l = torch.ones(adj.size(0), 1).to(self.device)
        else:
            s_l = F.softmax(self.assign_mat(x, adj), dim=1)
        xnext = torch.mm(s_l.t(), z_l)
        anext = s_l.t().mm(adj).mm(s_l)
        if not self.is_final:
            self.link_pred_loss = (adj - s_l.mm(s_l.t())).norm()
        return xnext, anext


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(64, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 6))

    def forward(self, x):
        return self.classifier(x)


class Model(nn.Module):
    def __init__(self, pool_size, device):
        super().__init__()
        self.device = device
        self.dps = nn.ModuleList([
            GraphSAGE(18, 128, device=self.device),
            GraphSAGE(128, 128, device=self.device),
            DiffPool(128, pool_size, 128, device=self.device),
            GraphSAGE(128, 64, device=self.device),
            DiffPool(64, 1, 64, is_final=True, device=self.device)
        ])
        self.classifier = Classifier()

    def forward(self, x, adj):
        x = self.dps[0](x, adj)
        x = self.dps[1](x, adj)
        x, adj = self.dps[2](x, adj)
        x = self.dps[3](x, adj)
        x, adj = self.dps[4](x, adj)
        return x

    def link_pred_loss(self):
        return self.dps[2].link_pred_loss
