import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, device='cpu', use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        self.device = device
        self.W = nn.Linear(infeat, outfeat, bias=True)
        if use_bn:
            self.bn = nn.BatchNorm1d(outfeat)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(self.device)
        # degree_matrix = torch.diag(1 / adj.sum(0)).to(self.device)

        h_k_N = torch.mm(adj, x) / adj.sum(1, keepdim=True)
        h_k = F.relu(self.W(h_k_N))
        h_k = h_k / (h_k.norm(dim=1, keepdim=True) + 1e-7)
        if self.use_bn:
            h_k = self.bn(h_k)
        return h_k
