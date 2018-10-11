import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGE(nn.Module):
    def __init__(self, infeat, outfeat, k=1):
        super().__init__()
        self.k = k
        self.W = nn.Linear(infeat, outfeat, bias=True)
        nn.init.xavier_uniform(self.W.weight)

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to("cuda")
        h_k_N = torch.spmm(adj, x)
        h_k = F.relu(self.W(h_k_N))
        h_k = h_k / (h_k.norm(dim=1, keepdim=True) + 1e-7)
        return h_k
