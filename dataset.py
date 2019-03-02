from __future__ import absolute_import
import numpy as np
import dgl
import os

import torch
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url


class TUDataset(object):
    _url = r"https://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets/{}.zip"

    def __init__(self, name, use_node_attr=True, use_node_label=False):

        self.name = name
        self.extract_dir = self._download()
        DS_edge_list = self._idx_from_zero(
            np.loadtxt(self._file_path("A"), delimiter=",", dtype=int))
        DS_indicator = self._idx_from_zero(
            np.loadtxt(self._file_path("graph_indicator"), dtype=int))
        DS_graph_labels = self._idx_from_zero(
            np.loadtxt(self._file_path("graph_labels"), dtype=int))
        DS_node_labels = self.to_onehot(
            self._idx_from_zero(np.loadtxt(self._file_path("node_labels"), dtype=int)))

        g = dgl.DGLGraph()
        g.add_nodes(DS_edge_list.max() + 1)
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])
        g.add_edges(DS_edge_list[:, 1], DS_edge_list[:, 0])

        node_idx_list = []
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
        self.graph_lists = g.subgraphs(node_idx_list)
        self.graph_labels = DS_graph_labels

        if use_node_label:
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata['node_label'] = DS_node_labels[idxs, :]

        if use_node_attr:
            DS_node_attr = np.loadtxt(self._file_path("node_attributes"), delimiter=",")
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata['feat'] = DS_node_attr[idxs, :]

    def __getitem__(self, idx):
        g = self.graph_lists[idx]
        return g.adjacency_matrix().to_dense(), g.ndata['feat'], self.graph_labels[idx]

    def __len__(self):
        return len(self.graph_lists)

    def _download(self):
        download_dir = get_download_dir()
        zip_file_path = os.path.join(download_dir, "tu_{}.zip".format(self.name))
        download(self._url.format(self.name), path=zip_file_path)
        extract_dir = os.path.join(download_dir, "tu_{}".format(self.name))
        extract_archive(zip_file_path, extract_dir)
        return extract_dir

    def _file_path(self, category):
        return os.path.join(self.extract_dir, self.name, "{}_{}.txt".format(self.name, category))

    @staticmethod
    def _idx_from_zero(idx_tensor):
        return idx_tensor - np.min(idx_tensor)

    @staticmethod
    def to_onehot(label_tensor):
        label_num = label_tensor.shape[0]
        assert np.min(label_tensor) == 0
        one_hot_tensor = np.zeros((label_num, np.max(label_tensor) + 1))
        one_hot_tensor[np.arange(label_num), label_tensor] = 1
        return one_hot_tensor


class CollateFn:
    def __init__(self, device='cpu'):
        self.device = device

    def __call__(self, batch):
        adj_tensor_list = []
        features_list = []
        mask_list = []
        adj, features, labels = zip(*batch)
        max_num_nodes = max([g.shape[0] for g in adj])
        for A, F, L in zip(adj, features, labels):
            length = A.shape[0]
            pad_len = max_num_nodes - length
            adj_tensor_list.append(np.pad(A, ((0, pad_len), (0, pad_len)), mode='constant'))
            features_list.append(np.pad(F, ((0, pad_len), (0, 0)), mode='constant'))
            mask = np.zeros(max_num_nodes)
            mask[:length] = 1
            mask_list.append(mask)
        return torch.from_numpy(np.stack(adj_tensor_list, 0)).float().to(self.device), \
               torch.from_numpy(np.stack(features_list, 0)).float().to(self.device), \
               torch.from_numpy(np.stack(mask_list, 0)).float().to(self.device), \
               torch.from_numpy(np.stack(labels, 0)).long().to(self.device)
