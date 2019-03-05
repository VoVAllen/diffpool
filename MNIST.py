from __future__ import absolute_import
import numpy as np
import dgl
import os

import torch
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url

from torch.utils.data import Dataset

import warnings
import torch.utils.data as data
from PIL import Image
import os
import os.path
import gzip
import numpy as np
import torch
import codecs
from torchvision import transforms

import networkx as nx


class GraphTransform:
    def __init__(self, device):
        self.adj = nx.to_numpy_matrix(nx.grid_2d_graph(28, 28))

    def __call__(self, img):
        return self.adj, \
               np.array(img).reshape(-1, 1)
