import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, confusion_matrix
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

class VanillaRGCN(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, hidden_channels: int = 16, num_bases: int = None, dropout: float = 0.1):
        super(VanillaRGCN, self).__init__()

        self.conv1 = RGCNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_relations=num_relations,
            num_bases=num_bases
        )
        
        self.conv2 = RGCNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_relations=num_relations,
            num_bases=num_bases
        )

        self.dropout = nn.Dropout(dropout)
        self.node_emb = nn.Embedding(num_nodes, hidden_channels)

        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_type):
        x = self.node_emb(x) 
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        return x

    def decode(self, z, edge_index):
        src, dst = edge_index
        src_embeds = z[src]
        dst_embeds = z[dst]

        x = torch.cat([src_embeds, dst_embeds], dim=1)
        return self.link_predictor(x).squeeze()

    def score_edges(self, z, edge_index):
        return self.decode(z, edge_index)
