import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, confusion_matrix
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np

class DistMult(nn.Module):
    def __init__(self, num_nodes: int, num_relations: int, embedding_dim: int = 200):
        super(DistMult, self).__init__()
        self.entity_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.relation_embedding = nn.Embedding(num_relations, embedding_dim)

        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)

    def forward(self, edge_index, edge_type):
        src, dst = edge_index[0], edge_index[1]
        src_embeds = self.entity_embedding(src)
        dst_embeds = self.entity_embedding(dst)
        rel_embeds = self.relation_embedding(edge_type)

        scores = torch.sum(src_embeds * rel_embeds * dst_embeds, dim=1)
        return torch.sigmoid(scores) 
    
    def score_edges(self, edge_index, edge_type):

        return self.forward(edge_index, edge_type)
