import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

class PRGCN(nn.Module):
    def __init__(self, num_nodes: int, num_node_types: int, num_relations: int, 
                 hidden_channels: int = 16, num_bases: int = None, dropout: float = 0.1, max_properties: int = None):
        super().__init__()
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
        self.node_type_emb = nn.Embedding(num_node_types, hidden_channels)
        self.property_encoder = nn.Sequential(
            nn.Linear(max_properties, hidden_channels),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
    
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels , 1),
            nn.Sigmoid()
        )
    def forward(self, x, edge_index, edge_type, node_type, node_properties): 
        x = self.node_emb(x) + self.node_type_emb(node_type) + self.property_encoder(node_properties)
        x1 = F.relu(self.conv1(x, edge_index, edge_type))
        x = x + x1
        x2 = F.relu(self.conv2(x, edge_index, edge_type))
        x = x + x2    
        return x
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        src_embeds = z[src]
        dst_embeds = z[dst]   
        x = torch.cat([src_embeds, dst_embeds], dim=1)
        return self.link_predictor(x).squeeze()
 