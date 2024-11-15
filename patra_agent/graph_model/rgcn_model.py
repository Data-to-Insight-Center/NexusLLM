import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List, Dict, Tuple
from neo4j import GraphDatabase
from tqdm import tqdm

class RGCN(nn.Module):
    def __init__(self, num_nodes: int, num_node_types: int, num_relations: int, 
                 hidden_channels: int = 16, num_bases: int = None, dropout: float = 0.1):
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
    
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, edge_index, edge_type, node_type):

        x = self.node_emb(x) + self.node_type_emb(node_type)
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


class RGCNLinkPrediction:
    def __init__(self, uri:str, username: str, password: str, batch_size: int = 1000):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.model = None
        self.node_mapping = {}
        self.node_type_mapping = {}
        self.relation_mapping = {}
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def close(self):
        self.driver.close()

    def extract_graph_data(self, node_labels: List[str], relationship_types: List[str]) -> Data:
        """Extract graph data efficiently using batching."""
        with self.driver.session() as session:
            nodes, node_types = self._extract_nodes_batch(session, node_labels)
            edge_index, edge_types = self._extract_edges_batch(session, relationship_types)
            
            return Data(
                x=torch.tensor(nodes, dtype=torch.long),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_type=torch.tensor(edge_types, dtype=torch.long),
                node_type=torch.tensor(node_types, dtype=torch.long),
                num_nodes=len(self.node_mapping)
            )
    
    def _extract_nodes_batch(self, session, node_labels: List[str]) -> Tuple[List[int], List[int]]:
        """Extract nodes in batches for better memory efficiency."""
        nodes = []
        node_types = []
        
        for label in node_labels:
            offset = 0
            while True:
                query = f"""
                MATCH (n:{label})
                RETURN elementId(n) as id
                SKIP {offset} LIMIT {self.batch_size}
                """
                result = list(session.run(query))
                if not result:
                    break
                    
                for record in result:
                    node_id = record["id"]
                    if node_id not in self.node_mapping:
                        self.node_mapping[node_id] = len(self.node_mapping)
                    if label not in self.node_type_mapping:
                        self.node_type_mapping[label] = len(self.node_type_mapping)
                        
                    nodes.append(self.node_mapping[node_id])
                    node_types.append(self.node_type_mapping[label])
                    
                offset += self.batch_size
                
        return nodes, node_types
        
    def _extract_edges_batch(self, session, relationship_types: List[str]) -> Tuple[List[List[int]], List[int]]:
        """Extract edges in batches for better memory efficiency."""
        edge_index = [[], []]
        edge_types = []
        
        for rel_type in relationship_types:
            if rel_type not in self.relation_mapping:
                self.relation_mapping[rel_type] = len(self.relation_mapping)
                
            offset = 0
            while True:
                query = f"""
                MATCH (s)-[r:{rel_type}]->(t)
                RETURN elementId(s) as source, elementId(t) as target
                SKIP {offset} LIMIT {self.batch_size}
                """
                result = list(session.run(query))
                if not result:
                    break
                    
                for record in result:
                    source = self.node_mapping[record["source"]]
                    target = self.node_mapping[record["target"]]
                    edge_index[0].append(source)
                    edge_index[1].append(target)
                    edge_types.append(self.relation_mapping[rel_type])
                    
                offset += self.batch_size
                
        return edge_index, edge_types

    
    def train_model(self, data: Data, num_epochs: int = 200, 
                   hidden_channels: int = 16, negative_sampling_ratio: float = 1.0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
    
        self.model = RGCN(
            num_nodes=data.num_nodes,
            num_node_types=len(self.node_type_mapping),
            num_relations=len(self.relation_mapping),
            hidden_channels=hidden_channels,
            num_bases=4  
        ).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

        best_auc = 0
        patience = 20
        patience_counter = 0
    
        for epoch in tqdm(range(num_epochs), desc="Training"):
            self.model.train()
            optimizer.zero_grad()
            
            z = self.model(data.x, data.edge_index, data.edge_type, data.node_type)
            
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=int(data.edge_index.size(1) * negative_sampling_ratio)
            )
            pos_edge_index = data.edge_index
            pos_pred = self.model.decode(z, pos_edge_index)
            neg_pred = self.model.decode(z, neg_edge_index)
            
            loss = self.compute_loss(pos_pred, neg_pred)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                auc, ap = self.evaluate_model(data)
                print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, Avg Precision: {ap:.4f}')
                scheduler.step(auc)
                
                if auc > best_auc:
                    best_auc = auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

    def compute_loss(self,pos_pred: torch.Tensor, neg_pred: torch.Tensor) -> torch.Tensor:
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        return pos_loss + neg_loss
    
    @torch.no_grad()
    def evaluate_model(self, data: Data) -> Tuple[float, float]:
        self.model.eval()
        z = self.model(data.x, data.edge_index, data.edge_type, data.node_type)
        batch_size = 10000
        all_preds = []
        all_true = []
        
        for i in range(0, data.edge_index.size(1), batch_size):
            batch_edge_index = data.edge_index[:, i:i+batch_size]
            pos_pred = self.model.decode(z, batch_edge_index)
            all_preds.append(pos_pred)
            all_true.append(torch.ones_like(pos_pred))
            
            neg_edge_index = negative_sampling(
                edge_index=batch_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=batch_edge_index.size(1)
            )
            neg_pred = self.model.decode(z, neg_edge_index)
            all_preds.append(neg_pred)
            all_true.append(torch.zeros_like(neg_pred))
        
        pred = torch.cat(all_preds).cpu()
        true = torch.cat(all_true).cpu()
        
        return roc_auc_score(true, pred), average_precision_score(true, pred)

    def save_model(self, path: str):
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'node_mapping': self.node_mapping,
                'node_type_mapping': self.node_type_mapping,
                'relation_mapping': self.relation_mapping
            }, path)

    def close(self):
        if self.driver:
            self.driver.close()
            self.model = None
            self._clear_mappings()

    def _clear_mappings(self):
        self.node_mapping.clear()
        self.node_type_mapping.clear()
        self.relation_mapping.clear()
