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
from dotenv import load_dotenv

class RGCN(torch.nn.Module):
    def __init__(self, num_nodes: int, num_node_types: int, num_relations: int, 
                 hidden_channels: int = 16, num_bases: int = None):
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
        

        self.node_emb = nn.Embedding(num_nodes, hidden_channels)

        self.node_type_emb = nn.Embedding(num_node_types, hidden_channels)
    
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
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
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.model = None
        self.node_mapping = {}
        self.node_type_mapping = {}
        self.relation_mapping = {}
        
    def close(self):
        self.driver.close()
        
    def extract_graph_data(self, node_labels: List[str], 
                          relationship_types: List[str]) -> Tuple[Data, Dict]: ## convert graph data to PyTorch Geometric format
        with self.driver.session() as session:

            nodes = []
            node_types = []
            for label in node_labels:
                query = f"MATCH (n:{label}) RETURN elementId(n) as id"
                result = session.run(query)
                for record in result:
                    node_id = record["id"]
                    if node_id not in self.node_mapping:
                        self.node_mapping[node_id] = len(self.node_mapping)
                    if label not in self.node_type_mapping:
                        self.node_type_mapping[label] = len(self.node_type_mapping)
                    nodes.append(self.node_mapping[node_id])
                    node_types.append(self.node_type_mapping[label])
            
            edge_index = [[], []]
            edge_types = []
            for rel_type in relationship_types:
                if rel_type not in self.relation_mapping:
                    self.relation_mapping[rel_type] = len(self.relation_mapping)
                
                query = f"""
                MATCH (s)-[r:{rel_type}]->(t)
                RETURN elementId(s) as source, elementId(t) as target
                """
                result = session.run(query)
                for record in result:
                    source = self.node_mapping[record["source"]]
                    target = self.node_mapping[record["target"]]
                    edge_index[0].append(source)
                    edge_index[1].append(target)
                    edge_types.append(self.relation_mapping[rel_type])

            edge_index = torch.tensor(edge_index, dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            x = torch.tensor(nodes, dtype=torch.long)
            node_type = torch.tensor(node_types, dtype=torch.long)
            

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_type=edge_type,
                node_type=node_type,
                num_nodes=len(self.node_mapping)
            )
            
            return data
        
    def train_model(self, data: Data, num_epochs: int = 200, 
                   hidden_channels: int = 16, negative_sampling_ratio: float = 1.0):
        """Train the RGCN model for link prediction"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        

        self.model = RGCN(
            num_nodes=data.num_nodes,
            num_node_types=len(self.node_type_mapping),
            num_relations=len(self.relation_mapping),
            hidden_channels=hidden_channels,
            num_bases=4  # based on number of relations
        ).to(device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        

        self.model.train()
        for epoch in range(num_epochs):
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
                print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')
    
    @staticmethod
    def compute_loss(pos_pred: torch.Tensor, neg_pred: torch.Tensor) -> torch.Tensor:
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        return pos_loss + neg_loss
    
    @torch.no_grad()
    def evaluate_model(self, data: Data) -> Tuple[float, float]:
        self.model.eval()
        
        z = self.model(data.x, data.edge_index, data.edge_type, data.node_type)
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1)
        )
        
        pos_pred = self.model.decode(z, pos_edge_index).cpu()
        neg_pred = self.model.decode(z, neg_edge_index).cpu()
        
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)
        
        auc = roc_auc_score(true.numpy(), pred.numpy())
        ap = average_precision_score(true.numpy(), pred.numpy())
        
        return auc, ap
    
    @torch.no_grad()
    def predict_link(self, source_id: str, target_id: str) -> Dict:
        if not self.model or source_id not in self.node_mapping or target_id not in self.node_mapping:
            return {'probability': 0.0, 'prediction': False}
        
        self.model.eval()
        device = next(self.model.parameters()).device

        edge_index = torch.tensor([[self.node_mapping[source_id]], 
                                 [self.node_mapping[target_id]]], 
                                device=device)
        

        z = self.model(self.last_data.x.to(device), 
                      self.last_data.edge_index.to(device),
                      self.last_data.edge_type.to(device),
                      self.last_data.node_type.to(device))
        

        prob = self.model.decode(z, edge_index).cpu().item()
        
        return {
            'probability': prob,
            'prediction': prob > 0.5
        }

def main():

    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USER")
    NEO4J_PWD = os.getenv("NEO4J_PWD")

    predictor = RGCNLinkPrediction(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PWD
    )
    
    try:

        node_labels = ["Model", "EdgeDevice", "Deployment"]
        relationship_types = ["On_Device", "HAS_DEPLOYMENT"]
 
        print("Extracting graph data...")
        data = predictor.extract_graph_data(node_labels, relationship_types)
        predictor.last_data = data
 
        print("Training RGCN model...")
        predictor.train_model(data, num_epochs=200)

        print("\nMaking sample predictions...")
        source_id = "4:b6ae30eb-5fdd-4c39-b281-fa2550f0ea84:24"
        target_id = "4:b6ae30eb-5fdd-4c39-b281-fa2550f0ea84:0"
        
        result = predictor.predict_link(source_id, target_id)
        print(f"Link prediction result: {result}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
    finally:
        predictor.close()

if __name__ == "__main__":
    main()