import os
import torch
from typing import Dict, List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from rgcn_model import RGCN, RGCNLinkPrediction
import heapq
from tqdm import tqdm
import warnings
from torch_geometric.data import Data

load_dotenv()  
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PWD")

class RGCNPredictor:
    def __init__(self, uri: str, username: str, password: str, model_path: str, batch_size: int = 1000):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.model_path = model_path
        self.model = None
        self.node_mapping = {}
        self.node_type_mapping = {}
        self.relation_mapping = {}
        self.last_data = None
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
        self.reverse_node_mapping = {v: k for k, v in self.node_mapping.items()}
        self.reverse_node_type_mapping = {v: k for k, v in self.node_type_mapping.items()}
        
    def _load_model(self):

        print("Loading saved model...")
        checkpoint = torch.load(self.model_path, weights_only=True)
        
        self.node_mapping = checkpoint['node_mapping']
        self.node_type_mapping = checkpoint['node_type_mapping']
        self.relation_mapping = checkpoint['relation_mapping']
        
        self.model = RGCN(
            num_nodes=len(self.node_mapping),
            num_node_types=len(self.node_type_mapping),
            num_relations=len(self.relation_mapping),
            hidden_channels=16,
            num_bases=4
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
    def get_node_type(self, node_id: str) -> str:
        if node_id not in self.node_mapping:
            return None
        node_idx = self.node_mapping[node_id]
        node_type_idx = self.last_data.node_type[node_idx].item()
        return self.reverse_node_type_mapping[node_type_idx]

    def find_potential_connections(self, node_id: str, top_k: int = 5, 
                                 threshold: float = 0.5,
                                 target_type: str = None) -> List[Dict]:

        if node_id not in self.node_mapping:
            return []

        z = self.get_embeddings()
        source_idx = self.node_mapping[node_id]
        
        potential_connections = []
        batch_size = 1000 
        
 
        if target_type is not None and target_type in self.node_type_mapping:
            target_type_idx = self.node_type_mapping[target_type]
            valid_targets = (self.last_data.node_type == target_type_idx).nonzero().squeeze()
        else:
            valid_targets = torch.arange(len(self.node_mapping))
            

        for i in range(0, len(valid_targets), batch_size):
            batch_targets = valid_targets[i:i + batch_size]

            edge_index = torch.vstack([
                torch.full((1, len(batch_targets)), source_idx, device=self.device),
                batch_targets.to(self.device)
            ])

            with torch.no_grad():
                probs = self.model.decode(z, edge_index).cpu().detach()
                
            for j, target_idx in enumerate(batch_targets):
                if target_idx == source_idx:  # Skip self-connections
                    continue
                    
                prob = probs[j].item()
                if prob >= threshold:
                    target_id = self.reverse_node_mapping[target_idx.item()]
                    target_type = self.get_node_type(target_id)
                    
                    heapq.heappush(potential_connections, 
                                 (-prob, { 
                                     'target_id': target_id,
                                     'target_type': target_type,
                                     'probability': prob
                                 }))
                    
                    if len(potential_connections) > top_k:
                        heapq.heappop(potential_connections)
                        
        return [item[1] for item in sorted(potential_connections, key=lambda x: -x[0])]

    def find_missing_edges(self, threshold: float = 0.8) -> List[Dict]:
        missing_edges = []

        existing_edges = set(map(tuple, self.last_data.edge_index.t().tolist()))
        

        for node_id in tqdm(self.node_mapping.keys(), desc="Finding missing edges"):
            potential_connections = self.find_potential_connections(
                node_id, 
                top_k=5, 
                threshold=threshold
            )
            
            for connection in potential_connections:
                source_idx = self.node_mapping[node_id]
                target_idx = self.node_mapping[connection['target_id']]

                if ((source_idx, target_idx) not in existing_edges and 
                    (target_idx, source_idx) not in existing_edges):
                    missing_edges.append({
                        'source_id': node_id,
                        'source_type': self.get_node_type(node_id),
                        'target_id': connection['target_id'],
                        'target_type': connection['target_type'],
                        'probability': connection['probability']
                    })
                    
        return sorted(missing_edges, key=lambda x: x['probability'], reverse=True)

    def extract_graph_data(self, node_labels, relationship_types):
        predictor = RGCNLinkPrediction(
                    NEO4J_URI, 
                    NEO4J_USERNAME,
                    NEO4J_PWD, 
        )
        predictor.driver = self.driver
        predictor.node_mapping = self.node_mapping
        predictor.node_type_mapping = self.node_type_mapping
        predictor.relation_mapping = self.relation_mapping
        
        self.last_data = predictor.extract_graph_data(node_labels, relationship_types)
        return self.last_data
    
    @torch.no_grad()
    def get_embeddings(self) -> torch.Tensor:
        if self.last_data is None:
            raise ValueError("No graph data available. Call extract_graph_data first.")
            
        self.model.eval()
        return self.model(
            self.last_data.x.to(self.device),
            self.last_data.edge_index.to(self.device),
            self.last_data.edge_type.to(self.device),
            self.last_data.node_type.to(self.device)
        )
    
    def close(self):
        if self.driver:
            self.driver.close()

def main():   
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PWD]):
        raise ValueError("Missing required environment variables. Please check NEO4J_URI, NEO4J_USER, and NEO4J_PWD")
    
    print(f"Connecting to Neo4j at {NEO4J_URI}")
    
    predictor = RGCNPredictor(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PWD,
        model_path='graph_model/trained_model/rgcn_model.pt'
        
    )
    
    try:
        node_labels = ["Model", "Device", "Deployment", "Server", "Service"]
        relationship_types = ["deployedIn", "used", "requestedBy", "modelOf"]
        
        print("Extracting current graph data...")
        predictor.extract_graph_data(node_labels, relationship_types)
        
        #  Finding  potential connections for a node
        test_node = "4:541e34b4-9f36-4570-9bcc-626a32d21a74:7"
        print(f"\nFinding potential connections for node {test_node}:")
        potential_connections = predictor.find_potential_connections(
            test_node,
            top_k=10,
            threshold=0.5
        )
        print("potential_connections",potential_connections,"----")
        for conn in potential_connections:
            print(f"Potential connection to {conn['target_id']} "
                  f"({conn['target_type']}) with probability {conn['probability']:.4f}")
        
        #Finding missing edges in the graph
        print("\nFinding potentially missing edges in the graph:")
        missing_edges = predictor.find_missing_edges(threshold=0.8)
        
        print("\nTop potentially missing edges:")
        for edge in missing_edges[:10]:  # Show top 10
            print(f"Missing edge: {edge['source_id']} ({edge['source_type']}) -> "
                  f"{edge['target_id']} ({edge['target_type']}) "
                  f"with probability {edge['probability']:.4f}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
    finally:
        predictor.close()

if __name__ == "__main__":
    main()