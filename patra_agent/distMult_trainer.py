import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from distMult_model import DistMult 
from graph_loader import GraphLoader  
from tqdm import tqdm

class DistMultTrainer:
    def __init__(self, data: Data, embedding_dim: int = 200, lr: float = 0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_nodes = data.num_nodes
        self.num_relations = len(set(data.edge_type.tolist())) if data.edge_type.numel() > 0 else 1  
        self.data = data.to(self.device)  

        self.model = DistMult(self.num_nodes, self.num_relations, embedding_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.BCELoss()

    def train(self, num_epochs: int = 100, negative_sampling_ratio: float = 2.0):
        for epoch in tqdm(range(num_epochs), desc="Training DistMult"):
            self.model.train()
            self.optimizer.zero_grad()

            pos_edge_index = self.data.edge_index.to(self.device)
            pos_edge_type = self.data.edge_type.to(self.device) if self.data.edge_type.numel() > 0 else torch.zeros(pos_edge_index.size(1), dtype=torch.long).to(self.device)
            pos_pred = self.model(pos_edge_index, pos_edge_type)
            pos_labels = torch.ones_like(pos_pred)

            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=self.num_nodes,
                num_neg_samples=int(pos_edge_index.size(1) * negative_sampling_ratio)
            ).to(self.device)

            neg_edge_type = torch.randint(0, self.num_relations, (neg_edge_index.size(1),), dtype=torch.long).to(self.device)
            neg_pred = self.model(neg_edge_index, neg_edge_type)
            neg_labels = torch.zeros_like(neg_pred)

            loss = self.criterion(pos_pred, pos_labels) + self.criterion(neg_pred, neg_labels)
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                auc, ap = self.evaluate()
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, AUC: {auc:.4f}, Avg Precision: {ap:.4f}")

    @torch.no_grad()
    def evaluate(self) -> tuple:
        self.model.eval()

        pos_edge_index = self.data.edge_index.to(self.device)
        pos_edge_type = self.data.edge_type.to(self.device) if self.data.edge_type.numel() > 0 else torch.zeros(pos_edge_index.size(1), dtype=torch.long).to(self.device)
        pos_pred = self.model(pos_edge_index, pos_edge_type)
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=self.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        ).to(self.device)

        neg_edge_type = torch.randint(0, self.num_relations, (neg_edge_index.size(1),), dtype=torch.long).to(self.device)
        neg_pred = self.model(neg_edge_index, neg_edge_type)
        pred = torch.cat([pos_pred, neg_pred]).cpu()
        true_labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        auc = roc_auc_score(true_labels, pred)
        ap = average_precision_score(true_labels, pred)

        return auc, ap

    def save_model(self, model_path: str = "distmult_model.pth"):
        """Save trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)
        print(f"Model saved at {model_path}")


def main():
    NEO4J_URI = "bolt://149.165.153.250:7688"
    NEO4J_USERNAME = os.getenv("NEO4J_USER")
    NEO4J_PWD = os.getenv("NEO4J_PWD")

    graph_loader = GraphLoader(uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PWD)


    node_labels = ["Model", "Device", "Deployment", "Server", "Service"]
    relationship_types = ["deployedIn", "used", "requestedBy", "modelOf"]

    try:
        print("Extracting graph data from Neo4j for training...")
        graph_data = graph_loader.extract_graph_data(node_labels, relationship_types)
        print(f"Graph Data Extracted! Nodes: {graph_data.x.shape[0]}, Edges: {graph_data.edge_index.shape[1]}")

        print("Initializing DistMult model and starting training...")
        trainer = DistMultTrainer(graph_data)
        trainer.train(num_epochs=100, negative_sampling_ratio=2.0)  # Train for 100 epochs

        print("Saving the trained DistMult model...")
        model_save_path = "distmult_model_new.pth"
        trainer.save_model(model_save_path)
        print(f"Model successfully saved at {model_save_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        graph_loader.close()
        print("Neo4j connection closed.")

if __name__ == "__main__":
    main()
