
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import RGCNConv
from tqdm import tqdm
from vanilla_rgcn_model import VanillaRGCN
from patra_rgcn.graph_loader import GraphLoader

class VanillaRGCNTrainer:
    def __init__(self, data: Data, hidden_channels: int = 16, dropout: float = 0.1, lr: float = 0.01):
        """Initialize the trainer with data and model parameters."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Graph Data Details
        self.num_nodes = data.num_nodes
        self.num_relations = len(set(data.edge_type.tolist()))
        self.data = data.to(self.device)

        # Initialize VanillaRGCN Model
        self.model = VanillaRGCN(
            num_nodes=self.num_nodes,
            num_relations=self.num_relations,
            hidden_channels=hidden_channels,
            dropout=dropout
        ).to(self.device)

        # Optimizer & Learning Rate Scheduler
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=10, factor=0.5)

        # Loss Function
        self.criterion = nn.BCELoss()

    def train(self, num_epochs: int = 100, negative_sampling_ratio: float = 2.0):
        """Train the VanillaRGCN model with negative sampling."""
        best_auc = 0
        patience = 20
        patience_counter = 0

        for epoch in tqdm(range(num_epochs), desc="Training VanillaRGCN"):
            self.model.train()
            self.optimizer.zero_grad()

            # Compute Node Embeddings
            z = self.model(self.data.x, self.data.edge_index, self.data.edge_type)

            # Positive Edges
            pos_edge_index = self.data.edge_index.to(self.device)
            pos_pred = self.model.decode(z, pos_edge_index)
            pos_labels = torch.ones_like(pos_pred)

            # Negative Sampling (2x more negative edges)
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=self.num_nodes,
                num_neg_samples=int(pos_edge_index.size(1) * negative_sampling_ratio)
            ).to(self.device)

            neg_pred = self.model.decode(z, neg_edge_index)
            neg_labels = torch.zeros_like(neg_pred)

            # Compute Loss
            loss = self.criterion(pos_pred, pos_labels) + self.criterion(neg_pred, neg_labels)
            loss.backward()
            self.optimizer.step()

            # Evaluate every 10 epochs
            if (epoch + 1) % 10 == 0:
                auc, ap = self.evaluate()
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
                self.scheduler.step(auc)

                # Early Stopping
                if auc > best_auc:
                    best_auc = auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break

    @torch.no_grad()
    def evaluate(self) -> tuple:
        """Evaluate model performance on positive and negative samples."""
        self.model.eval()

        # Compute Node Embeddings
        z = self.model(self.data.x, self.data.edge_index, self.data.edge_type)

        # Positive Predictions
        pos_edge_index = self.data.edge_index.to(self.device)
        pos_pred = self.model.decode(z, pos_edge_index)

        # Negative Predictions
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=self.num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        ).to(self.device)

        neg_pred = self.model.decode(z, neg_edge_index)

        # Compute Metrics
        pred = torch.cat([pos_pred, neg_pred]).cpu()
        true_labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        auc = roc_auc_score(true_labels, pred)
        ap = average_precision_score(true_labels, pred)

        return auc, ap

    def save_model(self, model_path: str = "new_vanilla_rgcn_model.pth"):
        """Save trained VanillaRGCN model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, model_path)
        print(f"Model saved at {model_path}")


def main():
    """Main function to train and save VanillaRGCN on a Neo4j graph."""
    NEO4J_URI = "bolt://149.165.153.250:7688"
    NEO4J_USERNAME = os.getenv("NEO4J_USER")
    NEO4J_PWD = os.getenv("NEO4J_PWD")

    # Initialize GraphLoader
    graph_loader = GraphLoader(uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PWD)

    # Define Nodes & Relationships to Extract
    node_labels = ["Model", "Device", "Deployment", "Server", "Service"]
    relationship_types = ["deployedIn", "used", "requestedBy", "modelOf"]

    try:
        print("Extracting graph data from Neo4j for training...")
        graph_data = graph_loader.extract_graph_data(node_labels, relationship_types)
        print(f"Graph Data Extracted! Nodes: {graph_data.x.shape[0]}, Edges: {graph_data.edge_index.shape[1]}")

        print("Initializing VanillaRGCN model and starting training...")
        trainer = VanillaRGCNTrainer(graph_data)
        trainer.train(num_epochs=100, negative_sampling_ratio=2.0)  # Train for 100 epochs

        print("Saving the trained VanillaRGCN model...")
        model_save_path = "new_vanilla_rgcn_model.pth"
        trainer.save_model(model_save_path)
        print(f"Model successfully saved at {model_save_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        graph_loader.close()
        print("Neo4j connection closed.")


if __name__ == "__main__":
    main()