import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, confusion_matrix
from vanilla_rgcn_model import VanillaRGCN
from graph_loader import GraphLoader

class VanillaRGCNEvaluator:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None

    def load_model(self, num_nodes: int, num_relations: int, hidden_channels: int = 16):
        """
        Load trained VanillaRGCN model and apply transfer learning.
        """
        self.model = VanillaRGCN(num_nodes, num_relations, hidden_channels).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_dict = self.model.state_dict()

        # Apply Transfer Learning: Load only matching parameters
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}

        # Handle node embedding transfer carefully
        if "node_emb.weight" in checkpoint:
            old_emb = checkpoint["node_emb.weight"]
            new_emb = model_dict["node_emb.weight"]
            new_size = new_emb.shape[0]
            model_dict["node_emb.weight"][:min(len(old_emb), new_size)] = old_emb[:min(len(old_emb), new_size)]

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        self.model.eval()

        print(f"Model loaded from {self.model_path}, with {len(pretrained_dict)} parameters restored.")

    def fine_tune_model(self, data: Data, num_epochs: int = 30, lr: float = 5e-4):
        """
        Fine-tune **full model** on the new graph to improve performance.
        """
        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            z = self.model(data.x, data.edge_index, data.edge_type)

            pos_edge_index = data.edge_index.to(self.device)
            pos_pred = self.model.decode(z, pos_edge_index)
            pos_labels = torch.ones_like(pos_pred)

            # **Hard Negative Sampling: Sample from high-degree nodes**
            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_edge_index.size(1) * 5,  # **Increase negatives to 5x**
                method="sparse"  # Sparse method improves sampling quality
            ).to(self.device)

            neg_pred = self.model.decode(z, neg_edge_index)
            neg_labels = torch.zeros_like(neg_pred)

            loss = F.binary_cross_entropy(pos_pred, pos_labels) + F.binary_cross_entropy(neg_pred, neg_labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Fine-tuning Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        self.model.eval()
        print("Fine-tuning completed.")

    @torch.no_grad()
    def evaluate(self, data: Data) -> dict:
        """
        Evaluate model performance on the new graph with **hard negative sampling**.
        """
        data = data.to(self.device)
        self.model.eval()

        z = self.model(data.x, data.edge_index, data.edge_type)
        pos_edge_index = data.edge_index.to(self.device)
        pos_pred = self.model.decode(z, pos_edge_index)

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1) * 5,  # **Increase negatives to 5x**
            method="sparse"
        ).to(self.device)

        neg_pred = self.model.decode(z, neg_edge_index)

        pred = torch.cat([pos_pred, neg_pred]).cpu()
        true_labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        return self.calculate_metrics(true_labels, pred)
    
    def calculate_mrr(self, true_labels: torch.Tensor, predictions: torch.Tensor) -> float:
        """ Compute Mean Reciprocal Rank (MRR) """
        sorted_indices = torch.argsort(predictions, descending=True)
        sorted_true = true_labels[sorted_indices]
        ranks = torch.arange(1, len(sorted_true) + 1)
        reciprocal_ranks = 1 / ranks[sorted_true == 1]
        return reciprocal_ranks.sum().item()

    def calculate_hit_at_k(self, true_labels: torch.Tensor, predictions: torch.Tensor, k: int) -> float:
        """ Compute Hit@K """
        sorted_indices = torch.argsort(predictions, descending=True)
        top_k = sorted_indices[:k]
        return true_labels[top_k].sum().item() / min(k, true_labels.sum().item())

    def calculate_metrics(self, true_labels: torch.Tensor, predictions: torch.Tensor) -> dict:
        """
        Compute AUC, Average Precision, and other evaluation metrics.
        """
        auc = roc_auc_score(true_labels, predictions)
        ap = average_precision_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions > 0.5)
        precision, recall, _, _ = precision_recall_fscore_support(true_labels, predictions > 0.5, average='binary')
        conf_matrix = confusion_matrix(true_labels, predictions > 0.5)
        mrr = self.calculate_mrr(true_labels, predictions)
        hit_at_5 = self.calculate_hit_at_k(true_labels, predictions, k=5)
        hit_at_10 = self.calculate_hit_at_k(true_labels, predictions, k=10)

        return {
            "AUC": auc,
            "Average Precision": ap,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Confusion Matrix": conf_matrix.tolist(),
            "MRR": mrr,
            "Hit@5": hit_at_5,
            "Hit@10": hit_at_10
        }

def main():
    NEO4J_URI = "bolt://149.165.175.36:7688"
    NEO4J_USERNAME = os.getenv("NEO4J_USER")
    NEO4J_PWD = os.getenv("NEO4J_PWD")
    MODEL_PATH = "new_vanilla_rgcn_model.pth"

    graph_loader = GraphLoader(uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PWD)

    node_labels = ["Model", "Device", "Deployment", "Server", "Service"]
    relationship_types = ["deployedIn", "used", "requestedBy", "modelOf"]

    try:
        print("Extracting new graph data for evaluation...")
        graph_data = graph_loader.extract_graph_data(node_labels, relationship_types)

        num_nodes = graph_data.num_nodes
        num_relations = len(set(graph_data.edge_type.tolist()))

        evaluator = VanillaRGCNEvaluator(MODEL_PATH)
        evaluator.load_model(num_nodes, num_relations)

        print("Fine-tuning the full model...")
        evaluator.fine_tune_model(graph_data, num_epochs=30)

        print("Evaluating the model on the new graph...")
        evaluation_metrics = evaluator.evaluate(graph_data)

        print("\nModel Evaluation Metrics on New Graph:")
        for metric, value in evaluation_metrics.items():
            print(f"{metric}: {value}")

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
    finally:
        graph_loader.close()
        print("Neo4j connection closed.")

if __name__ == "__main__":
    main()
