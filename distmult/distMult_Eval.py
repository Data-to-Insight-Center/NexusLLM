
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, confusion_matrix
from distMult_model import DistMult
from patra_rgcn.graph_loader import GraphLoader

class DistMultEvaluator:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.num_relations = None  

    def load_model(self, num_nodes: int, num_relations: int, embedding_dim=200):
        """ Load the trained model and transfer embeddings to the new graph """
        self.num_relations = num_relations  
        self.model = DistMult(num_nodes, num_relations, embedding_dim).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_dict = self.model.state_dict()
        pretrained_dict = checkpoint["model_state_dict"]

        if "entity_embedding.weight" in pretrained_dict:
            trained_size = pretrained_dict["entity_embedding.weight"].shape[0]
            new_size = model_dict["entity_embedding.weight"].shape[0]
            min_size = min(trained_size, new_size)
            model_dict["entity_embedding.weight"][:min_size] = pretrained_dict["entity_embedding.weight"][:min_size]

        updated_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(updated_dict)
        self.model.load_state_dict(model_dict, strict=False)
        self.model.eval()

        print(f"Model loaded from {self.model_path}, with {len(updated_dict)} parameters restored.")

    def fine_tune_model(self, data: Data, num_epochs: int = 20, lr: float = 1e-4):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            pos_edge_index = data.edge_index.to(self.device)
            pos_edge_type = data.edge_type.to(self.device) 
            pos_pred = self.model(pos_edge_index, pos_edge_type)
            pos_labels = torch.ones_like(pos_pred)

            neg_edge_index = negative_sampling(
                edge_index=pos_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=pos_edge_index.size(1) * 5
            ).to(self.device)

            neg_edge_type = torch.randint(0, self.num_relations, (neg_edge_index.size(1),), dtype=torch.long).to(self.device)
            neg_pred = self.model(neg_edge_index, neg_edge_type)
            neg_labels = torch.zeros_like(neg_pred)

            loss = F.binary_cross_entropy(pos_pred, pos_labels) + F.binary_cross_entropy(neg_pred, neg_labels)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Fine-tuning Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    @torch.no_grad()
    def evaluate(self, data: Data) -> dict:
        data = data.to(self.device)
        self.model.eval()

        pos_edge_index = data.edge_index.to(self.device)
        pos_edge_type = data.edge_type.to(self.device)  
        pos_pred = self.model(pos_edge_index, pos_edge_type)

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=pos_edge_index.size(1) * 5 
        ).to(self.device)

        neg_edge_type = torch.randint(0, self.num_relations, (neg_edge_index.size(1),), dtype=torch.long).to(self.device)
        neg_pred = self.model(neg_edge_index, neg_edge_type)

        pred = torch.cat([pos_pred, neg_pred]).cpu()
        true_labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])

        return self.calculate_metrics(true_labels, pred)

    def calculate_metrics(self, true_labels: torch.Tensor, predictions: torch.Tensor) -> dict:
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
    def calculate_mrr(self, true_labels: torch.Tensor, predictions: torch.Tensor) -> float:
        sorted_indices = torch.argsort(predictions, descending=True)
        sorted_true = true_labels[sorted_indices]
        ranks = torch.arange(1, len(sorted_true) + 1)
        reciprocal_ranks = 1 / ranks[sorted_true == 1]
        return reciprocal_ranks.sum().item()

    def calculate_hit_at_k(self, true_labels: torch.Tensor, predictions: torch.Tensor, k: int) -> float:
        sorted_indices = torch.argsort(predictions, descending=True)
        top_k = sorted_indices[:k]
        return true_labels[top_k].sum().item() / min(k, true_labels.sum().item())


def main():
    MODEL_PATH = "distmult_model_new.pth"
    NEO4J_URI = "bolt://149.165.175.36:7688"
    NEO4J_USERNAME = os.getenv("NEO4J_USER")
    NEO4J_PWD = os.getenv("NEO4J_PWD")

    loader = GraphLoader(NEO4J_URI, NEO4J_USERNAME, NEO4J_PWD)
    data = loader.extract_graph_data(
        ["Model", "Device", "Deployment", "Server", "Service"],
        ["deployedIn", "used", "requestedBy", "modelOf"]
    )
    loader.close()

    num_nodes = data.num_nodes
    num_relations = len(set(data.edge_type.tolist()))

    evaluator = DistMultEvaluator(MODEL_PATH)
    evaluator.load_model(num_nodes, num_relations, embedding_dim=200)

    print("Fine-tuning the model on the new graph...")
    evaluator.fine_tune_model(data, num_epochs=10)

    print("\nEvaluating the model on the new graph...")
    evaluation_metrics = evaluator.evaluate(data)

    print("\nModel Evaluation Metrics on New Graph:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
