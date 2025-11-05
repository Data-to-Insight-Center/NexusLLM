import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, confusion_matrix, precision_recall_curve
from prgcn_model import PRGCN  
from graph_loader import GraphLoader  
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
load_dotenv()

class PRGCNEvaluator:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None

    def load_model(self, num_nodes: int, num_node_types: int, num_relations: int, max_properties: int):
        self.model = PRGCN(
            num_nodes=num_nodes,
            num_node_types=num_node_types,
            num_relations=num_relations,
            hidden_channels=64,
            num_bases=4,
            max_properties=max_properties
        ).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
        
        if "node_emb.weight" in checkpoint and checkpoint["node_emb.weight"].shape[1] == model_dict["node_emb.weight"].shape[1]:
            old_emb = checkpoint["node_emb.weight"]
            new_emb = model_dict["node_emb.weight"]
            new_size = new_emb.shape[0]
            model_dict["node_emb.weight"][:min(len(old_emb), new_size)] = old_emb[:min(len(old_emb), new_size)]

        if "node_type_emb.weight" in checkpoint and checkpoint["node_type_emb.weight"].shape[1] == model_dict["node_type_emb.weight"].shape[1]:
            old_emb = checkpoint["node_type_emb.weight"]
            new_emb = model_dict["node_type_emb.weight"]
            new_size = new_emb.shape[0]
            model_dict["node_type_emb.weight"][:min(len(old_emb), new_size)] = old_emb[:min(len(old_emb), new_size)]

        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict, strict=False)
        self.model.eval()

        print(f"Model loaded from {self.model_path}, with {len(pretrained_dict)} parameters restored.")

    def optimal_threshold(self, true_labels, predictions):
        precisions, recalls, thresholds = precision_recall_curve(true_labels, predictions)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        return thresholds[np.argmax(f1_scores)]

    @torch.no_grad()
    def evaluate(self, data: Data) -> dict:
        data = data.to(self.device)
        self.model.eval()
        z = self.model( 
            data.x, data.edge_index, data.edge_type,
            data.node_type, data.node_properties
        )
        pos_pred = self.model.decode(z, data.edge_index).cpu()
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1)
        ).to(self.device)
        
        neg_pred = self.model.decode(z, neg_edge_index).cpu()
        print("pos_pred type:", type(pos_pred))
        print("neg_pred type:", type(neg_pred))

        pred = torch.cat([pos_pred, neg_pred])
        true_labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        return self.calculate_metrics(true_labels, pred, pos_pred, neg_pred)

    def calculate_metrics(self, true_labels: torch.Tensor, predictions: torch.Tensor, pos_pred: torch.Tensor, neg_pred: torch.Tensor) -> dict:
        threshold = self.optimal_threshold(true_labels, predictions)
        predictions_binary = predictions > threshold

        auc = roc_auc_score(true_labels, predictions)
        ap = average_precision_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions_binary)
        precision, recall, _, _ = precision_recall_fscore_support(true_labels, predictions_binary, average='binary')

        # precisions, recalls, thresholds = precision_recall_curve(true_labels, predictions)
        # optimal_threshold = thresholds[np.argmax(precisions - recalls)]
        # predictions_binary = predictions > optimal_threshold

        # f1 = f1_score(true_labels, predictions_binary)
        # precision, recall, _, _ = precision_recall_fscore_support(true_labels, predictions_binary, average='binary')
        # conf_matrix = confusion_matrix(true_labels, predictions_binary)

        conf_matrix = confusion_matrix(true_labels, predictions > 0.5)
        precision_at_10 = self.precision_at_k(true_labels, predictions, k=10)
        recall_at_10 = self.recall_at_k(true_labels, predictions, k=10)
        hit_at_5 = self.hit_at_k(true_labels, predictions, k=5)
        hit_at_10 = self.hit_at_k(true_labels, predictions, k=10)
        mrr = self.mean_reciprocal_rank(pos_pred, neg_pred)

        return {
            "AUC": auc,
            "Average Precision": ap,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Confusion Matrix": conf_matrix.tolist(),
            "Precision@10": precision_at_10,
            "Recall@10": recall_at_10,
            "MRR": mrr,
            "Hit@5": hit_at_5,
            "Hit@10": hit_at_10
        }
    
    def normalize_features(self, data: Data):
        scaler = StandardScaler()
        data.node_properties = torch.tensor(scaler.fit_transform(data.node_properties), dtype=torch.float32)
        return data

    def train_property_encoder(self, data: Data, num_epochs: int = 10, lr: float = 0.005):
        data = data.to(self.device)
        optimizer = torch.optim.Adam(self.model.property_encoder.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()  # Example loss for property encoding

        self.model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()

            x = self.model.property_encoder(data.node_properties)
            loss = loss_fn(x, torch.zeros_like(x))  # Dummy loss for illustration

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Property Encoder Loss: {loss.item():.4f}")

        self.model.eval()
        print("Property encoder fine-tuning completed.")

    def precision_at_k(self, true_labels: torch.Tensor, predictions: torch.Tensor, k: int) -> float:
        sorted_indices = torch.argsort(predictions, descending=True)
        top_k_indices = sorted_indices[:k]
        return true_labels[top_k_indices].sum().item() / k

    def recall_at_k(self, true_labels: torch.Tensor, predictions: torch.Tensor, k: int) -> float:
        sorted_indices = torch.argsort(predictions, descending=True)
        top_k_indices = sorted_indices[:k]
        return true_labels[top_k_indices].sum().item() / true_labels.sum().item()

    def hit_at_k(self, true_labels: torch.Tensor, predictions: torch.Tensor, k: int) -> float:
        sorted_indices = torch.argsort(predictions, descending=True)
        top_k_indices = sorted_indices[:k]
        hits = true_labels[top_k_indices].sum().item()
        return hits / min(k, true_labels.sum().item())
    def mean_reciprocal_rank(self, pos_pred: torch.Tensor, neg_pred: torch.Tensor) -> float:
        all_scores = torch.cat([pos_pred, neg_pred], dim=0)
        sorted_indices = torch.argsort(all_scores, descending=True)
        ranks = (sorted_indices < len(pos_pred)).nonzero(as_tuple=True)[0] + 1
        mrr = (1.0 / ranks.float()).mean().item()
        return mrr

def main():
    NEO4J_URI = "bolt://149.165.153.250:7688"
    NEO4J_USERNAME = os.getenv("NEO4J_USER")
    NEO4J_PWD = os.getenv("NEO4J_PWD")
    MODEL_PATH = "prgcn_wo_relu.pth"
    

    graph_loader = GraphLoader(uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PWD)

    node_labels = ["Model", "Device", "Deployment", "Server", "Service", "ModelCard", "Datasheet","Author","Category","InputType"]
    relationship_types = ["deployedIn", "used", "requestedBy", "modelOf","TRAINED_ON","USED","author","inputType","category"]

    try:
        print("Extracting new graph data for evaluation...")
        graph_data = graph_loader.extract_graph_data(node_labels, relationship_types)
        print("graph_data", graph_data)
        # num_nodes = graph_data.num_nodes
        # # num_node_types = len(set(graph_data.node_type.tolist()))
        # # num_relations = len(set(graph_data.edge_type.tolist()))
        # num_node_types = len(set(graph_data.node_type))
        # num_relations = len(set(graph_data.edge_type))
        # max_properties = graph_data.node_properties.size(1)

        evaluator = PRGCNEvaluator(MODEL_PATH)
        graph_data = evaluator.normalize_features(graph_data)
        num_nodes = graph_data.num_nodes
        # num_node_types = len(set(graph_data.node_type.tolist()))
        # num_relations = len(set(graph_data.edge_type.tolist()))
        num_node_types = len(set(graph_data.node_type))
        num_relations = len(set(graph_data.edge_type))
        max_properties = graph_data.node_properties.size(1)
        evaluator.load_model(num_nodes, num_node_types, num_relations, max_properties)

        print("Fine-tuning the property encoder...")
        evaluator.train_property_encoder(graph_data, num_epochs=10)

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
