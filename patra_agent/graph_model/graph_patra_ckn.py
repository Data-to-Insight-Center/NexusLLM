import os
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_fscore_support, confusion_matrix
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from rgcn_model import RGCNLinkPrediction
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import negative_sampling
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PWD")

class RGCNTrainer:
    def __init__(self, predictor: RGCNLinkPrediction):
        self.predictor = predictor

    def split_edge_index(self, data: Data, val_ratio: float = 0.1, 
                        test_ratio: float = 0.2) -> Tuple[Data, Data, Data]:
        num_edges = data.edge_index.size(1)

        perm = torch.randperm(num_edges)

        test_size = int(num_edges * test_ratio)
        val_size = int(num_edges * val_ratio)
        
        # Create split masks for efficient indexing
        test_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        
        test_mask[perm[:test_size]] = True
        val_mask[perm[test_size:test_size + val_size]] = True
        train_mask[perm[test_size + val_size:]] = True
        
        train_data = self._create_split_data(data, train_mask)
        val_data = self._create_split_data(data, val_mask)
        test_data = self._create_split_data(data, test_mask)
        
        return train_data, val_data, test_data
    
    def _create_split_data(self, data: Data, mask: torch.Tensor) -> Data:
        split_data = Data(
            x=data.x,
            edge_index=data.edge_index[:, mask],
            edge_type=data.edge_type[mask],
            node_type=data.node_type,
            node_properties = data.node_properties,
            num_nodes=data.num_nodes
        )
        return split_data

    @torch.no_grad()
    def evaluate_split(self, model: torch.nn.Module, data: Data, 
                      split_edge_index: torch.Tensor, batch_size: int = 10000) -> Dict[str, float]:
        model.eval()
        device = next(model.parameters()).device
        
        z = model(data.x.to(device), 
                 data.edge_index.to(device),
                 data.edge_type.to(device),
                 data.node_type.to(device),
                 data.node_properties.to(device))
        
        all_preds = []
        all_true = []
        
        for i in range(0, split_edge_index.size(1), batch_size):
            batch_edge_index = split_edge_index[:, i:i+batch_size].to(device)
            
            pos_pred = model.decode(z, batch_edge_index).cpu()
            all_preds.append(pos_pred)
            all_true.append(torch.ones_like(pos_pred))

            neg_edge_index = negative_sampling(
                edge_index=batch_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=batch_edge_index.size(1)
            ).to(device)
            
            neg_pred = model.decode(z, neg_edge_index).cpu()
            all_preds.append(neg_pred)
            all_true.append(torch.zeros_like(neg_pred))
        
        pred = torch.cat(all_preds)
        true = torch.cat(all_true)
        auc = roc_auc_score(true, pred)
        ap = average_precision_score(true, pred)
        f1 = f1_score(true, (pred > 0.5).float())
        precision, recall, f1_micro, _ = precision_recall_fscore_support(true, (pred > 0.5).float(), average='micro')
        conf_matrix = confusion_matrix(true, (pred > 0.5).float())
        mrr = self.calculate_mrr(true, pred)
        precision_at_k = self.calculate_precision_at_k(true, pred, k=10)
        recall_at_k = self.calculate_recall_at_k(true, pred, k=10)

        return {
            "AUC": auc,
            "Average Precision": ap,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Confusion Matrix": conf_matrix.tolist(),
            "MRR": mrr,
            "Precision@10": precision_at_k,
            "Recall@10": recall_at_k
        }
    
    def calculate_mrr(self, true: torch.Tensor, pred: torch.Tensor) -> float:
        sorted_indices = torch.argsort(pred, descending=True)
        sorted_true = true[sorted_indices]
        ranks = torch.arange(1, len(sorted_true) + 1)
        reciprocal_ranks = 1 / ranks[sorted_true == 1]
        return reciprocal_ranks.sum().item()

    def calculate_precision_at_k(self, true: torch.Tensor, pred: torch.Tensor, k: int) -> float:
        sorted_indices = torch.argsort(pred, descending=True)
        top_k = sorted_indices[:k]
        return true[top_k].sum().item() / k

    def calculate_recall_at_k(self, true: torch.Tensor, pred: torch.Tensor, k: int) -> float:
        sorted_indices = torch.argsort(pred, descending=True)
        top_k = sorted_indices[:k]
        return true[top_k].sum().item() / true.sum().item()

def main():
    
    predictor = RGCNLinkPrediction(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PWD
    )
    
    try:
        trainer = RGCNTrainer(predictor)
        node_labels = ["Model", "Device", "Deployment", "Server", "Service"]
        relationship_types = ["deployedIn", "used", "requestedBy", "modelOf"]
        
        print("Extracting graph data...")
        data = predictor.extract_graph_data(node_labels, relationship_types)
        
        print("Splitting data into train/validation/test sets...")
        train_data, val_data, test_data = trainer.split_edge_index(data)
        
        print("Training RGCN model...")
        predictor.train_model(train_data, num_epochs=200)
        
        print("\nEvaluating on test set...")
        metrics = trainer.evaluate_split(predictor.model, test_data, test_data.edge_index)
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value}")
        
        print("\nSaving the trained model...")
        predictor.save_model('rgcn_model3.pt')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e
    finally:
        predictor.close()

if __name__ == "__main__":
    main()