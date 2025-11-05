import os
import torch
import tqdm
from typing import List, Tuple
from torch_geometric.data import Data, HeteroData
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
from prgcn_model import PRGCN
from graph_loader import GraphLoader
from dotenv import load_dotenv
load_dotenv()

class PRGCNTrainer:
    def __init__(self, node_mapping, node_type_mapping, relation_mapping):
        self.node_mapping = node_mapping
        self.node_type_mapping = node_type_mapping
        self.relation_mapping = relation_mapping
        self.model = None
        self.loss_history = []

    
    def train_model(self, data: Data, num_epochs: int = 300, 
                    hidden_channels: int = 16, negative_sampling_ratio: float = 1.0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)

        max_properties = data.node_properties.size(1)
    
        self.model = PRGCN(
            num_nodes=data.num_nodes,
            num_node_types=len(self.node_type_mapping),
            num_relations=len(self.relation_mapping),
            hidden_channels=hidden_channels,
            num_bases=4,
            max_properties=max_properties
        ).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        best_auc = 0
        patience = 20
        patience_counter = 0
    
        for epoch in range(num_epochs):
            self.model.train()
            optimizer.zero_grad()
            
            z = self.model(data.x, data.edge_index, data.edge_type, data.node_type, data.node_properties)
            
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=int(data.edge_index.size(1) * negative_sampling_ratio)
            )
            pos_edge_index = data.edge_index
            pos_pred = self.model.decode(z, pos_edge_index)
            neg_pred = self.model.decode(z, neg_edge_index)
            
            loss = self.compute_loss(pos_pred, neg_pred)

            auc, ap = self.evaluate_model(data) 
            print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, AUC: {auc}, Avg Precision: {ap}')

            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

            # Evaluate model every 5 epochs
            if (epoch + 1) % 5 == 0:
                auc, ap = self.evaluate_model(data)
                print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, Avg Precision: {ap:.4f}')

                scheduler.step()

                if auc is not None and auc > best_auc:
                    best_auc = auc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    
    def compute_loss(self,pos_pred: torch.Tensor, neg_pred: torch.Tensor) -> torch.Tensor:
        pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
        neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        return pos_loss + neg_loss
    
    @torch.no_grad()
    def evaluate_model(self, data: Data) -> Tuple[float, float]:
        self.model.eval()

        z = self.model(data.x, data.edge_index, data.edge_type, data.node_type, data.node_properties)
        batch_size = 20000
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

        if len(true) == 0 or len(pred) == 0 or len(set(true.tolist())) <= 1:
            return None, None 

        auc = roc_auc_score(true, pred)
        ap = average_precision_score(true, pred)

        return auc, ap
    
    def plot_loss(self):
        """Plots the training loss over epochs."""
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o', linestyle='-', color='b')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.grid(True)
        plt.show()



def main():
    
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USER")  
    NEO4J_PWD = os.getenv("NEO4J_PWD")        

    graph_loader = GraphLoader(uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PWD)

    
    node_labels = ['Model', 'ModelCard', 'Device', 'Deployment', 'Server', 'Service', 'Datasheet', 'BiasAnalysis', 'ExplainabilityAnalysis', 'ModelRequirements']
    relationship_types = ['deployedIn', 'modelOf', 'requestedBy', 'used', 'BIAS_ANALYSIS','REQUIREMENTS','XAI_ANALYSIS','TRAINED_ON','USED']

    try:
        print("Extracting graph data from Neo4j...")
        graph_data = graph_loader.extract_graph_data(node_labels, relationship_types)
        print("Graph data extracted successfully!")
        print(f"Nodes: {graph_data.x.shape}")
        
        print("Starting model training...")
        model_trainer = PRGCNTrainer(
            node_mapping=graph_loader.node_mapping,
            node_type_mapping=graph_loader.node_type_mapping,
            relation_mapping=graph_loader.relation_mapping
        )
        model_trainer.train_model(data=graph_data, num_epochs=200, hidden_channels=64, negative_sampling_ratio=1.0)
        print("Model training completed!")

        model_save_path = "prgcn_wo_relu.pth"

        if model_trainer.model:
            torch.save({
                'model_state_dict': model_trainer.model.state_dict(),
                'node_mapping': graph_loader.node_mapping, 
                'node_type_mapping': graph_loader.node_type_mapping, 
                'relation_mapping': graph_loader.relation_mapping,
                'max_properties': graph_data.node_properties.size(1)
            }, model_save_path)
            print(f"Trained model and mappings saved at: {model_save_path}")
        else:
            print("Warning: Model object is None, skipping save.")

    finally:

        graph_loader.close() 
        print("Neo4j connection closed.")

if __name__ == "__main__":
    main()