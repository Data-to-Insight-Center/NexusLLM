import os
import torch
import torch_geometric.data
from graph_loader import GraphLoader
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Any
from prgcn_model import PRGCN
from constraints import VALID_LINK_CONSTRAINTS

load_dotenv()

class PRGCNPredictor:

    def __init__(self, model_class, hidden_channels=64, num_bases=4):
        self.model = None
        self.model_class = model_class
        self.hidden_channels = hidden_channels
        self.num_bases = num_bases
    
    def load_model(self, path: str):
        """Loads the model state and required mappings from a saved checkpoint."""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        node_mapping = checkpoint['node_mapping']
        node_type_mapping = checkpoint['node_type_mapping']
        relation_mapping = checkpoint['relation_mapping']
        
        max_properties = checkpoint['max_properties']

        self.model = self.model_class( 
            num_nodes=len(node_mapping),
            num_node_types=len(node_type_mapping),
            num_relations=len(relation_mapping),
            hidden_channels=self.hidden_channels,
            num_bases=self.num_bases,
            max_properties=max_properties
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded successfully from {path}")
        
        return node_mapping, node_type_mapping, relation_mapping

    @torch.no_grad()
    def predict_links(self, data: torch_geometric.data.Data, candidate_edges: torch.Tensor) -> torch.Tensor:
        """Scores a set of candidate edges using the loaded model."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        z = self.model(data.x, data.edge_index, data.edge_type, data.node_type, data.node_properties)
        
        scores = self.model.decode(z, candidate_edges)
        
        return scores.cpu()

def predict_top_links(
    model_path: str, 
    neo4j_uri: str, 
    neo4j_user: str, 
    neo4j_pwd: str, 
    top_n: int = 5
) -> List[Dict[str, Any]]:    

    node_labels = ['Model', 'ModelCard', 'Device', 'Deployment', 'Server', 'Service', 'Datasheet', 'BiasAnalysis', 'ExplainabilityAnalysis', 'ModelRequirements']
    relationship_types = ['deployedIn', 'modelOf', 'requestedBy', 'used', 'BIAS_ANALYSIS','REQUIREMENTS','XAI_ANALYSIS','TRAINED_ON','USED']
    predictor = PRGCNPredictor(model_class=PRGCN, hidden_channels=64)
    node_mapping, _, _ = predictor.load_model(model_path)


    graph_loader = GraphLoader(uri=neo4j_uri, username=neo4j_user, password=neo4j_pwd)
    
    try:
        print("Extracting current graph data for embedding generation...")
        graph_data = graph_loader.extract_graph_data(node_labels, relationship_types)
        print("Graph data extracted successfully.")
        
        print("\n--- Searching Neo4j for Unconnected Node Pairs ---")
        neo4j_candidate_ids = graph_loader.find_unconnected_nodes(
            node_labels=node_labels,
            constraints= VALID_LINK_CONSTRAINTS,
            limit=100)
        
        pyg_candidate_pairs = []
        name_lookup = {}
        pyg_to_neo4j_id = {v: k for k, v in node_mapping.items()} 

        if not neo4j_candidate_ids:
            print("No unconnected candidates found by Cypher query.")
            return []

        for neo4j_id_a, neo4j_id_b, name_a, name_b in neo4j_candidate_ids:
            if neo4j_id_a in node_mapping and neo4j_id_b in node_mapping:
                pyg_idx_a = node_mapping[neo4j_id_a]
                pyg_idx_b = node_mapping[neo4j_id_b]
                pyg_candidate_pairs.append((pyg_idx_a, pyg_idx_b))
                name_lookup[pyg_idx_a] = name_a
                name_lookup[pyg_idx_b] = name_b
        
        if not pyg_candidate_pairs:
            print("Warning: All Neo4j candidates were filtered out (not in the current graph_data).")
            return []

        source_indices = torch.tensor([pair[0] for pair in pyg_candidate_pairs], dtype=torch.long)
        target_indices = torch.tensor([pair[1] for pair in pyg_candidate_pairs], dtype=torch.long)
        candidate_indices = torch.stack([source_indices, target_indices], dim=0)


        print(f"\n--- Starting Link Prediction for {candidate_indices.size(1)} Candidates ---")
        prediction_scores = predictor.predict_links(data=graph_data, candidate_edges=candidate_indices)

        raw_results = zip(candidate_indices[0], candidate_indices[1], prediction_scores)
        
        sorted_results = sorted(raw_results, key=lambda x: x[2], reverse=True) 

        final_recommendations = []
        for i, (src_pyg, tgt_pyg, score) in enumerate(sorted_results):
            if i >= top_n:
                break
                
            final_recommendations.append({
                "source_neo4j_id": pyg_to_neo4j_id.get(src_pyg.item()),
                "source_name": str(name_lookup.get(src_pyg.item(), "N/A") or "N/A"),
                "target_neo4j_id": pyg_to_neo4j_id.get(tgt_pyg.item()),
                "target_name": str(name_lookup.get(tgt_pyg.item(), "N/A") or "N/A"),
                "score": score.item(),
            })
            
        return final_recommendations
    
    finally:
        graph_loader.close()

def main():
    
    MODEL_PATH = "prgcn_wo_relu.pth" 
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USER")
    NEO4J_PWD = os.getenv("NEO4J_PWD")
    
    print("--- Running Link Prediction Workflow ---")
    top_recommendations = predict_top_links(
        model_path=MODEL_PATH,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USERNAME,
        neo4j_pwd=NEO4J_PWD,
        top_n=5
    )

    if top_recommendations:
        print("\n--- FINAL TOP 5 RECOMMENDATIONS ---")
        for rec in top_recommendations:
            print(
                f"Source: {rec['source_name']:20} -> Target: {rec['target_name']:20} | " # <<< ADDED NAMES
                f"Score: {rec['score']:.4f}\n"
                f"{'':10} (IDs: {rec['source_neo4j_id']} -> {rec['target_neo4j_id']})" # <<< ADDED IDs on new line
            )    
        else:
            print("\nNo predictions were generated.")


if __name__ == "__main__":
    main()