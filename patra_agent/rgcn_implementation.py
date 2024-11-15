import os
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from patra_agent.graph_model.rgcn_model import RGCNLinkPrediction

def main():
    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USER")
    NEO4J_PWD = os.getenv("NEO4J_PWD")

    predictor = RGCNLinkPrediction(
        uri = NEO4J_URI,
        username = NEO4J_USERNAME,
        password = NEO4J_PWD
    )
    try:

        node_labels = ["Model", "Device", "Deployment","Server","Service"]
        relationship_types = ["deployedIn","used","requestedBy","modelOf"]
 
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