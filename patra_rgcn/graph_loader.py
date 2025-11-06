import os
import torch
from neo4j import GraphDatabase
from typing import List, Tuple
from torch_geometric.data import Data

class GraphLoader:
    def __init__(self, uri:str, username:str, password:str, batch_size: int=1000):
        self.driver = GraphDatabase.driver(uri, auth=(username,password))
        self.model = None
        self.node_mapping = {}
        self.node_type_mapping = {}
        self.relation_mapping = {}
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def close(self):
        self.driver.close()
    def extract_graph_data(self, node_labels: List[str], relationship_types: List[str]) -> Data:
        with self.driver.session() as session:
            nodes, nodes_types, node_properties = self.extract_nodes_batch(session, node_labels)
            edge_index, edge_type = self.extract_edges_batch(session, relationship_types)

            return Data(
                x = torch.tensor(nodes, dtype = torch.long),
                edge_index = torch.tensor(edge_index, dtype=torch.long),
                edge_type = torch.tensor(edge_type, dtype=torch.long),
                node_type = torch.tensor(nodes_types, dtype=torch.long),
                node_properties = torch.tensor(node_properties, dtype=torch.float),
                num_nodes=len(self.node_mapping)
            )
    def extract_nodes_batch(self, session, node_labels: List[str]) -> Tuple[List[int], List[int]]:
        nodes = []
        node_types = []
        node_properties = []

        def extract_device_name(device_id):
            return device_id.split('-')[0]
        
        property_mappings = {
            'Model': {
                'props': ['Backbone', 'Batch_Size', 'Learning_Rate', 'Precision', 
                        'Recall', 'test_accuracy'],
                'numeric': ['Batch_Size', 'Learning_Rate', 'Precision', 'Recall', 'test_accuracy'],
                'categorical': ['Backbone']
            },
            # 'ModelCard': {
            #     'props': ['categories', 'input_type', 'name'],
            #     'numeric': [],
            #     'categorical': ['categories', 'input_type', 'name']
            # },
            'ModelCard': {
                'props': ['name'],
                'numeric': [],
                'categorical': ['name']
            },
            'Datasheet': {
                'props': ['attribute_types', 'datapoints', 'name', 'source'],
                'numeric': ['datapoints'],
                'categorical': ['attribute_types', 'name', 'source']
            },
            'Device': {
                'props': ['device_id'],
                'numeric': [],
                'categorical': [extract_device_name(d) for d in 'device_id']
            },
            'Deployment': {
                'props': ['deployment_id', 'avg_accuracy', 'status'],
                'numeric': ['deployment_id', 'avg_accuracy', 'avg_accuracy_qoe','avg_compute_time','avg_delay_qoe',
                            'avg_probability','avg_req_acc','avg_req_delay','avg_total_qoe','total_requests'],
                'categorical': ['status']
            },
            'Server': {
                'props': ['server_id'],
                'categorical': ['server_id'],
                'numeric': []
            },
            'Service': {
                'props': ['service_id'],
                'numeric': ['service_id'],
                'categorical': []
            }
        }     
        for label in node_labels:
            offset = 0
            mapping = property_mappings.get(label, {'props': [], 'numeric': [], 'categorical': []})
            props_to_extract = mapping['props']
            while True:
                property_clause = ", ".join([
                f"n.{prop} as {prop}" for prop in props_to_extract
                ])
                
                query = f"""
                MATCH (n:{label})
                RETURN elementId(n) as id {', ' + property_clause if property_clause else ''}
                SKIP {offset} LIMIT {self.batch_size}
                """
                result = list(session.run(query))
                if not result:
                    break
                    
                for record in result:
                    node_id = record["id"]
                    if node_id not in self.node_mapping:
                        self.node_mapping[node_id] = len(self.node_mapping)
                    if label not in self.node_type_mapping:
                        self.node_type_mapping[label] = len(self.node_type_mapping)
                        
                    nodes.append(self.node_mapping[node_id])
                    node_types.append(self.node_type_mapping[label])

                    properties = []
                    for prop in mapping['numeric']:
                        value = record.get(prop, 0)
                        try:
                            properties.append(float(value) if value is not None else 0.0)
                        except (ValueError, TypeError):
                            properties.append(0.0)
                    # Handl categorical properties
                    for prop in mapping['categorical']:
                        value = record.get(prop, '')
                        if value is not None:
                            # Hashing categorical value and normalize to [0,1]
                            hash_val = hash(str(value)) % 1000000
                            properties.append(hash_val / 1000000.0)
                        else:
                            properties.append(0.0)
                    
                    max_props = max(
                        len(m['numeric']) + len(m['categorical']) 
                        for m in property_mappings.values()
                    )
                    properties.extend([0.0] * (max_props - len(properties)))
                    node_properties.append(properties)
                    
                offset += self.batch_size
                
        return nodes, node_types, node_properties 

    def extract_edges_batch(self, session, relationship_types: List[str]) -> Tuple[List[List[int]], List[int]]:
        edge_index = [[],[]]
        edge_types = []
        for rel_type in relationship_types:
            if rel_type not in self.relation_mapping:
                self.relation_mapping[rel_type] = len(self.relation_mapping)
            offset = 0
            while True:
                query = f"""
                MATCH (s)-[r:{rel_type}]->(t)
                RETURN elementId(s) as source, elementId(t) as target
                SKIP {offset} LIMIT {self.batch_size}
                """
                result = list(session.run(query))
                if not result:
                    break
                    
                for record in result:
                    source = self.node_mapping[record["source"]]
                    target = self.node_mapping[record["target"]]
                    edge_index[0].append(source)
                    edge_index[1].append(target)
                    edge_types.append(self.relation_mapping[rel_type])
                    
                offset += self.batch_size
        return edge_index, edge_types
    
    def find_unconnected_nodes(self, node_labels: list, constraints: dict, limit: int = 10):
        """
        Finds a list of distinct, unconnected node pairs based on their Neo4j IDs.
        Returns: List of tuples [(Neo4j ID A, Neo4j ID B). (..,..)]
        """
        label_list = str(node_labels)
    
        query = f"""
        MATCH (a)
        WHERE ANY(label IN labels(a) WHERE label IN {label_list})
        WITH a
        LIMIT 10 // Step 1: Force diversity by selecting a few distinct source nodes (a)

        MATCH (b)
        WHERE ANY(label IN labels(b) WHERE label IN {label_list})
        AND elementId(a) < elementId(b) // Optimization: avoid duplicate checks
        AND NOT (a)-[]-(b)

        // Retrieve Element IDs and all labels for Python-side filtering
        RETURN elementId(a) AS id_a, elementId(b) AS id_b, labels(a) AS labels_a, labels(b) AS labels_b
        LIMIT 100 // Fetch a larger number to ensure we get 10 candidates after filtering
        """

        all_candidates_raw = []

        try:
            with self.driver.session() as session:
                result = session.run(query)
                all_candidates_raw = list(result)
                
        except Exception as e:
            print(f"Error finding unconnected nodes: {e}")
            return []

        final_candidates = []
        
        for record in all_candidates_raw:
            src_label = record["labels_a"][0] 
            tgt_label = record["labels_b"][0]
            
            id_a = record["id_a"]
            id_b = record["id_b"]
            
            is_a_to_b_valid = src_label in constraints and tgt_label in constraints.get(src_label, [])
            
            is_b_to_a_valid = tgt_label in constraints and src_label in constraints.get(tgt_label, [])

            if is_a_to_b_valid:
                final_candidates.append((id_a, id_b))
            elif is_b_to_a_valid:
                final_candidates.append((id_b, id_a)) 
            if len(final_candidates) >= limit:
                break
                
        print(f"INFO: Cypher found {len(all_candidates_raw)} raw pairs. Filtered to {len(final_candidates)} valid pairs.")
        
        return final_candidates[:limit]
        
def main():
    NEO4J_URI = "bolt://149.165.175.36:7688"
    NEO4J_USERNAME = os.getenv("NEO4J_USER")
    NEO4J_PWD = os.getenv("NEO4J_PWD")

    graph_loader = GraphLoader(uri=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PWD)

    node_labels = ['Model', 'ModelCard', 'Device', 'Deployment', 'Server', 'Service', 'Datasheet', 'BiasAnalysis', 'ExplainabilityAnalysis', 'ModelRequirements']
    relationship_types = ['deployedIn', 'modelOf', 'requestedBy', 'used', 'BIAS_ANALYSIS','REQUIREMENTS','XAI_ANALYSIS','TRAINED_ON','USED']

    try:
        graph_data = graph_loader.extract_graph_data(node_labels, relationship_types)
        print(f"Nodes: {graph_data.x.shape}")
        print(f"Edges: {graph_data.edge_index.shape}")
        print(f"Edge Types: {graph_data.edge_type.shape}")
        print(f"Node Types: {graph_data.node_type.shape}")
        print(f"Node Properties: {graph_data.node_properties.shape}")

    finally:
        graph_loader.close()

if __name__ == "__main__":
    main()
