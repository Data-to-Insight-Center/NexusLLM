import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier


class LinkPrediction:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.node_properties = {}
        self.node_embeddings = {}
        self.scalers = {}
    
    def close(self):
        self.driver.close()

    def get_numeric_properties(self, node_labels: list) -> dict:
        with self.driver.session() as session:
            properties = {}
            for label in node_labels:
                query = f"""
                MATCH (n:{label})
                WHERE n IS NOT NULL
                WITH n LIMIT 1
                RETURN keys(n) as props, n as sample
                """
                result = session.run(query)
                record = result.single()
                
                if record:
                    node_data = record['sample']
                    all_props = record['props']
                    
                    numeric_props = []
                    for prop in all_props:
                        if prop in node_data:
                            value = node_data[prop]
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                numeric_props.append(prop)
                    
                    properties[label] = [p for p in numeric_props if not p.startswith('_')]
                else:
                    properties[label] = []
            return properties

    def get_node_features(self, label: str, properties: list) -> pd.DataFrame:
        with self.driver.session() as session:
            property_string = ", ".join([
                f"coalesce(n.{prop}, 0.0) as {prop}"        #missing values to 0
                for prop in properties
            ])
            query = f"""
            MATCH (n:{label})
            WHERE n IS NOT NULL
            RETURN elementId(n) as node_id, {property_string}
            """
            
            result = session.run(query)
            features = []
            for record in result:
                feature_dict = {'node_id': record['node_id']}
                for prop in properties:
                    feature_dict[prop] = record[prop]
                features.append(feature_dict)
            
            return pd.DataFrame(features)

    def prepare_node_features(self, node_labels: list):
        self.node_properties = self.get_numeric_properties(node_labels)
        print(f"Found numeric properties for nodes: {self.node_properties}")

        node_features = {}
        for label in node_labels:
            properties = self.node_properties[label]
            if properties:
                features_df = self.get_node_features(label, properties)
                
                if not features_df.empty:
                    scaler = StandardScaler()
                    feature_cols = [col for col in features_df.columns if col != 'node_id']
                    
                    if feature_cols: 
                        scaled_features = scaler.fit_transform(features_df[feature_cols])
                        self.scalers[label] = scaler
                        scaled_df = pd.DataFrame(
                            scaled_features, 
                            columns=feature_cols, 
                            index=features_df['node_id']
                        )
                        node_features[label] = scaled_df
                        print(f"Processed {len(feature_cols)} features for {label}")
                    else:
                        print(f"No feature columns found for {label}")
                else:
                    print(f"No features found for {label}")
            else:
                print(f"No properties defined for {label}")
                
        if not node_features:
            raise ValueError("No valid features could be created for any node type")
            
        return node_features

    def get_existing_relationships(self, relationship_types: list) -> list:
        with self.driver.session() as session:
            relationships = []
            for rel_type in relationship_types:
                query = f"""
                MATCH (a)-[r:{rel_type}]->(b)
                RETURN elementId(a) as source, elementId(b) as target, 
                       type(r) as relationship_type,
                       labels(a)[0] as source_label,
                       labels(b)[0] as target_label
                """
                result = session.run(query)
                for record in result:
                    relationships.append({
                        'source': record['source'],
                        'target': record['target'],
                        'type': record['relationship_type'],
                        'source_label': record['source_label'],
                        'target_label': record['target_label']
                    })
            print(f"Found {len(relationships)} existing relationships")
            return relationships

    def generate_negative_samples(self, node_labels: list, positive_edges: list, sample_ratio: float = 1.0) -> list:
        with self.driver.session() as session:
            node_ids = {}
            for label in node_labels:
                query = f"""
                MATCH (n:{label})
                RETURN collect({{
                    id: elementId(n),
                    label: labels(n)[0]
                }}) as nodes
                """
                result = session.run(query)
                nodes = result.single()['nodes']
                node_ids[label] = [(n['id'], n['label']) for n in nodes]

            existing_edges = {(edge['source'], edge['target']) for edge in positive_edges}
            existing_edges.update({(edge['target'], edge['source']) for edge in positive_edges})

            negative_samples = []
            num_samples = int(len(positive_edges) * sample_ratio)

            # negative samples based on relationships types
            for _ in range(num_samples):
                source_label = np.random.choice(node_labels)
                target_label = np.random.choice(node_labels)
                
                if node_ids[source_label] and node_ids[target_label]: 
                    source_idx = np.random.randint(0, len(node_ids[source_label]))
                    target_idx = np.random.randint(0, len(node_ids[target_label]))

                    source_id, source_node_label = node_ids[source_label][source_idx]
                    target_id, target_node_label = node_ids[target_label][target_idx]
                    
                    edge = (source_id, target_id)
                    if edge not in existing_edges and source_id != target_id:
                        negative_samples.append({
                            'source': source_id,
                            'target': target_id,
                            'exists': 0,
                            'source_label': source_label,
                            'target_label': target_label
                        })

            print(f"Generated {len(negative_samples)} negative samples")
            return negative_samples

    def pad_features(self, features: np.ndarray, max_length: int) -> np.ndarray:
        #padding vectors to avoid dimension mismatch
        if len(features) < max_length:
            padding = np.zeros(max_length - len(features))
            return np.concatenate([features, padding])
        return features

    def get_max_feature_length(self, node_features: dict) -> int:
        return max(df.shape[1] for df in node_features.values())

    def create_edge_features(self, edge: dict, node_features: dict) -> np.ndarray:
        try:
            source_id = edge['source']
            target_id = edge['target']
            source_label = edge['source_label']
            target_label = edge['target_label']
            
            if source_label not in node_features or target_label not in node_features:
                print(f"Missing features for label: {source_label} or {target_label}")
                return None
                
            try:
                source_features = node_features[source_label].loc[source_id].values
                target_features = node_features[target_label].loc[target_id].values
            except KeyError:
                print(f"Missing feature data for nodes: {source_id} or {target_id}")
                return None
                
            max_length = self.get_max_feature_length(node_features)
            source_features_padded = self.pad_features(source_features, max_length)
            target_features_padded = self.pad_features(target_features, max_length)
            
            node_types = list(node_features.keys())
            source_type_one_hot = np.zeros(len(node_types))
            target_type_one_hot = np.zeros(len(node_types))
            source_type_one_hot[node_types.index(source_label)] = 1
            target_type_one_hot[node_types.index(target_label)] = 1
            
            return np.concatenate([
                source_features_padded,
                target_features_padded,
                np.abs(source_features_padded - target_features_padded),
                source_features_padded * target_features_padded,
                source_type_one_hot,
                target_type_one_hot
            ])
        except Exception as e:
            print(f"Error creating edge features: {str(e)}")
            return None

    def train_link_prediction_model(self, node_labels: list, relationship_types: list):
        try:
            node_features = self.prepare_node_features(node_labels)
            if not node_features:
                raise ValueError("No node features could be extracted")
            
            positive_edges = self.get_existing_relationships(relationship_types)
            if not positive_edges:
                raise ValueError("No positive edges found")

            for edge in positive_edges:
                edge['exists'] = 1

            negative_edges = self.generate_negative_samples(node_labels, positive_edges)

            all_edges = positive_edges + negative_edges

            edge_features = []
            labels = []
            
            print("Creating edge features...")
            skipped_edges = 0
            for edge in all_edges:
                features = self.create_edge_features(edge, node_features)
                if features is not None:
                    edge_features.append(features)
                    labels.append(edge.get('exists', 0))
                else:
                    skipped_edges += 1

            if not edge_features:
                raise ValueError("No valid edge features could be created. Check if nodes have required properties.")

            print(f"Skipped {skipped_edges} edges due to missing features")
            
            X = np.array(edge_features)
            y = np.array(labels)

            print(f"Created features for {len(edge_features)} edges")
            print(f"Feature vector shape: {X.shape}")
            print(f"Number of positive samples: {sum(y)}")
            print(f"Number of negative samples: {len(y) - sum(y)}")
            return X, y
            
        except Exception as e:
            print(f"Error in train_link_prediction_model: {str(e)}")
            raise

    def predict_link_probability(self, source_id: int, target_id: int, 
                               node_features: dict, source_label: str, 
                               target_label: str) -> float:
        try:
            edge = {
                'source': source_id, 
                'target': target_id,
                'source_label': source_label,
                'target_label': target_label
            }
            features = self.create_edge_features(edge, node_features)
            
            if features is None:
                return 0.0
                
            similarity = cosine_similarity(features.reshape(1, -1), features.reshape(1, -1))[0][0]
            

            probability = (similarity + 1) / 2
            
            return probability
        except Exception as e:
            print(f"Error predicting link probability: {e}")
            return 0.0
        
    def predict_new_links(self, source_id: int, target_id: int, 
                     node_features: dict, source_label: str, 
                     target_label: str) -> dict:
        """Predict likelihood of a link between two nodes using trained model"""
        try:
            edge = {
                'source': source_id, 
                'target': target_id,
                'source_label': source_label,
                'target_label': target_label
            }
            features = self.create_edge_features(edge, node_features)
            
            if features is None:
                return {'probability': 0.0, 'prediction': False}
                
            if hasattr(self, 'model'):
                probability = self.model.predict_proba(features.reshape(1, -1))[0][1]
                prediction = self.model.predict(features.reshape(1, -1))[0]
                return {
                    'probability': float(probability),
                    'prediction': bool(prediction)
                }
            else:
                similarity = cosine_similarity(features.reshape(1, -1), features.reshape(1, -1))[0][0]
                probability = (similarity + 1) / 2
                return {
                    'probability': float(probability),
                    'prediction': probability > 0.5
                }
        except Exception as e:
            print(f"Error predicting link: {e}")
            return {'probability': 0.0, 'prediction': False}
