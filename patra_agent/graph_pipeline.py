import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from graph_link_prediction import LinkPrediction

def train_and_evaluate_model(predictor, node_labels, relationship_types):
    print("\n1. Training link prediction model...")
    X, y = predictor.train_link_prediction_model(node_labels, relationship_types)

    print("\n2. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n3. Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )

    print("\n4. Performing cross-validation...")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Average CV score:", cv_scores.mean())

    print("\n5. Training final model...")
    rf_model.fit(X_train, y_train)

    print("\n6. Making predictions...")
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    return rf_model, X, y_pred, y_pred_proba, y_test

def evaluate_model_performance(y_test, y_pred, y_pred_proba, X):
    """Evaluate model performance and print metrics"""
    print("\n=== Model Evaluation ===")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\nROC AUC Score: {roc_auc:.3f}")
    print(f"Precision-Recall AUC: {pr_auc:.3f}")

    return precision, recall, pr_auc

def analyze_feature_importance(rf_model, X):
    """Analyze and print feature importance"""
    print("\n=== Feature Importance Analysis ===")
    feature_length = X.shape[1] // 6
    feature_importances = rf_model.feature_importances_
    
    importance_sections = [
        "Source Node Features",
        "Target Node Features",
        "Feature Differences",
        "Feature Products",
        "Source Node Type",
        "Target Node Type"
    ]
    
    for i, section in enumerate(importance_sections):
        start_idx = i * feature_length
        end_idx = (i + 1) * feature_length
        section_importance = feature_importances[start_idx:end_idx].mean()
        print(f"{section}: {section_importance:.3f}")

def plot_precision_recall_curve(recall, precision, pr_auc):
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def example_prediction(predictor, node_features):
    result = predictor.predict_new_links(
        source_id=123,  
        target_id=456, 
        node_features=node_features,
        source_label="Model",
        target_label="EdgeDevice"
    )
    
    print(f"Link prediction result: {result}")
    print(f"Probability of link: {result['probability']:.3f}")
    print(f"Predicted link exists: {result['prediction']}")


def make_link_prediction(predictor, node_labels, source_label, target_label, source_id, target_id):
    print("\n=== Making Prediction ===")
    with predictor.driver.session() as session:
        query = """
        MATCH (source:{source_label}) WHERE elementId(source) = $source_id
        MATCH (target:{target_label}) WHERE elementId(target) = $target_id
        RETURN elementId(source) as source_id, elementId(target) as target_id
        """.format(source_label=source_label, target_label=target_label)
        
        result = session.run(query, {"source_id": source_id, "target_id": target_id})
        record = result.single()
        print("record: -- ", record)
        if record:
            source_id = record['source_id']
            target_id = record['target_id']

            result = predictor.predict_new_links(
                source_id=source_id,
                target_id=target_id,
                node_features=predictor.prepare_node_features(node_labels),
                source_label="Deployment",
                target_label="EdgeDevice"
            )
            
            print(f"\nPredicting link between {source_label} ({source_id}) and {target_label} ({target_id}): ")
            print(f"Probability of link: {result['probability']:.3f}")
            print(f"Predicted link exists: {result['prediction']}")

def main():

    load_dotenv()
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USERNAME = os.getenv("NEO4J_USER")
    NEO4J_PWD = os.getenv("NEO4J_PWD")

    predictor = LinkPrediction(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PWD
    )

    try:
        node_labels = ["Model", "EdgeDevice", "Deployment"]
        relationship_types = ["On_Device", "HAS_DEPLOYMENT"]
        
        print("=== Starting Link Prediction Pipeline ===")

        rf_model, X, y_pred, y_pred_proba, y_test = train_and_evaluate_model(
            predictor, node_labels, relationship_types
        )


        precision, recall, pr_auc = evaluate_model_performance(
            y_test, y_pred, y_pred_proba, X
        )

        analyze_feature_importance(rf_model, X)
        plot_precision_recall_curve(recall, precision, pr_auc)
        predictor.model = rf_model

        source_label = "Deployment"       
        target_label = "EdgeDevice"   
        source_id = "4:b6ae30eb-5fdd-4c39-b281-fa2550f0ea84:14"  
        target_id = "4:b6ae30eb-5fdd-4c39-b281-fa2550f0ea84:0"  

        make_link_prediction(predictor, node_labels, source_label, target_label, source_id, target_id)

    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        raise e
    finally:
        print("\n=== Closing Connection ===")
        predictor.close()

if __name__ == "__main__":
    main()