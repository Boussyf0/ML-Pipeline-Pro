"""Simple training script to test MLflow integration."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("customer-churn-prediction")

# Load data
print("Loading data...")
data = pd.read_csv("data/raw/customer_data.csv")
X = data.drop(columns=['churn'])
y = data['churn']

# Convert target to numeric (Yes=1, No=0)
y = (y == 'Yes').astype(int)
print(f"Target distribution: {y.value_counts().to_dict()}")

# Simple preprocessing - just use numeric columns
numeric_columns = X.select_dtypes(include=[np.number]).columns
X_numeric = X[numeric_columns]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

print(f"Training with {len(X_train)} samples, {len(numeric_columns)} features")

# Start MLflow run
with mlflow.start_run(run_name="simple_rf_training") as run:
    # Train model
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("features", len(numeric_columns))
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("auc_roc", auc)
    
    # Log model (using older API that should work)
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="simple-churn-predictor"
        )
        print("✅ Model logged successfully!")
    except Exception as e:
        print(f"⚠️ Model logging failed: {e}")
        # Try simpler model logging
        mlflow.sklearn.log_model(model, "model")
        print("✅ Model logged with simple method!")
    
    print(f"🏃 Run ID: {run.info.run_id}")
    print(f"🌐 View at: http://localhost:5001/#/experiments/1/runs/{run.info.run_id}")

print("Training completed!")