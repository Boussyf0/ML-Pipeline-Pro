#!/usr/bin/env python3
"""Load trained models into the API for serving."""
import os
import sys
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import requests
import time
import pickle
from pathlib import Path

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")

def get_latest_model_run():
    """Get the latest successful model run from MLflow."""
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name("customer-churn-prediction")
        if not experiment:
            print("❌ No experiment found named 'customer-churn-prediction'")
            return None
            
        # Search for runs with logged models
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=10
        )
        
        if runs.empty:
            print("❌ No finished runs found")
            return None
            
        print(f"✅ Found {len(runs)} finished runs")
        
        # Look for runs with good metrics
        for _, run in runs.iterrows():
            run_id = run['run_id']
            accuracy = run.get('metrics.accuracy', 0)
            auc_roc = run.get('metrics.auc_roc', 0)
            
            print(f"📊 Run {run_id[:8]}: accuracy={accuracy:.4f}, auc_roc={auc_roc:.4f}")
            
            # Check if this run has a good model (accuracy > 0.7 or auc_roc > 0.7)
            if accuracy > 0.7 or auc_roc > 0.7:
                return run_id, accuracy, auc_roc
                
        # If no run meets criteria, use the latest
        best_run = runs.iloc[0]
        run_id = best_run['run_id']
        accuracy = best_run.get('metrics.accuracy', 0)
        auc_roc = best_run.get('metrics.auc_roc', 0)
        
        print(f"⚠️ Using latest run (no runs met performance criteria)")
        return run_id, accuracy, auc_roc
        
    except Exception as e:
        print(f"❌ Error getting model run: {e}")
        return None

def load_model_from_mlflow(run_id):
    """Load model from MLflow."""
    try:
        # Try to load the model
        model_uri = f"runs:/{run_id}/model"
        print(f"🔄 Loading model from {model_uri}")
        
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"❌ Error loading model from MLflow: {e}")
        return None

def save_model_locally(model, model_info):
    """Save model locally for the API to load."""
    try:
        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Save the model
        model_path = models_dir / "churn_predictor.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Save model metadata
        metadata = {
            "model_name": "churn_predictor",
            "version": "1.0.0",
            "accuracy": model_info[1],
            "auc_roc": model_info[2],
            "run_id": model_info[0],
            "features": [
                "tenure", "monthly_charges", "total_charges"
            ]
        }
        
        metadata_path = models_dir / "model_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Metadata saved to {metadata_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving model locally: {e}")
        return False

def test_model_loading():
    """Test that the locally saved model can be loaded."""
    try:
        model_path = Path("models/churn_predictor.pkl")
        if not model_path.exists():
            print(f"❌ Model file not found: {model_path}")
            return False
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Test prediction with sample data
        test_data = [[12, 85.5, 1025.4]]  # tenure, monthly_charges, total_charges
        prediction = model.predict(test_data)
        probability = model.predict_proba(test_data)[0]
        
        print(f"✅ Model test successful:")
        print(f"   Prediction: {prediction[0]}")
        print(f"   Probabilities: {probability}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def restart_api():
    """Notify that API should be restarted to load new models."""
    print("\n🔄 To load the new model, you need to restart the FastAPI server.")
    print("   The API will automatically detect and load models from the models/ directory.")
    print("\n📋 Next steps:")
    print("   1. Stop the current API server (Ctrl+C)")
    print("   2. Restart with: source ml_pipeline_env/bin/activate && PYTHONPATH=/Users/abderrahim_boussyf/ML-Pipeline-Pro python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000")
    print("   3. Check health: curl http://localhost:8000/health")

def main():
    """Main function to deploy models."""
    print("🚀 MLOps Model Deployment")
    print("=" * 40)
    
    # Get latest model from MLflow
    print("\n1. Finding latest trained model...")
    model_info = get_latest_model_run()
    if not model_info:
        print("❌ No suitable model found in MLflow")
        sys.exit(1)
        
    run_id, accuracy, auc_roc = model_info
    print(f"✅ Selected model: run_id={run_id[:8]}, accuracy={accuracy:.4f}, auc_roc={auc_roc:.4f}")
    
    # Load model from MLflow
    print("\n2. Loading model from MLflow...")
    model = load_model_from_mlflow(run_id)
    if not model:
        print("❌ Failed to load model from MLflow")
        sys.exit(1)
        
    # Save model locally for API
    print("\n3. Saving model for API serving...")
    if not save_model_locally(model, model_info):
        print("❌ Failed to save model locally")
        sys.exit(1)
        
    # Test model loading
    print("\n4. Testing model loading...")
    if not test_model_loading():
        print("❌ Model loading test failed")
        sys.exit(1)
        
    print("\n✅ Model deployment completed successfully!")
    restart_api()

if __name__ == "__main__":
    main()