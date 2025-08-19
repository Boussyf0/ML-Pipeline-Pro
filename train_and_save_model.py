#!/usr/bin/env python3
"""Train and save model directly for API deployment."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
import json
from pathlib import Path
from datetime import datetime

def train_and_save_model():
    """Train a model and save it for API deployment."""
    print("🚀 Training Customer Churn Prediction Model")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    try:
        data = pd.read_csv("data/raw/customer_data.csv")
        print(f"✅ Loaded {len(data)} customer records")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
        
    # Prepare features and target
    X = data.drop(columns=['churn'])
    y = data['churn']
    
    # Convert target to numeric (Yes=1, No=0)
    y = (y == 'Yes').astype(int)
    churn_rate = y.mean()
    print(f"📉 Churn rate: {churn_rate:.1%}")
    
    # Use numeric columns for simplicity
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    X_numeric = X[numeric_columns]
    print(f"📋 Using {len(numeric_columns)} numeric features: {list(numeric_columns)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"🔄 Training set: {len(X_train)} samples")
    print(f"🔄 Test set: {len(X_test)} samples")
    
    # Train model
    print("\n🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("✅ Model training completed")
    
    # Evaluate model
    print("\n📈 Evaluating model performance...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    print("\n📊 Model Performance:")
    for metric, value in metrics.items():
        print(f"   {metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Feature importance
    feature_importance = dict(zip(numeric_columns, model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🔍 Top Features:")
    for feature, importance in top_features[:5]:
        print(f"   {feature}: {importance:.4f}")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "churn_predictor.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Save model metadata
    metadata = {
        "model_name": "churn_predictor",
        "model_type": "RandomForestClassifier",
        "version": "1.0.0",
        "trained_at": datetime.now().isoformat(),
        "dataset_size": len(data),
        "features": list(numeric_columns),
        "target_column": "churn",
        "churn_rate": churn_rate,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model_params": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "random_state": 42
        }
    }
    
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Metadata saved to: {metadata_path}")
    
    # Test model loading
    print("\n🧪 Testing model loading...")
    try:
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
            
        # Test prediction
        test_sample = X_test.iloc[0:1]
        prediction = loaded_model.predict(test_sample)[0]
        probability = loaded_model.predict_proba(test_sample)[0]
        
        print(f"✅ Model loading test successful")
        print(f"   Sample prediction: {prediction}")
        print(f"   Probabilities: [No Churn: {probability[0]:.3f}, Churn: {probability[1]:.3f}]")
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
        return False
    
    print(f"\n🎉 Model training and deployment preparation completed!")
    print(f"   📊 Model achieves {metrics['accuracy']:.1%} accuracy")
    print(f"   🎯 AUC-ROC score: {metrics['auc_roc']:.3f}")
    print(f"   💾 Model ready for API deployment")
    
    return True

if __name__ == "__main__":
    success = train_and_save_model()
    if success:
        print("\n📋 Next steps:")
        print("   1. Restart the FastAPI server to load the new model")
        print("   2. Check system health: curl http://localhost:8000/health")
        print("   3. The API will automatically detect and load the model")
    else:
        print("\n❌ Model training failed")