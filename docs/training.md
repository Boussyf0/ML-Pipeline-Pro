# 🎓 Training Pipeline Guide

This guide covers the complete model training pipeline, from data preprocessing to model deployment.

## 🏗️ Pipeline Architecture

```
Data Sources → Data Validation → Feature Engineering → Model Training → Model Validation → Registry → Deployment
     ↓              ↓                  ↓                ↓                ↓            ↓           ↓
 PostgreSQL    Great Expectations   Custom Features   XGBoost/LightGBM   Performance   MLflow    Production
   Redis         Data Profiling      Engineering       Random Forest      Testing              API/A/B Test
```

## 📊 Data Pipeline

### Data Ingestion

```python
# Load raw data
from src.data.preprocessor import DataPreprocessor

preprocessor = DataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(
    'data/raw/customer_data.csv'
)
```

### Data Validation

The pipeline uses **Great Expectations** for data quality validation:

```python
# Automated data validation
validation_results = preprocessor.validate_data(df)

# Example validations:
# - No missing values in critical columns
# - Data types match expected schema
# - Numerical features within expected ranges
# - Categorical features have valid values
# - Target distribution is reasonable
```

**Data Quality Checks:**
- ✅ **Completeness**: No missing values in key columns
- ✅ **Validity**: Data types and value ranges
- ✅ **Consistency**: Relationships between features
- ✅ **Uniqueness**: No duplicate records
- ✅ **Freshness**: Data recency requirements

### Data Drift Detection

```python
# Monitor data drift between training and production
from src.monitoring.drift_detector import DriftDetector

drift_detector = DriftDetector()
drift_results = drift_detector.detect_drift(reference_data, current_data)

# Drift detection methods:
# - Kolmogorov-Smirnov test
# - Population Stability Index (PSI)
# - Wasserstein distance
# - Jensen-Shannon divergence
```

## 🔧 Feature Engineering

### Automated Feature Creation

```python
def feature_engineering(df):
    # Tenure-based features
    df['tenure_category'] = pd.cut(df['tenure'], 
                                   bins=[0, 12, 24, 48, float('inf')],
                                   labels=['New', 'Medium', 'Long', 'Very_Long'])
    
    # Spending patterns
    df['total_amount'] = df['tenure'] * df['monthly_charges']
    df['avg_monthly_charges'] = df['monthly_charges'] / (df['tenure'] + 1)
    
    # Contract risk factors
    df['is_high_risk'] = (
        (df['contract'] == 'Month-to-month') & 
        (df['monthly_charges'] > df['monthly_charges'].median())
    ).astype(int)
    
    return df
```

### Feature Store Integration

Features are cached in Redis for fast access:

```python
# Cache features for real-time serving
feature_store = FeatureStore()
feature_store.store_features(customer_id, features)

# Retrieve during prediction
features = feature_store.get_features(customer_id)
```

## 🤖 Model Training

### Supported Algorithms

```yaml
# config/config.yaml - Model configurations
training:
  models:
    xgboost:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
      subsample: 0.8
      colsample_bytree: 0.8
      
    lightgbm:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
      feature_fraction: 0.8
      bagging_fraction: 0.8
      
    random_forest:
      n_estimators: 100
      max_depth: 10
      min_samples_split: 5
      min_samples_leaf: 2
```

### Hyperparameter Optimization

```python
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()

# Automated hyperparameter tuning with Optuna
if config["training"]["hyperparameter_tuning"]["enabled"]:
    best_params = trainer.optimize_hyperparameters(
        model_name="xgboost",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val
    )
```

**Optimization Strategy:**
- **Search Algorithm**: TPE (Tree-structured Parzen Estimator)
- **Objective**: ROC-AUC score on validation set
- **Search Space**: Defined per algorithm
- **Early Stopping**: Prevent overfitting
- **Cross-Validation**: 5-fold stratified CV

### Training Execution

```bash
# Manual training
python src/models/train.py \
    --data-path data/raw/customer_data.csv \
    --config-path config/config.yaml \
    --register-best \
    --promote-to-staging

# Programmatic training
from src.models.trainer import ModelTrainer

trainer = ModelTrainer()
results = trainer.train_models('data/raw/customer_data.csv')
```

## 📏 Model Evaluation

### Performance Metrics

```python
# Comprehensive evaluation metrics
metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1_score': f1_score(y_true, y_pred),
    'auc_roc': roc_auc_score(y_true, y_proba),
    'auc_pr': average_precision_score(y_true, y_proba)
}
```

### Cross-Validation

```python
# Stratified K-Fold cross-validation
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')

print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
```

### Performance Thresholds

```yaml
# Minimum performance requirements
training:
  thresholds:
    min_accuracy: 0.75
    min_precision: 0.70
    min_recall: 0.65
    min_f1_score: 0.70
    min_auc_roc: 0.75
```

### Model Interpretability

```python
# SHAP values for model interpretability
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
```

## 📈 MLflow Integration

### Experiment Tracking

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run(run_name="churn_model_training"):
    # Log parameters
    mlflow.log_params({
        "algorithm": "xgboost",
        "n_estimators": 100,
        "max_depth": 6
    })
    
    # Log metrics
    mlflow.log_metrics({
        "accuracy": accuracy,
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall
    })
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    mlflow.log_artifact("feature_importance.png")
    mlflow.log_artifact("confusion_matrix.png")
```

### Model Registry

```python
# Register best model
from src.models.registry import ModelRegistry

registry = ModelRegistry()

# Register model version
model_version = registry.register_model(
    run_id=best_run_id,
    model_path="model",
    description="Customer churn prediction model - XGBoost"
)

# Promote to staging
registry.promote_model(
    model_name="churn-predictor",
    version=model_version,
    stage="Staging"
)
```

## 🔄 Automated Pipeline (Airflow)

### DAG Configuration

```python
# airflow/dags/training_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'mlops_training_pipeline',
    default_args=default_args,
    description='MLOps training pipeline',
    schedule_interval='0 2 * * 0',  # Weekly at 2 AM Sunday
    catchup=False
)
```

### Pipeline Tasks

1. **Data Validation** → Validate input data quality
2. **Drift Detection** → Check for data drift
3. **Feature Engineering** → Create and transform features
4. **Model Training** → Train multiple algorithms
5. **Model Validation** → Evaluate performance
6. **Model Registration** → Register in MLflow
7. **Model Testing** → Run automated tests
8. **Deployment** → Deploy to staging

### Monitoring & Alerts

```python
# Email notification on success/failure
success_email = EmailOperator(
    task_id='send_success_email',
    to=['mlops-team@company.com'],
    subject='Training Pipeline - Success',
    html_content="Training completed successfully. Check MLflow for results."
)
```

## 🧪 Model Validation

### Automated Testing

```python
# tests/test_model_performance.py
def test_model_meets_performance_thresholds():
    """Test that model meets minimum performance requirements."""
    model_metrics = get_latest_model_metrics()
    
    assert model_metrics['accuracy'] >= 0.75
    assert model_metrics['precision'] >= 0.70
    assert model_metrics['recall'] >= 0.65
    assert model_metrics['auc_roc'] >= 0.75

def test_model_prediction_format():
    """Test that model predictions are properly formatted."""
    predictions = model.predict(sample_data)
    
    assert all(isinstance(p, (int, float)) for p in predictions)
    assert all(0 <= p <= 1 for p in predictions)  # For probabilities
```

### Production Readiness

```bash
# Run validation script
python scripts/validate_model.py \
    --model-name churn-predictor \
    --min-accuracy 0.75 \
    --output-file validation_results.json
```

**Validation Checklist:**
- [ ] Performance meets thresholds
- [ ] Model artifacts exist and loadable
- [ ] Prediction format is correct
- [ ] Model size is acceptable
- [ ] Inference time is within limits
- [ ] Memory usage is reasonable

## 📊 Training Monitoring

### Real-time Metrics

```python
# Track training progress
from src.monitoring.model_monitor import ModelMonitor

monitor = ModelMonitor()

# Log training metrics
monitor.log_training_metrics(
    model_name="churn-predictor",
    epoch=epoch,
    train_loss=train_loss,
    val_loss=val_loss,
    accuracy=accuracy
)
```

### Training Dashboard

Access training metrics at:
- **MLflow UI**: http://localhost:5000
- **Grafana Dashboard**: http://localhost:3000

### Performance Tracking

```sql
-- Query training history
SELECT 
    experiment_name,
    run_id,
    metric_name,
    metric_value,
    timestamp
FROM mlflow_metrics 
WHERE experiment_name = 'customer-churn-prediction'
ORDER BY timestamp DESC;
```

## 🚀 Deployment Pipeline

### Staging Deployment

```python
# Automatic deployment to staging after successful training
if model_meets_thresholds and passes_validation:
    deploy_to_staging(model_name, model_version)
    setup_canary_deployment(traffic_percentage=10)
```

### A/B Testing

```python
# Setup A/B test with new model
from src.ab_testing.experiment_manager import ExperimentManager

experiment_manager = ExperimentManager()

experiment_id = experiment_manager.create_experiment(
    name="Model v2.0 vs v1.5",
    model_a="churn-predictor:1.5.0",
    model_b="churn-predictor:2.0.0",
    traffic_split=0.5,
    duration_days=14
)
```

## 📝 Best Practices

### Code Organization

```
src/models/
├── __init__.py
├── trainer.py          # Main training logic
├── registry.py         # MLflow integration
├── preprocessor.py     # Data preprocessing
├── validators.py       # Model validation
└── utils.py           # Utility functions
```

### Configuration Management

```yaml
# Separate configs for different environments
config/
├── config.yaml         # Default configuration
├── development.yaml    # Development overrides
├── staging.yaml       # Staging overrides
└── production.yaml    # Production overrides
```

### Error Handling

```python
try:
    results = trainer.train_models(data_path)
except DataValidationError as e:
    logger.error(f"Data validation failed: {e}")
    send_alert("Data quality issue detected")
    raise
except ModelTrainingError as e:
    logger.error(f"Model training failed: {e}")
    fallback_to_previous_model()
    raise
```

### Logging

```python
import logging

# Structured logging
logger.info(
    "Model training completed",
    extra={
        "model_name": "churn-predictor",
        "accuracy": 0.85,
        "training_time_minutes": 45,
        "data_size": 100000
    }
)
```

## 🔍 Troubleshooting

### Common Issues

**Training fails with OOM error:**
```bash
# Reduce batch size or use smaller dataset
python src/models/train.py --batch-size 512 --sample-ratio 0.1
```

**Poor model performance:**
```python
# Check data quality and feature importance
validation_results = preprocessor.validate_data(df)
feature_importance = model.feature_importances_
```

**MLflow connection issues:**
```bash
# Verify MLflow server is running
curl http://localhost:5000/health

# Check environment variables
echo $MLFLOW_TRACKING_URI
```

### Performance Optimization

```python
# Use parallel processing for hyperparameter tuning
import joblib

# Parallel model training
with joblib.parallel_backend('threading', n_jobs=-1):
    cv_scores = cross_val_score(model, X, y, cv=5)

# GPU acceleration for deep learning models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

## 📚 Next Steps

After mastering the training pipeline:

1. **[API Documentation](api.md)**: Learn about model serving
2. **[Monitoring Guide](monitoring.md)**: Set up comprehensive monitoring
3. **[A/B Testing](ab_testing.md)**: Compare model versions
4. **[Production Deployment](deployment.md)**: Deploy to production

---

**Need help?** Check the [troubleshooting section](#troubleshooting) or review the training logs in MLflow.