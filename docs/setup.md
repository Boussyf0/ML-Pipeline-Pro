# 🚀 Setup Guide

This guide will walk you through setting up the complete MLOps pipeline from scratch.

## Prerequisites

- **Docker & Docker Compose**: Version 20.10+
- **Python**: Version 3.9+
- **Git**: Latest version
- **Hardware**: Minimum 8GB RAM, 50GB disk space

## 🏗️ Infrastructure Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-org/ML-Pipeline-Pro.git
cd ML-Pipeline-Pro
```

### 2. Environment Configuration

Create environment variables:

```bash
# Copy example configuration
cp config/config.example.yaml config/config.yaml

# Set environment variables
export MLFLOW_TRACKING_URI=http://localhost:5000
export DATABASE_URL=postgresql://mlops_user:mlops_password@localhost:5432/mlops_db
export REDIS_URL=redis://localhost:6379
```

### 3. Docker Infrastructure

Start all services:

```bash
# Start core infrastructure
docker-compose up -d postgres redis mlflow

# Wait for services to be ready
./scripts/wait_for_services.sh

# Start remaining services
docker-compose up -d
```

Verify services are running:

```bash
# Check service status
docker-compose ps

# Test connectivity
curl http://localhost:5000/health  # MLflow
curl http://localhost:8000/health  # API
curl http://localhost:8080         # Airflow
curl http://localhost:3000         # Grafana
```

### 4. Database Initialization

Setup database schemas:

```bash
# Initialize MLOps database
python scripts/setup_model_registry.py

# Verify database setup
psql postgresql://mlops_user:mlops_password@localhost:5432/mlops_db -c "\dt"
```

## 📚 Component Setup

### MLflow Model Registry

```bash
# Setup MLflow with PostgreSQL backend
python scripts/setup_model_registry.py

# Verify MLflow is working
python -c "import mlflow; print('MLflow version:', mlflow.__version__)"

# Access MLflow UI
open http://localhost:5000
```

### Apache Airflow

```bash
# Initialize Airflow database
docker-compose exec airflow-webserver airflow db init

# Create admin user
docker-compose exec airflow-webserver airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Access Airflow UI
open http://localhost:8080
```

### Monitoring Stack

```bash
# Import Grafana dashboards
./scripts/import_grafana_dashboards.sh

# Configure Prometheus targets
docker-compose restart prometheus

# Access monitoring
open http://localhost:3000  # Grafana (admin/admin)
open http://localhost:9090  # Prometheus
```

## 🧪 Testing Setup

### Install Development Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run Tests

```bash
# Unit tests
pytest tests/test_models.py -v

# Integration tests
pytest tests/integration/ -v

# Coverage report
pytest --cov=src tests/
```

### Sample Data

```bash
# Generate sample data
python scripts/generate_sample_data.py

# Or download real dataset
wget https://example.com/customer_churn.csv -O data/raw/customer_data.csv
```

## 🎯 First Model Training

### Manual Training

```bash
# Train your first model
python src/models/train.py \
    --data-path data/raw/customer_data.csv \
    --config-path config/config.yaml \
    --register-best

# Check MLflow for results
open http://localhost:5000
```

### Automated Training (Airflow)

```bash
# Enable DAG in Airflow UI
# Or via CLI:
docker-compose exec airflow-webserver airflow dags unpause mlops_training_pipeline

# Trigger manual run
docker-compose exec airflow-webserver airflow dags trigger mlops_training_pipeline
```

## 🚀 API Deployment

### Local Development

```bash
# Start API server
python scripts/start_api.py start --reload

# Test API
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -H "X-API-Key: admin_key_123" \
    -d '{
        "features": {
            "tenure": 12,
            "monthly_charges": 65.0,
            "contract": "Month-to-month",
            "payment_method": "Electronic check"
        }
    }'
```

### Production Deployment

```bash
# Build production image
docker build -f docker/api.Dockerfile -t mlops-api:latest .

# Deploy to Kubernetes
kubectl apply -f k8s/production/

# Check deployment
kubectl get pods -n production
```

## 📊 A/B Testing Setup

### Create First Experiment

```bash
# Create A/B test
python scripts/ab_test_cli.py create \
    --name "Model Comparison" \
    --model-a churn-predictor \
    --model-a-version 1.0.0 \
    --model-b churn-predictor \
    --model-b-version 1.1.0 \
    --traffic-split 0.5

# Start experiment
python scripts/ab_test_cli.py start <experiment-id>
```

### Monitor Experiment

```bash
# Check status
python scripts/ab_test_cli.py status <experiment-id>

# Analyze results
python scripts/ab_test_cli.py analyze <experiment-id>
```

## 🔍 Monitoring Setup

### Configure Alerts

```bash
# Setup monitoring alerts
python scripts/setup_monitoring.py --environment production

# Test alerting
python scripts/test_alerts.py
```

### Dashboard Access

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5000
- **Airflow**: http://localhost:8080 (admin/admin)
- **API Docs**: http://localhost:8000/docs

## 🔧 Configuration

### Environment Variables

```bash
# Required environment variables
export MLFLOW_TRACKING_URI=http://localhost:5000
export DATABASE_URL=postgresql://mlops_user:mlops_password@localhost:5432/mlops_db
export REDIS_URL=redis://localhost:6379
export API_KEY=your-secure-api-key
export ENVIRONMENT=development
```

### Configuration Files

```yaml
# config/config.yaml
project:
  name: "customer-churn-prediction"
  version: "1.0.0"

database:
  host: "localhost"
  port: 5432
  name: "mlops_db"

training:
  data:
    train_ratio: 0.7
    validation_ratio: 0.15
    test_ratio: 0.15
  
  models:
    xgboost:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
```

## ✅ Verification Checklist

- [ ] All Docker services running
- [ ] Database schemas created
- [ ] MLflow UI accessible
- [ ] Airflow UI accessible
- [ ] API responding to health checks
- [ ] First model trained successfully
- [ ] Monitoring dashboards visible
- [ ] A/B testing framework working
- [ ] Tests passing

## 🚨 Troubleshooting

### Common Issues

**MLflow not starting:**
```bash
# Check logs
docker-compose logs mlflow

# Restart service
docker-compose restart mlflow
```

**Database connection issues:**
```bash
# Test database connectivity
docker-compose exec postgres psql -U mlops_user -d mlops_db -c "SELECT 1;"
```

**API authentication errors:**
```bash
# Verify API key
curl -H "X-API-Key: admin_key_123" http://localhost:8000/health
```

**Training pipeline fails:**
```bash
# Check Airflow logs
docker-compose exec airflow-webserver airflow tasks logs mlops_training_pipeline train_models
```

### Getting Help

1. Check service logs: `docker-compose logs <service-name>`
2. Verify configuration: `python scripts/validate_config.py`
3. Run health checks: `python scripts/health_check.py`
4. Review documentation: Navigate to relevant docs section

## 📝 Next Steps

After successful setup:

1. **[Training Pipeline](training.md)**: Learn about automated model training
2. **[API Usage](api.md)**: Understand the prediction API
3. **[Monitoring](monitoring.md)**: Set up comprehensive monitoring
4. **[Deployment](deployment.md)**: Deploy to production environments

---

**Need help?** Check the [troubleshooting section](#troubleshooting) or open an issue.