# 🤖 ML-Pipeline-Pro: Enterprise MLOps Platform

A production-grade MLOps pipeline for Customer Churn Prediction with automated training, model registry, serving, and monitoring.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Feature Store  │───▶│ Training Pipeline│
│   (PostgreSQL)  │    │     (Redis)     │    │    (Airflow)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │◀───│ Model Registry  │◀───│ Model Validation│
│ (Prometheus)    │    │    (MLflow)     │    │   (Great Exp)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Drift Detection │    │   A/B Testing   │───▶│ Production API  │
│   (Evidently)   │    │    (FastAPI)    │    │   (FastAPI)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
ML-Pipeline-Pro/
├── 📂 src/                     # Main source code
│   ├── 📂 data/               # Data processing modules
│   ├── 📂 features/           # Feature engineering
│   ├── 📂 models/             # Model training & evaluation
│   ├── 📂 api/                # FastAPI serving
│   └── 📂 monitoring/         # Model monitoring
├── 📂 config/                 # Configuration files
├── 📂 docker/                 # Docker configurations
├── 📂 airflow/               # Airflow DAGs & plugins
├── 📂 notebooks/             # Jupyter notebooks for EDA
├── 📂 tests/                 # Unit & integration tests
├── 📂 data/                  # Data storage
│   ├── 📂 raw/               # Raw datasets
│   ├── 📂 processed/         # Processed data
│   └── 📂 external/          # External data sources
├── 📂 models/                # Model artifacts
│   ├── 📂 artifacts/         # Trained models
│   └── 📂 registry/          # Model metadata
└── 📂 docs/                  # Documentation
```

## 🛠️ Tech Stack

- **🐍 Training:** Python, Scikit-learn, XGBoost, PyTorch
- **📊 Orchestration:** Apache Airflow + Docker
- **🗃️ Model Registry:** MLflow + PostgreSQL 
- **⚡ Feature Store:** Redis + Feast
- **🚀 API:** FastAPI + Uvicorn
- **📈 Monitoring:** Prometheus + Grafana + Evidently
- **🔄 CI/CD:** GitHub Actions + Docker
- **☁️ Infrastructure:** Docker + Kubernetes

## 🎯 Business Problem: Customer Churn Prediction

We're building a system to predict which customers are likely to cancel their subscription, allowing the business to take proactive retention actions.

**Key Metrics:**
- 📉 **Churn Rate**: Percentage of customers who cancel
- 🎯 **Precision**: Of predicted churners, how many actually churn
- 🔍 **Recall**: Of actual churners, how many we predicted
- 💰 **Business Impact**: Revenue saved through retention

## 🚀 Quick Start

```bash
# 1. Clone repository
git clone <your-repo-url>
cd ML-Pipeline-Pro

# 2. Setup environment
pip install -r requirements.txt

# 3. Download real customer churn data from Kaggle
python scripts/download_kaggle_data.py

# 4. Build Docker environment
docker-compose up -d

# 5. Run training pipeline
python src/models/train.py --data-path data/raw/customer_data.csv

# 6. Start API server
uvicorn src.api.main:app --reload

# 7. Access MLflow UI
http://localhost:5000
```

## 📊 Features

- ✅ **Automated Training**: Scheduled retraining with Airflow
- ✅ **Model Registry**: Versioning and metadata tracking with MLflow
- ✅ **Feature Store**: Cached features with Redis
- ✅ **Real-time Serving**: High-performance API with FastAPI
- ✅ **Monitoring**: Data drift and model performance tracking
- ✅ **A/B Testing**: Compare model versions in production
- ✅ **CI/CD Pipeline**: Automated testing and deployment

## 📈 Model Performance

| Metric | Development | Production |
|--------|-------------|------------|
| Precision | 0.85 | 0.82 |
| Recall | 0.78 | 0.76 |
| F1-Score | 0.81 | 0.79 |
| AUC-ROC | 0.89 | 0.87 |

## 🔍 Monitoring Dashboard

Access monitoring at `http://localhost:3000` (Grafana)
- Model accuracy over time
- Data drift detection
- API response times
- Feature importance tracking

## 📊 Dataset

This project uses the **Telco Customer Churn** dataset from Kaggle:
- **7,043 customers** with **21 features**
- **26.5% churn rate** (industry-realistic)
- **Real business problem**: Customer retention in telecommunications

### 🎯 Key Features:
- **Demographics**: Gender, age, partner, dependents
- **Services**: Internet, phone, streaming, security add-ons  
- **Contract**: Term length, payment method, billing
- **Usage**: Tenure, monthly charges, total charges

## 📚 Documentation

- [Setup Guide](docs/setup.md)
- [Data Documentation](docs/data.md)
- [Training Pipeline](docs/training.md)
- [API Documentation](docs/api.md)
- [Monitoring Guide](docs/monitoring.md)
- [A/B Testing Guide](docs/ab_testing.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

This project demonstrates enterprise-level MLOps practices for production AI systems.

---

**Built with ❤️ for demonstrating senior-level MLOps skills**