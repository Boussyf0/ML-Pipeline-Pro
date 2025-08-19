# 📊 Monitoring Guide

Comprehensive monitoring and observability for the MLOps pipeline with Prometheus, Grafana, and custom monitoring components.

## 🏗️ Monitoring Architecture

```
Data Sources → Metrics Collection → Storage → Visualization → Alerting
     ↓              ↓               ↓         ↓            ↓
 API/Models    Prometheus Agent   TimeSeries  Grafana    PagerDuty
 Databases     Custom Collectors   Database    Dashboards  Slack
 Airflow       OpenTelemetry      Prometheus   Custom UI   Email
```

## 📈 Monitoring Stack

### Core Components

- **📊 Metrics**: Prometheus + custom collectors
- **📉 Visualization**: Grafana dashboards
- **🚨 Alerting**: Prometheus Alertmanager
- **📋 Logs**: Structured logging with JSON format
- **🔍 Tracing**: OpenTelemetry for request tracing
- **🎯 Data Quality**: Great Expectations + Evidently

## 🎯 Key Metrics

### Model Performance Metrics

```python
# Model accuracy over time
model_accuracy_score{model="churn-predictor", version="1.2.0"} 0.85

# Prediction latency
prediction_duration_seconds{model="churn-predictor"} 0.045

# Prediction volume
prediction_requests_total{model="churn-predictor", status="success"} 15420

# Data drift score
data_drift_score{model="churn-predictor", feature="tenure"} 0.02
```

### Infrastructure Metrics

```python
# API response times
http_request_duration_seconds{method="POST", endpoint="/predict"} 0.12

# Database connections
database_connections_active{database="mlops_db"} 15

# Redis cache hit rate
redis_cache_hits_total{operation="get"} 8924
redis_cache_misses_total{operation="get"} 1076

# MLflow server health
mlflow_server_up{instance="mlflow:5000"} 1
```

### Business Metrics

```python
# Model prediction distribution
model_predictions_by_class{model="churn-predictor", class="0"} 12336
model_predictions_by_class{model="churn-predictor", class="1"} 3084

# A/B test metrics
ab_test_conversions{experiment="model_v2", variant="A"} 245
ab_test_conversions{experiment="model_v2", variant="B"} 267
```

## 🔍 Data Drift Detection

### Automated Drift Monitoring

```python
# src/monitoring/drift_monitor.py
from src.monitoring.drift_detector import DriftDetector
import pandas as pd

def monitor_data_drift():
    """Continuous data drift monitoring."""
    detector = DriftDetector()
    
    # Get reference data (training data)
    reference_data = load_training_data()
    
    # Get current production data
    current_data = load_recent_predictions_data(hours=24)
    
    # Detect drift for each feature
    drift_results = detector.detect_drift(reference_data, current_data)
    
    for feature, result in drift_results.items():
        if result['drift_detected']:
            logger.warning(f"Drift detected in feature {feature}: {result}")
            
            # Send alert
            send_drift_alert(feature, result)
            
            # Update monitoring metrics
            drift_score_gauge.labels(
                model="churn-predictor", 
                feature=feature
            ).set(result['drift_score'])
```

### Drift Detection Methods

```python
# Kolmogorov-Smirnov Test
ks_statistic = stats.ks_2samp(reference_feature, current_feature)

# Population Stability Index (PSI)
psi_score = calculate_psi(reference_feature, current_feature)

# Wasserstein Distance
wasserstein_dist = stats.wasserstein_distance(reference_feature, current_feature)

# Jensen-Shannon Divergence
js_divergence = calculate_js_divergence(reference_feature, current_feature)
```

### Drift Thresholds

```yaml
# config/monitoring.yaml
drift_detection:
  enabled: true
  check_interval_minutes: 60
  
  thresholds:
    ks_test: 0.05      # p-value threshold
    psi_score: 0.1     # PSI threshold
    wasserstein: 0.2   # Wasserstein distance threshold
    js_divergence: 0.1 # JS divergence threshold
    
  features:
    tenure: 
      enabled: true
      method: "ks_test"
    monthly_charges:
      enabled: true
      method: "wasserstein"
    contract:
      enabled: true  
      method: "psi"
```

## 📊 Grafana Dashboards

### MLOps Overview Dashboard

```json
{
  "dashboard": {
    "title": "MLOps Pipeline Overview",
    "panels": [
      {
        "title": "Model Performance",
        "targets": [
          "model_accuracy_score{model=\"churn-predictor\"}"
        ]
      },
      {
        "title": "Prediction Volume", 
        "targets": [
          "rate(prediction_requests_total[5m])"
        ]
      },
      {
        "title": "API Response Times",
        "targets": [
          "histogram_quantile(0.95, http_request_duration_seconds)"
        ]
      }
    ]
  }
}
```

### Data Quality Dashboard

Key panels:
- **Data Freshness**: Time since last data update
- **Missing Values**: Percentage of missing values per feature
- **Feature Distributions**: Histograms of feature distributions over time
- **Data Drift Scores**: Drift scores for all monitored features
- **Data Volume**: Number of records processed over time

### Model Performance Dashboard

Key panels:
- **Accuracy Trends**: Model accuracy over time
- **Precision/Recall**: Classification metrics trends
- **Prediction Distribution**: Distribution of model predictions
- **Feature Importance**: Top contributing features
- **Model Comparison**: A/B test performance comparison

### Infrastructure Dashboard

Key panels:
- **Service Health**: Status of all services (API, MLflow, DB, Redis)
- **Resource Usage**: CPU, memory, disk usage
- **API Metrics**: Request rates, error rates, latency
- **Database Performance**: Connection pool, query performance
- **Cache Performance**: Redis hit rates, memory usage

## 🚨 Alerting Rules

### Critical Alerts

```yaml
# config/alerts.yaml
groups:
- name: model_performance
  rules:
  - alert: ModelAccuracyDrop
    expr: model_accuracy_score < 0.75
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Model accuracy dropped below threshold"
      description: "Model {{ $labels.model }} accuracy is {{ $value }}"

  - alert: HighPredictionLatency
    expr: histogram_quantile(0.95, prediction_duration_seconds) > 1.0
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High prediction latency detected"
      
  - alert: DataDriftDetected
    expr: data_drift_score > 0.1
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Data drift detected in feature {{ $labels.feature }}"

- name: infrastructure
  rules:
  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service {{ $labels.instance }} is down"
      
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
```

### Alert Destinations

```yaml
# config/alertmanager.yaml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default-receiver'
  routes:
  - match:
      severity: critical
    receiver: 'pagerduty-critical'
  - match:
      severity: warning
    receiver: 'slack-warnings'

receivers:
- name: 'default-receiver'
  email_configs:
  - to: 'mlops-team@company.com'
    subject: 'MLOps Alert: {{ .GroupLabels.alertname }}'
    
- name: 'pagerduty-critical'
  pagerduty_configs:
  - service_key: 'your-pagerduty-service-key'
    
- name: 'slack-warnings'
  slack_configs:
  - api_url: 'https://hooks.slack.com/services/...'
    channel: '#mlops-alerts'
```

## 📋 Custom Monitoring Components

### Model Monitor Class

```python
# src/monitoring/model_monitor.py
import time
import logging
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge
import mlflow

class ModelMonitor:
    """Custom model monitoring with metrics collection."""
    
    def __init__(self):
        self.prediction_counter = Counter(
            'model_predictions_total',
            'Total model predictions',
            ['model', 'version', 'status']
        )
        
        self.prediction_latency = Histogram(
            'prediction_duration_seconds',
            'Model prediction duration',
            ['model', 'version']
        )
        
        self.accuracy_gauge = Gauge(
            'model_accuracy_score',
            'Model accuracy score',
            ['model', 'version']
        )
        
        self.drift_gauge = Gauge(
            'data_drift_score', 
            'Data drift score',
            ['model', 'feature']
        )
    
    def log_prediction(self, model_name: str, model_version: str, 
                      latency: float, success: bool):
        """Log prediction metrics."""
        status = "success" if success else "error"
        
        self.prediction_counter.labels(
            model=model_name,
            version=model_version, 
            status=status
        ).inc()
        
        if success:
            self.prediction_latency.labels(
                model=model_name,
                version=model_version
            ).observe(latency)
    
    def update_model_metrics(self, model_name: str, model_version: str,
                           metrics: Dict[str, float]):
        """Update model performance metrics."""
        if 'accuracy' in metrics:
            self.accuracy_gauge.labels(
                model=model_name,
                version=model_version
            ).set(metrics['accuracy'])
            
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
    
    def log_drift_score(self, model_name: str, feature: str, 
                       drift_score: float):
        """Log data drift score."""
        self.drift_gauge.labels(
            model=model_name,
            feature=feature
        ).set(drift_score)
```

### Performance Tracker

```python
# src/monitoring/performance_tracker.py
import pandas as pd
from typing import List, Dict
import numpy as np

class PerformanceTracker:
    """Track model performance over time."""
    
    def __init__(self, database_url: str):
        self.db_url = database_url
        
    def log_prediction_result(self, prediction_id: str, model_name: str,
                            model_version: str, features: Dict,
                            prediction: int, probability: float,
                            actual_outcome: int = None):
        """Log prediction and actual outcome."""
        record = {
            'prediction_id': prediction_id,
            'model_name': model_name,
            'model_version': model_version,
            'features': features,
            'prediction': prediction,
            'probability': probability,
            'actual_outcome': actual_outcome,
            'timestamp': pd.Timestamp.now()
        }
        
        # Save to database
        self._save_prediction_record(record)
    
    def calculate_performance_metrics(self, model_name: str,
                                    time_window_hours: int = 24) -> Dict[str, float]:
        """Calculate performance metrics for recent predictions."""
        # Get predictions with actual outcomes from last N hours
        predictions_df = self._get_recent_predictions_with_outcomes(
            model_name, time_window_hours
        )
        
        if len(predictions_df) < 10:  # Insufficient data
            return {}
            
        y_true = predictions_df['actual_outcome']
        y_pred = predictions_df['prediction']
        y_prob = predictions_df['probability']
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'), 
            'auc_roc': roc_auc_score(y_true, y_prob),
            'sample_size': len(predictions_df)
        }
```

## 🔧 Monitoring Setup

### Prometheus Configuration

```yaml
# config/prometheus.yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yaml"

scrape_configs:
  - job_name: 'mlops-api'
    static_configs:
      - targets: ['api:8000']
    scrape_interval: 5s
    
  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow:5000']
    scrape_interval: 30s
    
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
      
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Provisioning

```yaml
# config/grafana/provisioning/dashboards/dashboard.yaml
apiVersion: 1
providers:
  - name: 'MLOps Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards

# config/grafana/provisioning/datasources/prometheus.yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    url: http://prometheus:9090
    isDefault: true
    access: proxy
```

### Logging Configuration

```python
# config/logging.py
import logging
import json
from datetime import datetime

class StructuredLogger:
    """Structured JSON logging for better observability."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        handler.setFormatter(self._get_formatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _get_formatter(self):
        """Custom JSON formatter."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields
                if hasattr(record, 'model_name'):
                    log_entry['model_name'] = record.model_name
                if hasattr(record, 'customer_id'):
                    log_entry['customer_id'] = record.customer_id
                if hasattr(record, 'prediction_id'):
                    log_entry['prediction_id'] = record.prediction_id
                    
                return json.dumps(log_entry)
        
        return JSONFormatter()
    
    def info(self, message: str, **kwargs):
        """Log info with extra context."""
        extra_dict = {f'extra_{k}': v for k, v in kwargs.items()}
        self.logger.info(message, extra=extra_dict)
```

## 📊 Monitoring Scripts

### Health Check Script

```python
#!/usr/bin/env python3
# scripts/health_check.py
import requests
import sys
from typing import Dict, Any

def check_service_health() -> Dict[str, Any]:
    """Check health of all services."""
    services = {
        'api': 'http://localhost:8000/health',
        'mlflow': 'http://localhost:5000/health', 
        'grafana': 'http://localhost:3000/api/health',
        'prometheus': 'http://localhost:9090/-/healthy'
    }
    
    results = {}
    all_healthy = True
    
    for service, url in services.items():
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                results[service] = {'status': 'healthy', 'response_time': response.elapsed.total_seconds()}
            else:
                results[service] = {'status': 'unhealthy', 'error': f'HTTP {response.status_code}'}
                all_healthy = False
        except requests.RequestException as e:
            results[service] = {'status': 'unreachable', 'error': str(e)}
            all_healthy = False
    
    return {'services': results, 'overall_healthy': all_healthy}

if __name__ == '__main__':
    health = check_service_health()
    print(f"Overall Health: {'✅ Healthy' if health['overall_healthy'] else '❌ Unhealthy'}")
    
    for service, status in health['services'].items():
        symbol = '✅' if status['status'] == 'healthy' else '❌'
        print(f"  {service}: {symbol} {status['status']}")
        
    sys.exit(0 if health['overall_healthy'] else 1)
```

### Performance Report Generator

```python
# scripts/generate_performance_report.py
from src.monitoring.performance_tracker import PerformanceTracker
import pandas as pd
from datetime import datetime, timedelta

def generate_weekly_report():
    """Generate weekly performance report."""
    tracker = PerformanceTracker(database_url=os.getenv('DATABASE_URL'))
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    report = {
        'period': f"{start_date.date()} to {end_date.date()}",
        'models': {}
    }
    
    for model_name in ['churn-predictor', 'revenue-predictor']:
        metrics = tracker.calculate_performance_metrics(model_name, 168)  # 7 days
        
        if metrics:
            report['models'][model_name] = {
                'accuracy': f"{metrics['accuracy']:.3f}",
                'precision': f"{metrics['precision']:.3f}",
                'recall': f"{metrics['recall']:.3f}",
                'auc_roc': f"{metrics['auc_roc']:.3f}",
                'predictions_count': metrics['sample_size']
            }
    
    # Generate HTML report
    html_report = generate_html_report(report)
    
    # Save report
    with open(f"reports/performance_report_{end_date.strftime('%Y%m%d')}.html", 'w') as f:
        f.write(html_report)
    
    print(f"Performance report generated: {end_date.strftime('%Y%m%d')}")
```

## 🔍 Troubleshooting

### Common Monitoring Issues

**Metrics not appearing in Grafana:**
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metric collection
curl http://localhost:8000/metrics | grep model_accuracy

# Check Grafana datasource
curl -u admin:admin http://localhost:3000/api/datasources
```

**High memory usage in Prometheus:**
```yaml
# Reduce retention period
global:
  retention: "15d"  # Instead of default 30d
  
# Increase scrape intervals for less critical metrics
scrape_configs:
  - job_name: 'low-priority-metrics'
    scrape_interval: 60s  # Instead of 15s
```

**Alert fatigue:**
```yaml
# Use proper alert grouping and suppression
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h  # Don't repeat too frequently
```

## 📚 Next Steps

After setting up monitoring:

1. **[A/B Testing Guide](ab_testing.md)**: Compare model versions
2. **[Deployment Guide](deployment.md)**: Deploy to production
3. **[API Documentation](api.md)**: Understand serving endpoints
4. **[Training Pipeline](training.md)**: Automated model training

---

**Need help?** Check service logs with `docker-compose logs <service>` or contact the MLOps team.