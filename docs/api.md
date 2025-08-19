# 🚀 API Documentation

FastAPI-based serving infrastructure for real-time model predictions with authentication, monitoring, and A/B testing.

## 🏗️ API Architecture

```
Client Request → Authentication → Rate Limiting → Model Loading → Prediction → Response
     ↓              ↓               ↓              ↓             ↓          ↓
  HTTP/JSON      API Key        Redis Cache    MLflow Registry  Inference  JSON Response
```

## 📡 Base URL

- **Local Development**: `http://localhost:8000`
- **Staging**: `https://api-staging.yourcompany.com`
- **Production**: `https://api.yourcompany.com`

## 🔐 Authentication

All API endpoints require authentication via API key header:

```bash
curl -H "X-API-Key: your-api-key-here" http://localhost:8000/health
```

### API Key Management

```python
# Generate new API key
python scripts/generate_api_key.py --user admin --role read-write

# List API keys
python scripts/list_api_keys.py

# Revoke API key
python scripts/revoke_api_key.py --key-id abc123
```

## 🎯 Endpoints

### Health Check

Check API server health and connectivity.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "dependencies": {
    "database": "connected",
    "redis": "connected", 
    "mlflow": "connected"
  }
}
```

### Single Prediction

Make prediction for a single customer.

```http
POST /predict/{model_name}
```

**Path Parameters:**
- `model_name` (string): Model name from registry (e.g., "churn-predictor")

**Query Parameters:**
- `version` (string, optional): Specific model version (defaults to latest production)
- `explain` (boolean, optional): Include prediction explanation (default: false)

**Request Body:**
```json
{
  "features": {
    "tenure": 12,
    "monthly_charges": 65.0,
    "total_charges": 780.0,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "DSL",
    "phone_service": true,
    "multiple_lines": false,
    "online_security": false,
    "tech_support": false
  },
  "customer_id": "CUST12345"
}
```

**Response:**
```json
{
  "prediction": 0,
  "probability": 0.23,
  "confidence": "high",
  "model_version": "1.2.0",
  "prediction_id": "pred_abc123",
  "timestamp": "2024-01-15T10:30:00Z",
  "explanation": {
    "top_features": [
      {"feature": "contract", "importance": 0.45, "value": "Month-to-month"},
      {"feature": "tenure", "importance": 0.32, "value": 12},
      {"feature": "monthly_charges", "importance": 0.18, "value": 65.0}
    ],
    "shap_values": [0.12, -0.08, 0.05, ...]
  }
}
```

### Batch Predictions

Process multiple predictions in a single request.

```http
POST /predict/batch/{model_name}
```

**Request Body:**
```json
{
  "predictions": [
    {
      "customer_id": "CUST001",
      "features": {...}
    },
    {
      "customer_id": "CUST002", 
      "features": {...}
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "customer_id": "CUST001",
      "prediction": 0,
      "probability": 0.23,
      "prediction_id": "pred_001"
    },
    {
      "customer_id": "CUST002",
      "prediction": 1,
      "probability": 0.87,
      "prediction_id": "pred_002"
    }
  ],
  "batch_id": "batch_abc123",
  "processed_count": 2,
  "model_version": "1.2.0",
  "processing_time_ms": 45
}
```

### Model Management

#### List Available Models

```http
GET /models
```

**Response:**
```json
{
  "models": [
    {
      "name": "churn-predictor",
      "versions": [
        {
          "version": "1.2.0",
          "stage": "Production",
          "created_at": "2024-01-15T08:00:00Z",
          "metrics": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.79
          }
        },
        {
          "version": "1.3.0",
          "stage": "Staging", 
          "created_at": "2024-01-15T10:00:00Z",
          "metrics": {
            "accuracy": 0.87,
            "precision": 0.84,
            "recall": 0.81
          }
        }
      ]
    }
  ]
}
```

#### Get Model Details

```http
GET /models/{model_name}
```

**Response:**
```json
{
  "name": "churn-predictor",
  "description": "Customer churn prediction model",
  "current_version": "1.2.0",
  "feature_schema": {
    "tenure": {"type": "integer", "min": 0, "max": 100},
    "monthly_charges": {"type": "number", "min": 0, "max": 200},
    "contract": {"type": "string", "enum": ["Month-to-month", "One year", "Two year"]}
  },
  "created_at": "2024-01-10T12:00:00Z",
  "last_prediction": "2024-01-15T10:29:00Z",
  "total_predictions": 15420
}
```

### A/B Testing

#### Get Traffic Assignment

```http
GET /ab-test/assignment/{customer_id}
```

**Response:**
```json
{
  "customer_id": "CUST12345",
  "experiment_id": "exp_001",
  "variant": "B",
  "model_name": "churn-predictor",
  "model_version": "1.3.0",
  "assigned_at": "2024-01-15T10:30:00Z"
}
```

#### Record Experiment Event

```http
POST /ab-test/events
```

**Request Body:**
```json
{
  "customer_id": "CUST12345",
  "experiment_id": "exp_001",
  "event_type": "conversion",
  "event_value": 1.0,
  "timestamp": "2024-01-15T10:35:00Z",
  "metadata": {
    "source": "email_campaign",
    "campaign_id": "camp_123"
  }
}
```

### Monitoring

#### Get Metrics

```http
GET /metrics
```

**Response:** Prometheus metrics format
```
# HELP prediction_requests_total Total number of prediction requests
# TYPE prediction_requests_total counter
prediction_requests_total{model="churn-predictor",version="1.2.0"} 1542

# HELP prediction_latency_seconds Request latency in seconds
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.1"} 1200
prediction_latency_seconds_bucket{le="0.5"} 1540
prediction_latency_seconds_bucket{le="1.0"} 1542
```

#### Get Model Performance

```http
GET /monitoring/performance/{model_name}
```

**Response:**
```json
{
  "model_name": "churn-predictor", 
  "current_version": "1.2.0",
  "time_range": "24h",
  "metrics": {
    "total_predictions": 1542,
    "avg_latency_ms": 45,
    "error_rate": 0.001,
    "accuracy": 0.84,
    "drift_score": 0.02
  },
  "hourly_stats": [
    {"hour": "2024-01-15T09:00:00Z", "predictions": 67, "latency_ms": 43},
    {"hour": "2024-01-15T10:00:00Z", "predictions": 89, "latency_ms": 48}
  ]
}
```

## 🛡️ Error Handling

### Error Response Format

```json
{
  "error": "ValidationError",
  "message": "Invalid feature values provided",
  "details": {
    "field": "monthly_charges",
    "issue": "Value must be positive"
  },
  "request_id": "req_abc123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### HTTP Status Codes

- **200**: Success
- **400**: Bad Request (validation error)
- **401**: Unauthorized (invalid API key)
- **403**: Forbidden (insufficient permissions)
- **404**: Not Found (model not found)
- **422**: Unprocessable Entity (invalid input format)
- **429**: Too Many Requests (rate limit exceeded)
- **500**: Internal Server Error
- **503**: Service Unavailable (dependency failure)

## 📊 Rate Limiting

API endpoints are rate limited to ensure fair usage:

- **Default**: 100 requests/minute per API key
- **Batch predictions**: 10 requests/minute per API key
- **Model management**: 20 requests/minute per API key

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

## 🚀 Client Libraries

### Python Client

```python
from mlops_client import MLOpsClient

# Initialize client
client = MLOpsClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Make prediction
result = client.predict(
    model_name="churn-predictor",
    features={
        "tenure": 12,
        "monthly_charges": 65.0,
        "contract": "Month-to-month"
    }
)

print(f"Churn probability: {result.probability}")
```

### JavaScript Client

```javascript
import { MLOpsClient } from '@company/mlops-client';

const client = new MLOpsClient({
  baseURL: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

const result = await client.predict('churn-predictor', {
  tenure: 12,
  monthly_charges: 65.0,
  contract: 'Month-to-month'
});

console.log(`Churn probability: ${result.probability}`);
```

### cURL Examples

```bash
# Single prediction
curl -X POST http://localhost:8000/predict/churn-predictor \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "features": {
      "tenure": 12,
      "monthly_charges": 65.0,
      "contract": "Month-to-month"
    }
  }'

# Get model list
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/models

# Health check
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/health
```

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_LOG_LEVEL=info

# Authentication
API_KEY_STORE=redis
JWT_SECRET_KEY=your-secret-key

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Model Serving
MODEL_CACHE_TTL=3600
PREDICTION_TIMEOUT=30
BATCH_SIZE_LIMIT=1000

# Monitoring
METRICS_ENABLED=true
TRACING_ENABLED=true
LOG_PREDICTIONS=true
```

### API Configuration File

```yaml
# config/api.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false
  
authentication:
  enabled: true
  api_key_header: "X-API-Key"
  jwt_enabled: false
  
rate_limiting:
  enabled: true
  default_limit: 100
  window_seconds: 60
  storage: "redis"
  
models:
  cache_ttl_seconds: 3600
  timeout_seconds: 30
  max_batch_size: 1000
  
monitoring:
  metrics_enabled: true
  log_predictions: true
  trace_requests: true
```

## 🧪 Testing

### Unit Tests

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health", headers={"X-API-Key": "test_key"})
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction():
    response = client.post(
        "/predict/churn-predictor",
        headers={"X-API-Key": "test_key"},
        json={
            "features": {
                "tenure": 12,
                "monthly_charges": 65.0,
                "contract": "Month-to-month"
            }
        }
    )
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability" in response.json()
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 -H "X-API-Key: test_key" \
  -p prediction_payload.json \
  -T "application/json" \
  http://localhost:8000/predict/churn-predictor

# Using Locust
locust -f tests/load_test.py --host=http://localhost:8000
```

## 📈 Performance Optimization

### Response Caching

```python
# Enable Redis caching for predictions
CACHE_PREDICTIONS=true
CACHE_TTL_SECONDS=300

# Cache key format: predict:{model}:{version}:{features_hash}
```

### Async Processing

```python
# API supports async request processing
@app.post("/predict/async/{model_name}")
async def async_predict(model_name: str, request: PredictionRequest):
    # Queue prediction task
    task_id = await queue_prediction_task(model_name, request)
    return {"task_id": task_id, "status": "queued"}

@app.get("/predict/async/{task_id}/result")
async def get_async_result(task_id: str):
    # Get result from task queue
    result = await get_task_result(task_id)
    return result
```

### Model Warming

```python
# Pre-load frequently used models
python scripts/warm_models.py --models churn-predictor,revenue-predictor

# Auto-warming based on usage patterns
AUTO_WARM_MODELS=true
WARM_THRESHOLD_REQUESTS_PER_HOUR=10
```

## 🚨 Monitoring & Alerting

### Key Metrics

- **Request Rate**: Requests per second
- **Response Time**: P50, P95, P99 latencies
- **Error Rate**: 4xx and 5xx error percentages
- **Model Performance**: Accuracy, drift scores
- **Resource Usage**: CPU, memory, disk

### Alerts Configuration

```yaml
# config/alerts.yaml
alerts:
  - name: "High Error Rate"
    condition: "error_rate > 0.05"
    duration: "5m"
    severity: "critical"
    
  - name: "High Latency"
    condition: "p95_latency > 1000ms"
    duration: "2m"
    severity: "warning"
    
  - name: "Model Drift Detected"
    condition: "drift_score > 0.1"
    duration: "1m"
    severity: "warning"
```

## 📝 API Changelog

### v1.2.0 (2024-01-15)
- Added batch prediction endpoint
- Improved error handling with detailed messages
- Added model explanation support
- Enhanced monitoring metrics

### v1.1.0 (2024-01-10)
- Added A/B testing endpoints
- Implemented rate limiting
- Added async prediction support
- Improved authentication system

### v1.0.0 (2024-01-01)
- Initial API release
- Basic prediction endpoints
- Health check and metrics
- Model management endpoints

---

**Need help?** Check the [troubleshooting guide](setup.md#troubleshooting) or contact the MLOps team.