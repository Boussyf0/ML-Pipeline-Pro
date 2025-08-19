"""Pydantic models for API request/response schemas."""
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum


class ModelType(str, Enum):
    """Model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTICLASS = "multiclass"


class ModelStatus(str, Enum):
    """Model status."""
    ACTIVE = "active"
    INACTIVE = "inactive" 
    LOADING = "loading"
    ERROR = "error"


class HealthStatus(str, Enum):
    """Health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: Dict[str, Union[str, int, float, bool]] = Field(
        ...,
        description="Input features for prediction",
        example={
            "tenure": 12,
            "monthly_charges": 65.0,
            "total_charges": 780.0,
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
            "gender": "Female",
            "partner": "No",
            "dependents": "No",
            "phone_service": "Yes",
            "internet_service": "DSL"
        }
    )
    
    model_version: Optional[str] = Field(
        None,
        description="Specific model version to use (optional)"
    )
    
    return_proba: bool = Field(
        True,
        description="Whether to return prediction probabilities"
    )
    
    explain: bool = Field(
        False,
        description="Whether to return prediction explanations (SHAP values)"
    )
    
    @validator('features')
    def validate_features(cls, v):
        """Validate features dictionary."""
        if not v:
            raise ValueError("Features cannot be empty")
        
        # Check for required features (this would be model-specific)
        required_features = ['tenure', 'monthly_charges']
        for feature in required_features:
            if feature not in v:
                raise ValueError(f"Required feature '{feature}' missing")
                
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    request_id: str = Field(..., description="Unique request identifier")
    model_name: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Version of model used")
    prediction: Union[int, float, str] = Field(..., description="Model prediction")
    prediction_proba: Optional[Dict[str, float]] = Field(
        None, 
        description="Prediction probabilities (for classification)"
    )
    confidence: Optional[float] = Field(
        None,
        description="Prediction confidence score",
        ge=0.0,
        le=1.0
    )
    explanation: Optional[Dict[str, float]] = Field(
        None,
        description="Feature importance/SHAP values"
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    environment: str = Field("production", description="Environment (production/staging)")
    ab_test_group: Optional[str] = Field(None, description="A/B test group if applicable")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "request_id": "12345678-1234-1234-1234-123456789012",
                "model_name": "churn-predictor",
                "model_version": "1.2.0",
                "prediction": 1,
                "prediction_proba": {"0": 0.3, "1": 0.7},
                "confidence": 0.7,
                "explanation": {
                    "tenure": 0.15,
                    "monthly_charges": 0.25,
                    "total_charges": 0.10
                },
                "processing_time_ms": 45.2,
                "timestamp": "2024-01-15T10:30:00Z",
                "environment": "production",
                "ab_test_group": "A"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    instances: List[Dict[str, Union[str, int, float, bool]]] = Field(
        ...,
        description="List of instances for batch prediction",
        max_items=1000  # Limit batch size
    )
    
    model_version: Optional[str] = Field(
        None,
        description="Specific model version to use (optional)"
    )
    
    return_proba: bool = Field(
        True,
        description="Whether to return prediction probabilities"
    )
    
    explain: bool = Field(
        False,
        description="Whether to return prediction explanations"
    )
    
    @validator('instances')
    def validate_instances(cls, v):
        """Validate batch instances."""
        if not v:
            raise ValueError("Instances cannot be empty")
        if len(v) > 1000:
            raise ValueError("Batch size cannot exceed 1000 instances")
        return v


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    request_id: str = Field(..., description="Unique request identifier")
    model_name: str = Field(..., description="Model used for prediction")
    model_version: str = Field(..., description="Version of model used")
    predictions: List[PredictionResponse] = Field(..., description="Batch predictions")
    total_instances: int = Field(..., description="Total number of instances processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    failed_predictions: int = Field(..., description="Number of failed predictions")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    timestamp: datetime = Field(..., description="Batch processing timestamp")


class ModelInfo(BaseModel):
    """Model information."""
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: ModelType = Field(..., description="Model type")
    status: ModelStatus = Field(..., description="Current model status")
    description: Optional[str] = Field(None, description="Model description")
    created_at: datetime = Field(..., description="Model creation timestamp")
    updated_at: datetime = Field(..., description="Model last update timestamp")
    deployed_at: Optional[datetime] = Field(None, description="Model deployment timestamp")
    
    # Model metrics
    accuracy: Optional[float] = Field(None, description="Model accuracy", ge=0.0, le=1.0)
    precision: Optional[float] = Field(None, description="Model precision", ge=0.0, le=1.0)
    recall: Optional[float] = Field(None, description="Model recall", ge=0.0, le=1.0)
    f1_score: Optional[float] = Field(None, description="Model F1 score", ge=0.0, le=1.0)
    auc_roc: Optional[float] = Field(None, description="Model AUC-ROC", ge=0.0, le=1.0)
    
    # Runtime information
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    avg_prediction_time_ms: Optional[float] = Field(None, description="Average prediction time")
    total_predictions: Optional[int] = Field(None, description="Total predictions made")
    
    # Environment info
    environment: str = Field("production", description="Deployment environment")
    framework: Optional[str] = Field(None, description="ML framework used")
    python_version: Optional[str] = Field(None, description="Python version")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "name": "churn-predictor",
                "version": "1.2.0",
                "type": "classification",
                "status": "active",
                "description": "Customer churn prediction model using XGBoost",
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-10T15:30:00Z",
                "deployed_at": "2024-01-10T16:00:00Z",
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80,
                "auc_roc": 0.88,
                "memory_usage_mb": 256.5,
                "avg_prediction_time_ms": 45.2,
                "total_predictions": 10000,
                "environment": "production",
                "framework": "xgboost",
                "python_version": "3.9"
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: HealthStatus = Field(..., description="Overall health status")
    timestamp: float = Field(..., description="Health check timestamp")
    version: str = Field("1.0.0", description="API version")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime in seconds")
    details: Dict[str, Any] = Field(default_factory=dict, description="Detailed health information")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": 1642248600.0,
                "version": "1.0.0",
                "uptime_seconds": 3600.0,
                "details": {
                    "loaded_models": 2,
                    "database_healthy": True,
                    "redis_healthy": True,
                    "models": ["churn-predictor", "fraud-detector"]
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID if available")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "detail": "Model 'unknown-model' not found",
                "error_type": "model_not_found",
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "12345678-1234-1234-1234-123456789012"
            }
        }


class ABTestConfig(BaseModel):
    """A/B test configuration."""
    experiment_name: str = Field(..., description="Experiment name")
    model_a: str = Field(..., description="Model A name")
    model_a_version: str = Field(..., description="Model A version")
    model_b: str = Field(..., description="Model B name")
    model_b_version: str = Field(..., description="Model B version")
    traffic_split: float = Field(
        0.5, 
        description="Traffic split for model A (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    start_date: datetime = Field(..., description="Experiment start date")
    end_date: Optional[datetime] = Field(None, description="Experiment end date")
    status: str = Field("active", description="Experiment status")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "experiment_name": "churn_model_comparison",
                "model_a": "churn-predictor",
                "model_a_version": "1.1.0",
                "model_b": "churn-predictor",
                "model_b_version": "1.2.0",
                "traffic_split": 0.5,
                "start_date": "2024-01-15T00:00:00Z",
                "end_date": "2024-02-15T00:00:00Z",
                "status": "active"
            }
        }


class ABTestResult(BaseModel):
    """A/B test results."""
    experiment_name: str = Field(..., description="Experiment name")
    model_a_metrics: Dict[str, float] = Field(..., description="Model A performance metrics")
    model_b_metrics: Dict[str, float] = Field(..., description="Model B performance metrics")
    statistical_significance: bool = Field(..., description="Whether results are statistically significant")
    confidence_level: float = Field(..., description="Confidence level")
    winner: Optional[str] = Field(None, description="Winning model (A, B, or tie)")
    sample_size_a: int = Field(..., description="Sample size for model A")
    sample_size_b: int = Field(..., description="Sample size for model B")
    test_duration_days: int = Field(..., description="Test duration in days")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "experiment_name": "churn_model_comparison",
                "model_a_metrics": {
                    "accuracy": 0.85,
                    "precision": 0.82,
                    "recall": 0.78
                },
                "model_b_metrics": {
                    "accuracy": 0.87,
                    "precision": 0.84,
                    "recall": 0.80
                },
                "statistical_significance": True,
                "confidence_level": 0.95,
                "winner": "B",
                "sample_size_a": 5000,
                "sample_size_b": 5000,
                "test_duration_days": 14
            }
        }


class DriftAlert(BaseModel):
    """Data drift alert."""
    alert_id: str = Field(..., description="Alert identifier")
    model_name: str = Field(..., description="Model name")
    feature_name: str = Field(..., description="Feature with drift")
    drift_score: float = Field(..., description="Drift score")
    threshold: float = Field(..., description="Drift threshold")
    severity: str = Field(..., description="Alert severity")
    timestamp: datetime = Field(..., description="Alert timestamp")
    
    class Config:
        """Pydantic config."""
        schema_extra = {
            "example": {
                "alert_id": "drift_001",
                "model_name": "churn-predictor",
                "feature_name": "monthly_charges",
                "drift_score": 0.15,
                "threshold": 0.1,
                "severity": "medium",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }