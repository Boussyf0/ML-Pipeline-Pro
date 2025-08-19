"""Custom exceptions for the API."""
from typing import Any, Dict, Optional


class MLOpsAPIException(Exception):
    """Base exception for MLOps API."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """Initialize exception."""
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ModelNotFoundError(MLOpsAPIException):
    """Exception raised when model is not found."""
    
    def __init__(self, message: str, model_name: Optional[str] = None):
        """Initialize model not found error."""
        super().__init__(message, "MODEL_NOT_FOUND", {"model_name": model_name})
        self.model_name = model_name


class PredictionError(MLOpsAPIException):
    """Exception raised when prediction fails."""
    
    def __init__(self, message: str, model_name: Optional[str] = None):
        """Initialize prediction error."""
        super().__init__(message, "PREDICTION_ERROR", {"model_name": model_name})
        self.model_name = model_name


class ValidationError(MLOpsAPIException):
    """Exception raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        """Initialize validation error."""
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})
        self.field = field
        self.value = value


class AuthenticationError(MLOpsAPIException):
    """Exception raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication required"):
        """Initialize authentication error."""
        super().__init__(message, "AUTHENTICATION_ERROR")


class AuthorizationError(MLOpsAPIException):
    """Exception raised when authorization fails."""
    
    def __init__(self, message: str = "Insufficient permissions"):
        """Initialize authorization error."""
        super().__init__(message, "AUTHORIZATION_ERROR")


class RateLimitError(MLOpsAPIException):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        """Initialize rate limit error."""
        super().__init__(message, "RATE_LIMIT_ERROR", {"retry_after": retry_after})
        self.retry_after = retry_after


class ServiceUnavailableError(MLOpsAPIException):
    """Exception raised when service is unavailable."""
    
    def __init__(self, message: str = "Service temporarily unavailable"):
        """Initialize service unavailable error."""
        super().__init__(message, "SERVICE_UNAVAILABLE")


class ModelLoadError(MLOpsAPIException):
    """Exception raised when model loading fails."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 model_version: Optional[str] = None):
        """Initialize model load error."""
        super().__init__(
            message, 
            "MODEL_LOAD_ERROR", 
            {"model_name": model_name, "model_version": model_version}
        )
        self.model_name = model_name
        self.model_version = model_version


class DataDriftError(MLOpsAPIException):
    """Exception raised when data drift is detected."""
    
    def __init__(self, message: str, features: Optional[list] = None):
        """Initialize data drift error."""
        super().__init__(message, "DATA_DRIFT_ERROR", {"features": features})
        self.features = features


class PerformanceDegradationError(MLOpsAPIException):
    """Exception raised when model performance degrades."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 metric: Optional[str] = None, current_value: Optional[float] = None,
                 threshold: Optional[float] = None):
        """Initialize performance degradation error."""
        super().__init__(
            message, 
            "PERFORMANCE_DEGRADATION_ERROR",
            {
                "model_name": model_name,
                "metric": metric,
                "current_value": current_value,
                "threshold": threshold
            }
        )
        self.model_name = model_name
        self.metric = metric
        self.current_value = current_value
        self.threshold = threshold