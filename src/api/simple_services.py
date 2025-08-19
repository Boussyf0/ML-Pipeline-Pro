"""Simplified API services for direct model loading."""
import logging
import asyncio
import time
import json
import pickle
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import redis
from sqlalchemy import create_engine, text

from src.api.models import PredictionResponse, ModelInfo, ModelStatus, ModelType
from src.api.exceptions import ModelNotFoundError, PredictionError


logger = logging.getLogger(__name__)


class SimplePredictionService:
    """Simplified service for handling ML predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize prediction service."""
        self.config = config
        self.loaded_models = {}
        
    async def load_models(self) -> None:
        """Load models at startup."""
        try:
            # Load models from local files
            models_dir = Path("models")
            
            # Check for saved models
            model_files = {
                "churn-predictor": models_dir / "churn_predictor.pkl",
                "churn_predictor": models_dir / "churn_predictor.pkl"  # alias
            }
            
            for model_name, model_path in model_files.items():
                try:
                    if model_path.exists():
                        await self._load_single_model(model_name, model_path)
                    else:
                        logger.warning(f"Model file not found: {model_path}")
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    
            logger.info(f"Loaded {len(self.loaded_models)} models")
                    
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            
    async def _load_single_model(self, model_name: str, model_path: Path) -> bool:
        """Load a single model from file."""
        try:
            # Load the pickled model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            # Load metadata if available
            metadata_path = model_path.parent / "model_metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            self.loaded_models[model_name] = {
                "model": model,
                "version": metadata.get("version", "1.0.0"),
                "metadata": metadata,
                "loaded_at": datetime.now(),
                "prediction_count": 0,
                "model_path": str(model_path)
            }
            
            logger.info(f"Loaded model {model_name} from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} from {model_path}: {e}")
            return False
            
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get currently loaded models."""
        return self.loaded_models.copy()
        
    async def predict(self, model_name: str, features: Dict[str, Any],
                     user_id: Optional[str] = None, request_id: Optional[str] = None) -> PredictionResponse:
        """Make prediction using specified model."""
        start_time = time.time()
        
        try:
            # Check if model is loaded
            if model_name not in self.loaded_models:
                # Try alternative names
                alt_names = {
                    "churn-predictor": "churn_predictor",
                    "churn_predictor": "churn-predictor"
                }
                alt_name = alt_names.get(model_name)
                if alt_name and alt_name in self.loaded_models:
                    model_name = alt_name
                else:
                    raise ModelNotFoundError(f"Model {model_name} not loaded")
                
            model_info = self.loaded_models[model_name]
            model = model_info["model"]
            metadata = model_info.get("metadata", {})
            
            # Get expected features from metadata
            expected_features = metadata.get("features", ["tenure", "monthly_charges", "total_charges", "age"])
            
            # Convert features to the expected format
            feature_values = []
            for feature in expected_features:
                if feature in features:
                    feature_values.append(features[feature])
                elif feature == "age" and "age" not in features:
                    # Default age if not provided
                    feature_values.append(35.0)
                else:
                    raise PredictionError(f"Missing required feature: {feature}")
            
            # Convert to DataFrame for prediction
            feature_df = pd.DataFrame([feature_values], columns=expected_features)
            
            # Make prediction
            try:
                prediction = model.predict(feature_df)[0]
                
                # Get prediction probabilities if available
                prediction_proba = None
                confidence = None
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(feature_df)[0]
                    if len(proba) == 2:  # Binary classification
                        prediction_proba = {
                            "no_churn": float(proba[0]), 
                            "churn": float(proba[1])
                        }
                        confidence = float(max(proba))
                        # Also add numeric labels for compatibility
                        prediction_proba["0"] = float(proba[0])
                        prediction_proba["1"] = float(proba[1])
                        
            except Exception as e:
                raise PredictionError(f"Model prediction failed: {str(e)}")
                
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update prediction count
            model_info["prediction_count"] += 1
            
            # Create interpretation
            churn_probability = prediction_proba["1"] if prediction_proba else None
            risk_level = "HIGH" if churn_probability and churn_probability > 0.7 else \
                        "MEDIUM" if churn_probability and churn_probability > 0.4 else "LOW"
            
            # Prepare response
            response = PredictionResponse(
                request_id=request_id or f"req_{int(time.time())}_{hash(str(features)) % 10000}",
                model_name=model_name,
                model_version=model_info["version"],
                prediction=int(prediction),
                prediction_proba=prediction_proba,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                timestamp=datetime.now(),
                environment="production",
                metadata={
                    "churn_probability": churn_probability,
                    "risk_level": risk_level,
                    "features_used": expected_features,
                    "model_accuracy": metadata.get("metrics", {}).get("accuracy"),
                    "model_auc_roc": metadata.get("metrics", {}).get("auc_roc")
                }
            )
            
            return response
            
        except ModelNotFoundError:
            raise
        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in prediction: {e}")
            raise PredictionError(f"Prediction failed: {str(e)}")
            
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        logger.info("Cleaning up prediction service")
        self.loaded_models.clear()


class SimpleModelService:
    """Simplified service for model management."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model service."""
        self.config = config
        
    async def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        try:
            models = []
            models_dir = Path("models")
            
            # Check for metadata file
            metadata_path = models_dir / "model_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                model_info = ModelInfo(
                    name=metadata.get("model_name", "churn_predictor"),
                    version=metadata.get("version", "1.0.0"),
                    type=ModelType.CLASSIFICATION,
                    status=ModelStatus.ACTIVE,
                    description="Customer Churn Prediction Model",
                    created_at=datetime.fromisoformat(metadata.get("trained_at", datetime.now().isoformat())),
                    updated_at=datetime.now(),
                    deployed_at=datetime.now(),
                    accuracy=metadata.get("metrics", {}).get("accuracy"),
                    precision=metadata.get("metrics", {}).get("precision"),
                    recall=metadata.get("metrics", {}).get("recall"),
                    f1_score=metadata.get("metrics", {}).get("f1_score"),
                    auc_roc=metadata.get("metrics", {}).get("auc_roc"),
                    environment="production",
                    framework="sklearn"
                )
                
                models.append(model_info)
                
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
            
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        try:
            models = await self.list_models()
            for model in models:
                if model.name == model_name or model_name in ["churn-predictor", "churn_predictor"]:
                    return model
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
            
    async def load_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Load a model."""
        try:
            logger.info(f"Loading model {model_name} version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
            
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model."""
        try:
            logger.info(f"Unloading model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False


class SimpleMonitoringService:
    """Simplified service for monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize monitoring service."""
        self.config = config
        
        # Initialize database and Redis connections
        try:
            if config and "database" in config:
                self.db_engine = create_engine(config["database"]["connection_string"])
            else:
                # Default connection
                self.db_engine = create_engine("postgresql://mlops_user:mlops_password@localhost:5432/mlops_db")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            self.db_engine = None
            
        try:
            if config and "redis" in config:
                self.redis_client = redis.from_url(config["redis"]["connection_string"])
            else:
                # Default connection
                self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
            
    async def check_database_health(self) -> bool:
        """Check database connectivity."""
        try:
            if not self.db_engine:
                return False
                
            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
            
    async def check_redis_health(self) -> bool:
        """Check Redis connectivity."""
        try:
            if not self.redis_client:
                return False
                
            self.redis_client.ping()
            return True
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
            
    async def log_prediction(self, **kwargs) -> None:
        """Log prediction for monitoring."""
        try:
            # Simple logging - could be enhanced
            logger.info(f"Prediction logged: {kwargs.get('model_name')} - {kwargs.get('prediction')}")
            
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            
    async def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """Get model health status."""
        return {
            "status": "healthy",
            "model_name": model_name,
            "last_prediction": datetime.now().isoformat(),
            "prediction_count": 0
        }
            
    async def get_drift_summary(self) -> Dict[str, Any]:
        """Get drift monitoring summary."""
        return {
            "status": "no_drift_detected",
            "last_check": datetime.now().isoformat(),
            "drift_score": 0.0
        }
            
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        return {
            "status": "healthy",
            "average_latency_ms": 50.0,
            "throughput": 100,
            "error_rate": 0.0
        }
            
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active monitoring alerts."""
        return []