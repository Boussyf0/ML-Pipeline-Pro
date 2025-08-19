"""API services for model serving and monitoring."""
import logging
import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import redis
from sqlalchemy import create_engine, text
import yaml

from src.api.models import PredictionResponse, ModelInfo, ModelStatus, ModelType
from src.api.exceptions import ModelNotFoundError, PredictionError
from src.models.model_manager import ModelManager
from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.alerting import AlertManager


logger = logging.getLogger(__name__)


class PredictionService:
    """Service for handling ML predictions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize prediction service."""
        self.config = config
        self.loaded_models = {}
        self.model_manager = ModelManager() if config else None
        
    async def load_models(self) -> None:
        """Load models at startup."""
        try:
            if not self.model_manager:
                logger.warning("No model manager available, skipping model loading")
                return
                
            # Load default models
            models_to_load = ["churn-predictor"]  # Add more models as needed
            
            for model_name in models_to_load:
                try:
                    await self._load_single_model(model_name)
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            
    async def _load_single_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Load a single model."""
        try:
            # Get model from manager (this handles A/B testing logic)
            model, model_version, experiment_group = self.model_manager.get_model_for_prediction(model_name)
            
            self.loaded_models[model_name] = {
                "model": model,
                "version": model_version,
                "experiment_group": experiment_group,
                "loaded_at": datetime.now(),
                "prediction_count": 0
            }
            
            logger.info(f"Loaded model {model_name} version {model_version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
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
                raise ModelNotFoundError(f"Model {model_name} not loaded")
                
            model_info = self.loaded_models[model_name]
            model = model_info["model"]
            
            # Convert features to DataFrame for prediction
            feature_df = pd.DataFrame([features])
            
            # Make prediction
            try:
                prediction = model.predict(feature_df)[0]
                
                # Get prediction probabilities if available
                prediction_proba = None
                confidence = None
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(feature_df)[0]
                    if len(proba) == 2:  # Binary classification
                        prediction_proba = {"0": float(proba[0]), "1": float(proba[1])}
                        confidence = float(max(proba))
                    else:  # Multi-class
                        prediction_proba = {str(i): float(p) for i, p in enumerate(proba)}
                        confidence = float(max(proba))
                        
            except Exception as e:
                raise PredictionError(f"Model prediction failed: {str(e)}")
                
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update prediction count
            model_info["prediction_count"] += 1
            
            # Prepare response
            response = PredictionResponse(
                request_id=request_id or f"req_{int(time.time())}_{hash(str(features)) % 10000}",
                model_name=model_name,
                model_version=model_info["version"],
                prediction=prediction,
                prediction_proba=prediction_proba,
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                timestamp=datetime.now(),
                environment="production",  # This could be configurable
                ab_test_group=model_info.get("experiment_group")
            )
            
            return response
            
        except ModelNotFoundError:
            raise
        except PredictionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in prediction: {e}")
            raise PredictionError(f"Prediction failed: {str(e)}")
            
    async def get_ab_test_status(self, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Get A/B test status."""
        try:
            if not self.model_manager:
                return None
                
            # This would call the model manager's A/B test analysis
            return {"status": "not_implemented"}
            
        except Exception as e:
            logger.error(f"Failed to get A/B test status: {e}")
            return None
            
    async def setup_ab_test(self, model_a_name: str, model_a_version: str,
                           model_b_name: str, model_b_version: str,
                           traffic_split: float, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Setup A/B test."""
        try:
            if not self.model_manager:
                raise PredictionError("Model manager not available")
                
            result = self.model_manager.setup_ab_test(
                model_a_name, model_a_version,
                model_b_name, model_b_version,
                traffic_split, experiment_name
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to setup A/B test: {e}")
            raise PredictionError(f"A/B test setup failed: {str(e)}")
            
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        logger.info("Cleaning up prediction service")
        self.loaded_models.clear()


class ModelService:
    """Service for model management."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize model service."""
        self.config = config
        self.mlflow_client = MlflowClient()
        
    async def list_models(self) -> List[ModelInfo]:
        """List all available models."""
        try:
            # Get models from MLflow registry
            registered_models = self.mlflow_client.search_registered_models()
            
            models = []
            for model in registered_models:
                # Get latest version
                latest_versions = self.mlflow_client.get_latest_versions(
                    model.name, stages=["Production", "Staging"]
                )
                
                if latest_versions:
                    latest_version = latest_versions[0]
                    
                    # Get run metrics if available
                    run = self.mlflow_client.get_run(latest_version.run_id)
                    metrics = run.data.metrics
                    
                    model_info = ModelInfo(
                        name=model.name,
                        version=latest_version.version,
                        type=ModelType.CLASSIFICATION,  # This could be inferred
                        status=ModelStatus.ACTIVE if latest_version.current_stage == "Production" else ModelStatus.INACTIVE,
                        description=model.description or f"MLflow model: {model.name}",
                        created_at=datetime.fromtimestamp(model.creation_timestamp / 1000),
                        updated_at=datetime.fromtimestamp(model.last_updated_timestamp / 1000),
                        deployed_at=datetime.fromtimestamp(latest_version.creation_timestamp / 1000),
                        accuracy=metrics.get("accuracy"),
                        precision=metrics.get("precision"),
                        recall=metrics.get("recall"),
                        f1_score=metrics.get("f1_score"),
                        auc_roc=metrics.get("auc_roc"),
                        environment="production",
                        framework="sklearn"  # This could be detected
                    )
                    
                    models.append(model_info)
                    
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
            
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        try:
            # Get model from MLflow registry
            model = self.mlflow_client.get_registered_model(model_name)
            
            # Get latest version
            latest_versions = self.mlflow_client.get_latest_versions(
                model_name, stages=["Production", "Staging"]
            )
            
            if not latest_versions:
                return None
                
            latest_version = latest_versions[0]
            
            # Get run metrics
            run = self.mlflow_client.get_run(latest_version.run_id)
            metrics = run.data.metrics
            
            return ModelInfo(
                name=model.name,
                version=latest_version.version,
                type=ModelType.CLASSIFICATION,
                status=ModelStatus.ACTIVE if latest_version.current_stage == "Production" else ModelStatus.INACTIVE,
                description=model.description or f"MLflow model: {model.name}",
                created_at=datetime.fromtimestamp(model.creation_timestamp / 1000),
                updated_at=datetime.fromtimestamp(model.last_updated_timestamp / 1000),
                deployed_at=datetime.fromtimestamp(latest_version.creation_timestamp / 1000),
                accuracy=metrics.get("accuracy"),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                f1_score=metrics.get("f1_score"),
                auc_roc=metrics.get("auc_roc"),
                environment="production"
            )
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return None
            
    async def load_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Load a model."""
        try:
            # This would delegate to the prediction service
            logger.info(f"Loading model {model_name} version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
            
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model."""
        try:
            # This would delegate to the prediction service
            logger.info(f"Unloading model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False


class MonitoringService:
    """Service for monitoring and alerting."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize monitoring service."""
        self.config = config
        
        if config:
            self.db_engine = create_engine(config["database"]["connection_string"])
            self.redis_client = redis.from_url(config["redis"]["connection_string"])
            self.model_monitor = ModelMonitor()
            self.drift_detector = DriftDetector()
            self.alert_manager = AlertManager()
        else:
            self.db_engine = None
            self.redis_client = None
            self.model_monitor = None
            self.drift_detector = None
            self.alert_manager = None
            
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
            
    async def log_prediction(self, model_name: str, model_version: str,
                           features: Dict[str, Any], prediction: Any,
                           prediction_proba: Optional[Dict[str, float]] = None,
                           latency_ms: Optional[float] = None,
                           user_id: Optional[str] = None,
                           request_id: Optional[str] = None) -> None:
        """Log prediction for monitoring."""
        try:
            if not self.model_monitor:
                return
                
            self.model_monitor.log_prediction(
                model_name=model_name,
                model_version=model_version,
                environment="production",
                prediction=prediction,
                prediction_proba=prediction_proba.get("1") if prediction_proba else None,
                latency_ms=latency_ms,
                input_features=features,
                user_id=user_id
            )
            
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            
    async def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """Get model health status."""
        try:
            if not self.model_monitor:
                return {"status": "unknown", "error": "Monitoring not available"}
                
            health = self.model_monitor.get_model_health_dashboard(model_name, "production")
            return health
            
        except Exception as e:
            logger.error(f"Failed to get model health: {e}")
            return {"status": "error", "error": str(e)}
            
    async def get_drift_summary(self) -> Dict[str, Any]:
        """Get drift monitoring summary."""
        try:
            if not self.drift_detector:
                return {"status": "unknown", "error": "Drift detection not available"}
                
            summary = self.drift_detector.get_drift_summary(days=7)
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get drift summary: {e}")
            return {"error": str(e)}
            
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        try:
            if not self.model_monitor:
                return {"status": "unknown", "error": "Performance monitoring not available"}
                
            # This would aggregate performance metrics
            return {"status": "healthy", "summary": "Performance monitoring active"}
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
            
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active monitoring alerts."""
        try:
            if not self.alert_manager:
                return []
                
            alerts = self.alert_manager.get_active_alerts()
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []