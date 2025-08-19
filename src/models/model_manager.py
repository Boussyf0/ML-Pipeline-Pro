"""Advanced model management with versioning, rollback, and A/B testing support."""
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from sklearn.base import BaseEstimator
import psycopg2
from sqlalchemy import create_engine, text
import redis
import yaml


logger = logging.getLogger(__name__)


class ModelManager:
    """Advanced model management system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model manager."""
        self.config = self._load_config(config_path)
        
        # Initialize connections
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        self.mlflow_client = MlflowClient()
        
        self.db_engine = create_engine(self.config["database"]["connection_string"])
        self.redis_client = redis.from_url(self.config["redis"]["connection_string"])
        
        self.registered_model_name = self.config["mlflow"]["registered_model_name"]
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def deploy_model(self, model_name: str, version: str, 
                    environment: str = "staging") -> Dict[str, Any]:
        """Deploy a model version to specified environment."""
        try:
            # Get model from registry
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            # Cache model in Redis for fast serving
            cache_key = f"model:{model_name}:{version}:{environment}"
            
            # Serialize model for caching (simplified - in production use proper serialization)
            model_data = {
                "model_uri": model_uri,
                "version": version,
                "environment": environment,
                "deployed_at": datetime.now().isoformat(),
                "status": "active"
            }
            
            self.redis_client.hset(cache_key, mapping=model_data)
            
            # Update model stage in MLflow
            if environment.lower() == "production":
                self.mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Production"
                )
            elif environment.lower() == "staging":
                self.mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Staging"
                )
                
            # Log deployment event
            self._log_deployment_event(model_name, version, environment, "deployed")
            
            logger.info(f"Model {model_name} v{version} deployed to {environment}")
            
            return {
                "model_name": model_name,
                "version": version,
                "environment": environment,
                "status": "deployed",
                "deployed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise
            
    def rollback_model(self, model_name: str, environment: str, 
                      target_version: Optional[str] = None) -> Dict[str, Any]:
        """Rollback to previous model version."""
        try:
            # Get current deployed version
            current_version = self._get_deployed_version(model_name, environment)
            
            if not current_version:
                raise ValueError(f"No model currently deployed in {environment}")
                
            # Get target version (previous if not specified)
            if not target_version:
                target_version = self._get_previous_version(model_name, current_version)
                
            if not target_version:
                raise ValueError("No previous version available for rollback")
                
            # Deploy target version
            rollback_result = self.deploy_model(model_name, target_version, environment)
            
            # Archive current version
            self.mlflow_client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Archived"
            )
            
            # Log rollback event
            self._log_deployment_event(
                model_name, target_version, environment, "rollback",
                metadata={"from_version": current_version}
            )
            
            logger.info(f"Rolled back {model_name} from v{current_version} to v{target_version}")
            
            return rollback_result
            
        except Exception as e:
            logger.error(f"Failed to rollback model: {e}")
            raise
            
    def setup_ab_test(self, model_a_name: str, model_a_version: str,
                     model_b_name: str, model_b_version: str,
                     traffic_split: float = 0.5,
                     experiment_name: Optional[str] = None) -> Dict[str, Any]:
        """Setup A/B test between two model versions."""
        try:
            if not experiment_name:
                experiment_name = f"ab_test_{model_a_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            # Validate traffic split
            if not 0 < traffic_split < 1:
                raise ValueError("Traffic split must be between 0 and 1")
                
            # Deploy both models to A/B testing environment
            model_a_deployment = self.deploy_model(model_a_name, model_a_version, "ab_test_a")
            model_b_deployment = self.deploy_model(model_b_name, model_b_version, "ab_test_b")
            
            # Store A/B test configuration
            ab_config = {
                "experiment_name": experiment_name,
                "model_a": {
                    "name": model_a_name,
                    "version": model_a_version,
                    "traffic_ratio": traffic_split
                },
                "model_b": {
                    "name": model_b_name,
                    "version": model_b_version,
                    "traffic_ratio": 1 - traffic_split
                },
                "start_time": datetime.now().isoformat(),
                "status": "active",
                "minimum_sample_size": self.config["ab_testing"]["minimum_sample_size"],
                "significance_level": self.config["ab_testing"]["significance_level"]
            }
            
            # Cache A/B test config
            self.redis_client.set(
                f"ab_test:{experiment_name}",
                json.dumps(ab_config),
                ex=86400 * 30  # 30 days expiry
            )
            
            logger.info(f"A/B test setup: {experiment_name}")
            return ab_config
            
        except Exception as e:
            logger.error(f"Failed to setup A/B test: {e}")
            raise
            
    def get_model_for_prediction(self, model_name: str, 
                                user_id: Optional[str] = None) -> Tuple[Any, str, str]:
        """Get model for prediction, considering A/B tests."""
        try:
            # Check if there's an active A/B test
            ab_test_key = None
            for key in self.redis_client.scan_iter(match="ab_test:*"):
                ab_config = json.loads(self.redis_client.get(key))
                if (ab_config["model_a"]["name"] == model_name or 
                    ab_config["model_b"]["name"] == model_name) and \
                   ab_config["status"] == "active":
                    ab_test_key = key
                    break
                    
            if ab_test_key:
                # Route traffic for A/B test
                ab_config = json.loads(self.redis_client.get(ab_test_key))
                
                # Determine which model to use based on user hash or random
                if user_id:
                    import hashlib
                    user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
                    use_model_a = (user_hash % 100) / 100 < ab_config["model_a"]["traffic_ratio"]
                else:
                    use_model_a = np.random.random() < ab_config["model_a"]["traffic_ratio"]
                    
                if use_model_a:
                    selected_model = ab_config["model_a"]
                    experiment_group = "A"
                else:
                    selected_model = ab_config["model_b"]
                    experiment_group = "B"
                    
                # Load model
                model_uri = f"models:/{selected_model['name']}/{selected_model['version']}"
                model = mlflow.pyfunc.load_model(model_uri)
                
                return model, selected_model["version"], experiment_group
            else:
                # Get production model
                production_version = self._get_deployed_version(model_name, "production")
                if not production_version:
                    # Fallback to staging
                    production_version = self._get_deployed_version(model_name, "staging")
                    
                if not production_version:
                    raise ValueError(f"No deployed model found for {model_name}")
                    
                model_uri = f"models:/{model_name}/{production_version}"
                model = mlflow.pyfunc.load_model(model_uri)
                
                return model, production_version, "production"
                
        except Exception as e:
            logger.error(f"Failed to get model for prediction: {e}")
            raise
            
    def log_prediction(self, model_name: str, model_version: str,
                      input_features: Dict[str, Any], prediction: float,
                      prediction_proba: Optional[float] = None,
                      response_time_ms: Optional[int] = None,
                      user_id: Optional[str] = None,
                      experiment_group: Optional[str] = None) -> None:
        """Log prediction for monitoring and analysis."""
        try:
            # Insert into database
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO app_data.prediction_logs 
                    (model_name, model_version, input_features, prediction, 
                     prediction_proba, response_time_ms, user_id, session_id)
                    VALUES 
                    (:model_name, :model_version, :input_features, :prediction,
                     :prediction_proba, :response_time_ms, :user_id, :experiment_group)
                """)
                
                conn.execute(query, {
                    "model_name": model_name,
                    "model_version": model_version,
                    "input_features": json.dumps(input_features),
                    "prediction": float(prediction),
                    "prediction_proba": float(prediction_proba) if prediction_proba else None,
                    "response_time_ms": response_time_ms,
                    "user_id": user_id,
                    "experiment_group": experiment_group
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            # Don't raise - logging failure shouldn't break prediction serving
            
    def analyze_ab_test(self, experiment_name: str) -> Dict[str, Any]:
        """Analyze A/B test results."""
        try:
            # Get A/B test configuration
            ab_config = json.loads(self.redis_client.get(f"ab_test:{experiment_name}"))
            
            if not ab_config:
                raise ValueError(f"A/B test {experiment_name} not found")
                
            # Get prediction logs for both models
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT 
                        model_name,
                        model_version,
                        session_id as experiment_group,
                        COUNT(*) as prediction_count,
                        AVG(prediction) as avg_prediction,
                        AVG(prediction_proba) as avg_prediction_proba,
                        AVG(response_time_ms) as avg_response_time
                    FROM app_data.prediction_logs 
                    WHERE model_name IN (:model_a, :model_b)
                    AND model_version IN (:version_a, :version_b)
                    AND created_at >= :start_time
                    GROUP BY model_name, model_version, session_id
                """)
                
                results = conn.execute(query, {
                    "model_a": ab_config["model_a"]["name"],
                    "model_b": ab_config["model_b"]["name"],
                    "version_a": ab_config["model_a"]["version"],
                    "version_b": ab_config["model_b"]["version"],
                    "start_time": ab_config["start_time"]
                }).fetchall()
                
            # Analyze results
            model_a_metrics = {}
            model_b_metrics = {}
            
            for row in results:
                if row.experiment_group == "A":
                    model_a_metrics = {
                        "prediction_count": row.prediction_count,
                        "avg_prediction": row.avg_prediction,
                        "avg_response_time": row.avg_response_time
                    }
                elif row.experiment_group == "B":
                    model_b_metrics = {
                        "prediction_count": row.prediction_count,
                        "avg_prediction": row.avg_prediction,
                        "avg_response_time": row.avg_response_time
                    }
                    
            # Statistical significance test (simplified)
            min_sample_size = ab_config["minimum_sample_size"]
            total_samples = model_a_metrics.get("prediction_count", 0) + model_b_metrics.get("prediction_count", 0)
            
            analysis = {
                "experiment_name": experiment_name,
                "model_a": {
                    "name": ab_config["model_a"]["name"],
                    "version": ab_config["model_a"]["version"],
                    "metrics": model_a_metrics
                },
                "model_b": {
                    "name": ab_config["model_b"]["name"],
                    "version": ab_config["model_b"]["version"],
                    "metrics": model_b_metrics
                },
                "total_samples": total_samples,
                "sufficient_samples": total_samples >= min_sample_size,
                "test_duration_days": (
                    datetime.now() - datetime.fromisoformat(ab_config["start_time"])
                ).days,
                "status": ab_config["status"]
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze A/B test: {e}")
            raise
            
    def _get_deployed_version(self, model_name: str, environment: str) -> Optional[str]:
        """Get currently deployed version for environment."""
        try:
            cache_pattern = f"model:{model_name}:*:{environment}"
            for key in self.redis_client.scan_iter(match=cache_pattern):
                model_data = self.redis_client.hgetall(key)
                if model_data.get(b"status") == b"active":
                    return model_data.get(b"version").decode()
            return None
        except Exception:
            return None
            
    def _get_previous_version(self, model_name: str, current_version: str) -> Optional[str]:
        """Get previous version of a model."""
        try:
            versions = self.mlflow_client.search_model_versions(f"name='{model_name}'")
            version_numbers = [int(v.version) for v in versions]
            version_numbers.sort(reverse=True)
            
            current_version_num = int(current_version)
            for version_num in version_numbers:
                if version_num < current_version_num:
                    return str(version_num)
                    
            return None
        except Exception:
            return None
            
    def _log_deployment_event(self, model_name: str, version: str, 
                             environment: str, event_type: str,
                             metadata: Optional[Dict] = None) -> None:
        """Log deployment events for audit trail."""
        try:
            event = {
                "model_name": model_name,
                "version": version,
                "environment": environment,
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Store in Redis for recent events
            self.redis_client.lpush(
                f"deployment_events:{model_name}",
                json.dumps(event)
            )
            
            # Keep only last 100 events per model
            self.redis_client.ltrim(f"deployment_events:{model_name}", 0, 99)
            
        except Exception as e:
            logger.error(f"Failed to log deployment event: {e}")
            
    def get_model_health(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive model health status."""
        try:
            health_status = {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "environments": {},
                "recent_predictions": 0,
                "avg_response_time": None,
                "error_rate": 0.0
            }
            
            # Check each environment
            for env in ["production", "staging", "ab_test_a", "ab_test_b"]:
                version = self._get_deployed_version(model_name, env)
                health_status["environments"][env] = {
                    "deployed": version is not None,
                    "version": version,
                    "status": "healthy" if version else "not_deployed"
                }
                
            # Get recent prediction statistics
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT 
                        COUNT(*) as prediction_count,
                        AVG(response_time_ms) as avg_response_time
                    FROM app_data.prediction_logs 
                    WHERE model_name = :model_name
                    AND created_at >= :since
                """)
                
                since = datetime.now() - timedelta(hours=24)
                result = conn.execute(query, {
                    "model_name": model_name,
                    "since": since
                }).fetchone()
                
                if result:
                    health_status["recent_predictions"] = result.prediction_count
                    health_status["avg_response_time"] = result.avg_response_time
                    
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to get model health: {e}")
            return {"error": str(e)}