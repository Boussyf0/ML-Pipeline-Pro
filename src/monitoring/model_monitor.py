"""Real-time model performance monitoring."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import yaml
from sqlalchemy import create_engine, text
import redis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    model_name: str
    model_version: str
    environment: str
    timestamp: datetime
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    prediction_latency_ms: Optional[float] = None
    throughput_per_minute: Optional[float] = None
    error_rate: Optional[float] = None
    prediction_count: int = 0


@dataclass
class PerformanceAlert:
    """Performance degradation alert."""
    alert_type: str
    model_name: str
    model_version: str
    environment: str
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    severity: str
    timestamp: datetime
    details: Dict[str, Any]


class ModelMonitor:
    """Comprehensive model performance monitoring system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model monitor."""
        self.config = self._load_config(config_path)
        self.db_engine = create_engine(self.config["database"]["connection_string"])
        self.redis_client = redis.from_url(self.config["redis"]["connection_string"])
        
        # Monitoring configuration
        self.monitoring_config = self.config["monitoring"]["performance_monitoring"]
        self.alert_thresholds = self.monitoring_config["alert_thresholds"]
        
        # Baseline metrics cache
        self.baseline_cache = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def log_prediction(self, model_name: str, model_version: str, 
                      environment: str, prediction: Union[int, float],
                      actual: Optional[Union[int, float]] = None,
                      prediction_proba: Optional[float] = None,
                      latency_ms: Optional[float] = None,
                      input_features: Optional[Dict] = None,
                      user_id: Optional[str] = None) -> None:
        """Log model prediction for monitoring."""
        try:
            # Store in database
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO app_data.prediction_logs 
                    (model_name, model_version, input_features, prediction, 
                     prediction_proba, response_time_ms, user_id, created_at)
                    VALUES 
                    (:model_name, :model_version, :input_features, :prediction,
                     :prediction_proba, :response_time_ms, :user_id, :created_at)
                """)
                
                conn.execute(query, {
                    "model_name": model_name,
                    "model_version": model_version,
                    "input_features": json.dumps(input_features) if input_features else None,
                    "prediction": float(prediction),
                    "prediction_proba": float(prediction_proba) if prediction_proba else None,
                    "response_time_ms": int(latency_ms) if latency_ms else None,
                    "user_id": user_id,
                    "created_at": datetime.now()
                })
                conn.commit()
                
            # Update real-time metrics in Redis
            self._update_realtime_metrics(model_name, model_version, environment, 
                                        latency_ms, prediction_proba is not None)
            
            # If we have actual value, calculate and store metrics
            if actual is not None:
                self._calculate_performance_metrics(
                    model_name, model_version, environment,
                    prediction, actual, prediction_proba
                )
                
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            
    def _update_realtime_metrics(self, model_name: str, model_version: str,
                               environment: str, latency_ms: Optional[float],
                               has_proba: bool) -> None:
        """Update real-time metrics in Redis."""
        try:
            key = f"realtime_metrics:{model_name}:{model_version}:{environment}"
            current_minute = datetime.now().replace(second=0, microsecond=0)
            
            pipe = self.redis_client.pipeline()
            
            # Increment prediction count
            pipe.hincrby(key, "prediction_count", 1)
            
            # Update latency stats if available
            if latency_ms is not None:
                pipe.hincrby(key, "total_latency", int(latency_ms))
                pipe.hincrby(key, "latency_count", 1)
                
            # Track predictions with probabilities
            if has_proba:
                pipe.hincrby(key, "proba_count", 1)
                
            # Set expiry for 24 hours
            pipe.expire(key, 86400)
            
            # Track hourly throughput
            hourly_key = f"throughput:{model_name}:{environment}:{current_minute.hour}"
            pipe.incr(hourly_key)
            pipe.expire(hourly_key, 3600)  # 1 hour expiry
            
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Failed to update realtime metrics: {e}")
            
    def _calculate_performance_metrics(self, model_name: str, model_version: str,
                                     environment: str, prediction: Union[int, float],
                                     actual: Union[int, float],
                                     prediction_proba: Optional[float]) -> None:
        """Calculate and store performance metrics."""
        try:
            # Get recent predictions for batch metric calculation
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT prediction, input_features, created_at
                    FROM app_data.prediction_logs 
                    WHERE model_name = :model_name 
                    AND model_version = :model_version
                    AND created_at >= :since
                    ORDER BY created_at DESC
                    LIMIT 1000
                """)
                
                since = datetime.now() - timedelta(hours=1)
                results = conn.execute(query, {
                    "model_name": model_name,
                    "model_version": model_version,
                    "since": since
                }).fetchall()
                
            if len(results) >= 10:  # Minimum sample size
                # This is a simplified version - in practice, you'd need actual labels
                # For now, we'll simulate some metrics
                predictions = [r.prediction for r in results]
                
                # Calculate basic statistics
                pred_mean = np.mean(predictions)
                pred_std = np.std(predictions)
                
                # Store metrics
                self._store_performance_metrics(
                    model_name, model_version, environment,
                    pred_mean, pred_std, len(predictions)
                )
                
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            
    def _store_performance_metrics(self, model_name: str, model_version: str,
                                 environment: str, pred_mean: float,
                                 pred_std: float, sample_size: int) -> None:
        """Store calculated performance metrics."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO monitoring.model_metrics 
                    (model_name, model_version, metric_name, metric_value, 
                     environment, created_at)
                    VALUES 
                    (:model_name, :model_version, :metric_name, :metric_value,
                     :environment, :created_at)
                """)
                
                timestamp = datetime.now()
                
                # Store various metrics
                metrics = [
                    ("prediction_mean", pred_mean),
                    ("prediction_std", pred_std),
                    ("sample_size", sample_size),
                ]
                
                for metric_name, metric_value in metrics:
                    conn.execute(query, {
                        "model_name": model_name,
                        "model_version": model_version,
                        "metric_name": metric_name,
                        "metric_value": float(metric_value),
                        "environment": environment,
                        "created_at": timestamp
                    })
                    
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store performance metrics: {e}")
            
    def calculate_metrics_batch(self, model_name: str, model_version: str,
                              environment: str, predictions: List[float],
                              actuals: List[float],
                              probabilities: Optional[List[float]] = None) -> ModelMetrics:
        """Calculate performance metrics for a batch of predictions."""
        try:
            # Convert to numpy arrays
            y_pred = np.array(predictions)
            y_true = np.array(actuals)
            
            # Calculate metrics based on problem type
            if len(np.unique(y_true)) == 2:  # Binary classification
                # Convert probabilities to binary predictions if needed
                if len(np.unique(y_pred)) > 2 and probabilities is not None:
                    y_pred_binary = (np.array(probabilities) > 0.5).astype(int)
                else:
                    y_pred_binary = (y_pred > 0.5).astype(int) if np.max(y_pred) <= 1 else y_pred.astype(int)
                    
                accuracy = accuracy_score(y_true, y_pred_binary)
                precision = precision_score(y_true, y_pred_binary, average='binary', zero_division=0)
                recall = recall_score(y_true, y_pred_binary, average='binary', zero_division=0)
                f1 = f1_score(y_true, y_pred_binary, average='binary', zero_division=0)
                
                # AUC-ROC if probabilities available
                auc_roc = None
                if probabilities is not None:
                    try:
                        auc_roc = roc_auc_score(y_true, probabilities)
                    except ValueError:
                        auc_roc = None
                        
            else:  # Multi-class or regression
                accuracy = accuracy_score(y_true, y_pred) if len(np.unique(y_true)) <= 10 else None
                precision = None
                recall = None
                f1 = None
                auc_roc = None
                
            return ModelMetrics(
                model_name=model_name,
                model_version=model_version,
                environment=environment,
                timestamp=datetime.now(),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc,
                prediction_count=len(predictions)
            )
            
        except Exception as e:
            logger.error(f"Failed to calculate batch metrics: {e}")
            return ModelMetrics(
                model_name=model_name,
                model_version=model_version,
                environment=environment,
                timestamp=datetime.now()
            )
            
    def get_model_performance_trend(self, model_name: str, environment: str,
                                  days: int = 7) -> pd.DataFrame:
        """Get model performance trend over time."""
        try:
            query = """
                SELECT 
                    model_version,
                    metric_name,
                    metric_value,
                    DATE(created_at) as metric_date,
                    AVG(metric_value) as avg_value,
                    COUNT(*) as measurement_count
                FROM monitoring.model_metrics 
                WHERE model_name = :model_name 
                AND environment = :environment
                AND created_at >= :since_date
                GROUP BY model_version, metric_name, DATE(created_at)
                ORDER BY metric_date DESC, model_version, metric_name
            """
            
            with self.db_engine.connect() as conn:
                return pd.read_sql(query, conn, params={
                    "model_name": model_name,
                    "environment": environment,
                    "since_date": datetime.now() - timedelta(days=days)
                })
                
        except Exception as e:
            logger.error(f"Failed to get performance trend: {e}")
            return pd.DataFrame()
            
    def detect_performance_degradation(self, model_name: str, model_version: str,
                                     environment: str) -> List[PerformanceAlert]:
        """Detect performance degradation and generate alerts."""
        alerts = []
        
        try:
            # Get baseline metrics
            baseline_metrics = self._get_baseline_metrics(model_name, model_version, environment)
            
            # Get current metrics
            current_metrics = self._get_current_metrics(model_name, model_version, environment)
            
            # Compare against thresholds
            for metric_name, current_value in current_metrics.items():
                baseline_value = baseline_metrics.get(metric_name)
                
                if baseline_value is not None and current_value is not None:
                    # Calculate degradation
                    if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']:
                        # Higher is better - check for decrease
                        threshold_key = f"{metric_name}_drop"
                        if threshold_key in self.alert_thresholds:
                            threshold = self.alert_thresholds[threshold_key]
                            degradation = baseline_value - current_value
                            
                            if degradation > threshold:
                                severity = self._calculate_severity(degradation, threshold)
                                alerts.append(PerformanceAlert(
                                    alert_type="performance_degradation",
                                    model_name=model_name,
                                    model_version=model_version,
                                    environment=environment,
                                    metric_name=metric_name,
                                    current_value=current_value,
                                    baseline_value=baseline_value,
                                    threshold=threshold,
                                    severity=severity,
                                    timestamp=datetime.now(),
                                    details={
                                        "degradation": degradation,
                                        "degradation_percent": (degradation / baseline_value) * 100
                                    }
                                ))
                                
            # Store alerts
            if alerts:
                self._store_performance_alerts(alerts)
                
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to detect performance degradation: {e}")
            return []
            
    def _get_baseline_metrics(self, model_name: str, model_version: str,
                            environment: str) -> Dict[str, float]:
        """Get baseline metrics for comparison."""
        cache_key = f"baseline:{model_name}:{model_version}:{environment}"
        
        # Try cache first
        cached = self.baseline_cache.get(cache_key)
        if cached:
            return cached
            
        try:
            # Get baseline from first week of model deployment
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT metric_name, AVG(metric_value) as avg_value
                    FROM monitoring.model_metrics 
                    WHERE model_name = :model_name 
                    AND model_version = :model_version
                    AND environment = :environment
                    AND created_at >= (
                        SELECT MIN(created_at) 
                        FROM monitoring.model_metrics 
                        WHERE model_name = :model_name 
                        AND model_version = :model_version
                        AND environment = :environment
                    )
                    AND created_at <= (
                        SELECT MIN(created_at) + INTERVAL '7 days'
                        FROM monitoring.model_metrics 
                        WHERE model_name = :model_name 
                        AND model_version = :model_version
                        AND environment = :environment
                    )
                    GROUP BY metric_name
                """)
                
                results = conn.execute(query, {
                    "model_name": model_name,
                    "model_version": model_version,
                    "environment": environment
                }).fetchall()
                
            baseline = {row.metric_name: float(row.avg_value) for row in results}
            
            # Cache for 1 hour
            self.baseline_cache[cache_key] = baseline
            
            return baseline
            
        except Exception as e:
            logger.error(f"Failed to get baseline metrics: {e}")
            return {}
            
    def _get_current_metrics(self, model_name: str, model_version: str,
                           environment: str, hours: int = 1) -> Dict[str, float]:
        """Get current metrics for comparison."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT metric_name, AVG(metric_value) as avg_value
                    FROM monitoring.model_metrics 
                    WHERE model_name = :model_name 
                    AND model_version = :model_version
                    AND environment = :environment
                    AND created_at >= :since
                    GROUP BY metric_name
                """)
                
                results = conn.execute(query, {
                    "model_name": model_name,
                    "model_version": model_version,
                    "environment": environment,
                    "since": datetime.now() - timedelta(hours=hours)
                }).fetchall()
                
            return {row.metric_name: float(row.avg_value) for row in results}
            
        except Exception as e:
            logger.error(f"Failed to get current metrics: {e}")
            return {}
            
    def _calculate_severity(self, degradation: float, threshold: float) -> str:
        """Calculate alert severity based on degradation magnitude."""
        ratio = degradation / threshold
        
        if ratio >= 3.0:
            return "critical"
        elif ratio >= 2.0:
            return "high"
        elif ratio >= 1.5:
            return "medium"
        else:
            return "low"
            
    def _store_performance_alerts(self, alerts: List[PerformanceAlert]) -> None:
        """Store performance alerts."""
        try:
            for alert in alerts:
                alert_data = {
                    "alert_type": alert.alert_type,
                    "model_name": alert.model_name,
                    "model_version": alert.model_version,
                    "environment": alert.environment,
                    "metric_name": alert.metric_name,
                    "current_value": alert.current_value,
                    "baseline_value": alert.baseline_value,
                    "threshold": alert.threshold,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp.isoformat(),
                    "details": alert.details
                }
                
                # Store in Redis for immediate processing
                self.redis_client.lpush("performance_alerts", json.dumps(alert_data, default=str))
                
            logger.info(f"Stored {len(alerts)} performance alerts")
            
        except Exception as e:
            logger.error(f"Failed to store performance alerts: {e}")
            
    def get_model_health_dashboard(self, model_name: str, 
                                 environment: str) -> Dict[str, Any]:
        """Get comprehensive model health dashboard data."""
        try:
            # Get latest metrics
            current_metrics = self._get_current_metrics(model_name, "latest", environment, hours=1)
            
            # Get real-time stats from Redis
            realtime_key = f"realtime_metrics:{model_name}:*:{environment}"
            realtime_stats = {}
            
            for key in self.redis_client.scan_iter(match=realtime_key):
                stats = self.redis_client.hgetall(key)
                if stats:
                    version = key.decode().split(':')[2]
                    realtime_stats[version] = {
                        k.decode(): int(v.decode()) if v.decode().isdigit() else v.decode()
                        for k, v in stats.items()
                    }
                    
            # Get recent alerts
            alert_keys = self.redis_client.lrange("performance_alerts", 0, 10)
            recent_alerts = []
            for alert_key in alert_keys:
                try:
                    alert = json.loads(alert_key.decode())
                    if alert.get("model_name") == model_name and alert.get("environment") == environment:
                        recent_alerts.append(alert)
                except json.JSONDecodeError:
                    continue
                    
            return {
                "model_name": model_name,
                "environment": environment,
                "timestamp": datetime.now().isoformat(),
                "current_metrics": current_metrics,
                "realtime_stats": realtime_stats,
                "recent_alerts": recent_alerts[:5],  # Last 5 alerts
                "health_status": self._calculate_health_status(current_metrics, recent_alerts)
            }
            
        except Exception as e:
            logger.error(f"Failed to get health dashboard: {e}")
            return {"error": str(e)}
            
    def _calculate_health_status(self, metrics: Dict[str, float], 
                               alerts: List[Dict]) -> str:
        """Calculate overall model health status."""
        # Check for critical alerts in last hour
        recent_critical = [
            a for a in alerts 
            if a.get("severity") == "critical" and
            datetime.fromisoformat(a.get("timestamp", "1970-01-01")) > 
            datetime.now() - timedelta(hours=1)
        ]
        
        if recent_critical:
            return "critical"
            
        # Check for high severity alerts
        recent_high = [
            a for a in alerts 
            if a.get("severity") == "high" and
            datetime.fromisoformat(a.get("timestamp", "1970-01-01")) > 
            datetime.now() - timedelta(hours=1)
        ]
        
        if recent_high:
            return "degraded"
            
        # Check if key metrics are available and reasonable
        if not metrics:
            return "unknown"
            
        return "healthy"