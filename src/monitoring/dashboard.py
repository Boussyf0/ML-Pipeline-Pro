"""MLOps monitoring dashboard backend."""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import json
import yaml
from sqlalchemy import create_engine, text
import redis
from dataclasses import asdict

from .drift_detector import DriftDetector
from .model_monitor import ModelMonitor
from .alerting import AlertManager


logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Comprehensive MLOps monitoring dashboard."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize monitoring dashboard."""
        self.config = self._load_config(config_path)
        self.db_engine = create_engine(self.config["database"]["connection_string"])
        self.redis_client = redis.from_url(self.config["redis"]["connection_string"])
        
        # Initialize monitoring components
        self.drift_detector = DriftDetector(config_path)
        self.model_monitor = ModelMonitor(config_path)
        self.alert_manager = AlertManager(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def get_overview_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive overview dashboard data."""
        try:
            logger.info("Generating overview dashboard")
            
            # Get model health summary
            model_health = self._get_models_health_summary()
            
            # Get drift summary
            drift_summary = self.drift_detector.get_drift_summary(days=7)
            
            # Get alert summary
            alert_summary = self.alert_manager.get_alert_summary(days=7)
            
            # Get performance metrics summary
            performance_summary = self._get_performance_summary()
            
            # Get system health
            system_health = self._get_system_health()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "model_health": model_health,
                "drift_summary": drift_summary,
                "alert_summary": alert_summary,
                "performance_summary": performance_summary,
                "system_health": system_health,
                "dashboard_status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"Failed to generate overview dashboard: {e}")
            return {
                "error": str(e),
                "dashboard_status": "error",
                "timestamp": datetime.now().isoformat()
            }
            
    def get_model_dashboard(self, model_name: str, environment: str = "production") -> Dict[str, Any]:
        """Get detailed dashboard for a specific model."""
        try:
            logger.info(f"Generating model dashboard for {model_name} in {environment}")
            
            # Get model health
            model_health = self.model_monitor.get_model_health_dashboard(model_name, environment)
            
            # Get performance trend
            performance_trend = self.model_monitor.get_model_performance_trend(
                model_name, environment, days=30
            )
            
            # Get drift history
            drift_history = self.drift_detector.get_drift_history(
                feature_name=None, days=30
            )
            
            # Get recent predictions stats
            prediction_stats = self._get_prediction_stats(model_name, environment, hours=24)
            
            # Get active alerts for this model
            active_alerts = self.alert_manager.get_active_alerts(
                model_name=model_name, environment=environment
            )
            
            # Get A/B testing status if applicable
            ab_test_status = self._get_ab_test_status(model_name, environment)
            
            return {
                "model_name": model_name,
                "environment": environment,
                "timestamp": datetime.now().isoformat(),
                "model_health": model_health,
                "performance_trend": performance_trend.to_dict("records") if not performance_trend.empty else [],
                "drift_history": drift_history.to_dict("records") if not drift_history.empty else [],
                "prediction_stats": prediction_stats,
                "active_alerts": active_alerts,
                "ab_test_status": ab_test_status
            }
            
        except Exception as e:
            logger.error(f"Failed to generate model dashboard: {e}")
            return {
                "error": str(e),
                "model_name": model_name,
                "environment": environment,
                "timestamp": datetime.now().isoformat()
            }
            
    def get_drift_dashboard(self, days: int = 30) -> Dict[str, Any]:
        """Get drift monitoring dashboard."""
        try:
            logger.info(f"Generating drift dashboard for {days} days")
            
            # Get drift summary
            drift_summary = self.drift_detector.get_drift_summary(days=days)
            
            # Get detailed drift history
            drift_history = self.drift_detector.get_drift_history(days=days)
            
            # Process drift history for visualization
            drift_viz_data = self._process_drift_for_visualization(drift_history)
            
            # Get drift alerts
            drift_alerts = self.alert_manager.get_active_alerts(severity=None)
            drift_alerts = [a for a in drift_alerts if a.get("alert_type") == "data_drift_detected"]
            
            return {
                "period_days": days,
                "timestamp": datetime.now().isoformat(),
                "drift_summary": drift_summary,
                "drift_history": drift_history.to_dict("records") if not drift_history.empty else [],
                "drift_visualization": drift_viz_data,
                "drift_alerts": drift_alerts
            }
            
        except Exception as e:
            logger.error(f"Failed to generate drift dashboard: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def get_performance_dashboard(self, days: int = 30) -> Dict[str, Any]:
        """Get performance monitoring dashboard."""
        try:
            logger.info(f"Generating performance dashboard for {days} days")
            
            # Get performance metrics for all models
            performance_data = self._get_all_models_performance(days)
            
            # Get performance alerts
            perf_alerts = self.alert_manager.get_active_alerts(severity=None)
            perf_alerts = [a for a in perf_alerts if a.get("alert_type") == "performance_degradation"]
            
            # Get throughput and latency trends
            throughput_data = self._get_throughput_trends(days)
            latency_data = self._get_latency_trends(days)
            
            return {
                "period_days": days,
                "timestamp": datetime.now().isoformat(),
                "performance_data": performance_data,
                "performance_alerts": perf_alerts,
                "throughput_trends": throughput_data,
                "latency_trends": latency_data
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance dashboard: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def get_alerts_dashboard(self, days: int = 7) -> Dict[str, Any]:
        """Get alerts dashboard."""
        try:
            logger.info(f"Generating alerts dashboard for {days} days")
            
            # Get alert summary
            alert_summary = self.alert_manager.get_alert_summary(days)
            
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            
            # Get alert trends
            alert_trends = self._get_alert_trends(days)
            
            return {
                "period_days": days,
                "timestamp": datetime.now().isoformat(),
                "alert_summary": alert_summary,
                "active_alerts": active_alerts,
                "alert_trends": alert_trends
            }
            
        except Exception as e:
            logger.error(f"Failed to generate alerts dashboard: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    def _get_models_health_summary(self) -> Dict[str, Any]:
        """Get summary of all models' health."""
        try:
            # This would typically query a models registry or configuration
            models = ["churn-predictor"]  # In practice, get from config or database
            environments = ["production", "staging"]
            
            health_summary = {
                "total_models": 0,
                "healthy_models": 0,
                "degraded_models": 0,
                "critical_models": 0,
                "models": []
            }
            
            for model_name in models:
                for environment in environments:
                    health = self.model_monitor.get_model_health_dashboard(model_name, environment)
                    status = health.get("health_status", "unknown")
                    
                    model_info = {
                        "model_name": model_name,
                        "environment": environment,
                        "status": status,
                        "last_prediction": health.get("timestamp"),
                        "recent_alerts": len(health.get("recent_alerts", []))
                    }
                    
                    health_summary["models"].append(model_info)
                    health_summary["total_models"] += 1
                    
                    if status == "healthy":
                        health_summary["healthy_models"] += 1
                    elif status == "degraded":
                        health_summary["degraded_models"] += 1
                    elif status == "critical":
                        health_summary["critical_models"] += 1
                        
            return health_summary
            
        except Exception as e:
            logger.error(f"Failed to get models health summary: {e}")
            return {"error": str(e)}
            
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT 
                        model_name,
                        environment,
                        metric_name,
                        AVG(metric_value) as avg_value,
                        COUNT(*) as measurement_count
                    FROM monitoring.model_metrics 
                    WHERE created_at >= :since
                    GROUP BY model_name, environment, metric_name
                    ORDER BY model_name, environment, metric_name
                """)
                
                results = conn.execute(query, {
                    "since": datetime.now() - timedelta(days=1)
                }).fetchall()
                
            summary = {
                "total_measurements": sum(r.measurement_count for r in results),
                "models": {}
            }
            
            for row in results:
                model_key = f"{row.model_name}:{row.environment}"
                if model_key not in summary["models"]:
                    summary["models"][model_key] = {}
                    
                summary["models"][model_key][row.metric_name] = {
                    "avg_value": float(row.avg_value),
                    "measurement_count": row.measurement_count
                }
                
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}
            
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health status."""
        try:
            health = {
                "database": "unknown",
                "redis": "unknown",
                "mlflow": "unknown",
                "overall": "unknown"
            }
            
            # Test database connection
            try:
                with self.db_engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                health["database"] = "healthy"
            except Exception:
                health["database"] = "unhealthy"
                
            # Test Redis connection
            try:
                self.redis_client.ping()
                health["redis"] = "healthy"
            except Exception:
                health["redis"] = "unhealthy"
                
            # Test MLflow (would need actual MLflow client)
            health["mlflow"] = "healthy"  # Placeholder
            
            # Overall health
            if all(status == "healthy" for status in health.values() if status != "unknown"):
                health["overall"] = "healthy"
            elif any(status == "unhealthy" for status in health.values()):
                health["overall"] = "unhealthy"
            else:
                health["overall"] = "unknown"
                
            return health
            
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {"error": str(e)}
            
    def _get_prediction_stats(self, model_name: str, environment: str, 
                            hours: int) -> Dict[str, Any]:
        """Get prediction statistics for a model."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT 
                        COUNT(*) as total_predictions,
                        AVG(response_time_ms) as avg_response_time,
                        MIN(response_time_ms) as min_response_time,
                        MAX(response_time_ms) as max_response_time,
                        AVG(prediction) as avg_prediction,
                        STDDEV(prediction) as std_prediction
                    FROM app_data.prediction_logs 
                    WHERE model_name = :model_name
                    AND created_at >= :since
                """)
                
                result = conn.execute(query, {
                    "model_name": model_name,
                    "since": datetime.now() - timedelta(hours=hours)
                }).fetchone()
                
            if result:
                return {
                    "total_predictions": result.total_predictions or 0,
                    "avg_response_time_ms": float(result.avg_response_time) if result.avg_response_time else 0,
                    "min_response_time_ms": float(result.min_response_time) if result.min_response_time else 0,
                    "max_response_time_ms": float(result.max_response_time) if result.max_response_time else 0,
                    "avg_prediction": float(result.avg_prediction) if result.avg_prediction else 0,
                    "std_prediction": float(result.std_prediction) if result.std_prediction else 0,
                    "predictions_per_hour": (result.total_predictions or 0) / hours
                }
            else:
                return {"total_predictions": 0}
                
        except Exception as e:
            logger.error(f"Failed to get prediction stats: {e}")
            return {"error": str(e)}
            
    def _get_ab_test_status(self, model_name: str, environment: str) -> Optional[Dict[str, Any]]:
        """Get A/B test status for a model."""
        try:
            # Check Redis for active A/B tests
            for key in self.redis_client.scan_iter(match="ab_test:*"):
                ab_config = json.loads(self.redis_client.get(key))
                
                if (ab_config.get("model_a", {}).get("name") == model_name or
                    ab_config.get("model_b", {}).get("name") == model_name):
                    
                    return {
                        "experiment_name": ab_config.get("experiment_name"),
                        "status": ab_config.get("status"),
                        "start_time": ab_config.get("start_time"),
                        "model_a": ab_config.get("model_a"),
                        "model_b": ab_config.get("model_b"),
                        "traffic_split": {
                            "model_a": ab_config.get("model_a", {}).get("traffic_ratio", 0),
                            "model_b": ab_config.get("model_b", {}).get("traffic_ratio", 0)
                        }
                    }
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to get A/B test status: {e}")
            return None
            
    def _process_drift_for_visualization(self, drift_history: pd.DataFrame) -> Dict[str, Any]:
        """Process drift history data for visualization."""
        if drift_history.empty:
            return {}
            
        try:
            viz_data = {
                "timeline": [],
                "feature_scores": {},
                "drift_rate_by_feature": {}
            }
            
            # Group by date for timeline
            if 'created_at' in drift_history.columns:
                daily_drift = drift_history.groupby([
                    drift_history['created_at'].dt.date, 'feature_name'
                ]).agg({
                    'drift_detected': 'sum',
                    'drift_score': 'mean'
                }).reset_index()
                
                for _, row in daily_drift.iterrows():
                    viz_data["timeline"].append({
                        "date": row['created_at'].isoformat(),
                        "feature": row['feature_name'],
                        "drift_detected": int(row['drift_detected']),
                        "avg_drift_score": float(row['drift_score'])
                    })
                    
            # Feature-wise drift rates
            feature_stats = drift_history.groupby('feature_name').agg({
                'drift_detected': ['sum', 'count'],
                'drift_score': ['mean', 'max']
            }).round(4)
            
            for feature in feature_stats.index:
                viz_data["drift_rate_by_feature"][feature] = {
                    "drift_count": int(feature_stats.loc[feature, ('drift_detected', 'sum')]),
                    "total_checks": int(feature_stats.loc[feature, ('drift_detected', 'count')]),
                    "drift_rate": float(feature_stats.loc[feature, ('drift_detected', 'sum')] / 
                                       feature_stats.loc[feature, ('drift_detected', 'count')]),
                    "avg_score": float(feature_stats.loc[feature, ('drift_score', 'mean')]),
                    "max_score": float(feature_stats.loc[feature, ('drift_score', 'max')])
                }
                
            return viz_data
            
        except Exception as e:
            logger.error(f"Failed to process drift visualization data: {e}")
            return {}
            
    def _get_all_models_performance(self, days: int) -> List[Dict[str, Any]]:
        """Get performance data for all models."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT 
                        model_name,
                        model_version,
                        environment,
                        metric_name,
                        AVG(metric_value) as avg_value,
                        MIN(metric_value) as min_value,
                        MAX(metric_value) as max_value,
                        COUNT(*) as measurement_count
                    FROM monitoring.model_metrics 
                    WHERE created_at >= :since
                    GROUP BY model_name, model_version, environment, metric_name
                    ORDER BY model_name, model_version, metric_name
                """)
                
                results = conn.execute(query, {
                    "since": datetime.now() - timedelta(days=days)
                }).fetchall()
                
            performance_data = []
            for row in results:
                performance_data.append({
                    "model_name": row.model_name,
                    "model_version": row.model_version,
                    "environment": row.environment,
                    "metric_name": row.metric_name,
                    "avg_value": float(row.avg_value),
                    "min_value": float(row.min_value),
                    "max_value": float(row.max_value),
                    "measurement_count": row.measurement_count
                })
                
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to get all models performance: {e}")
            return []
            
    def _get_throughput_trends(self, days: int) -> List[Dict[str, Any]]:
        """Get throughput trends data."""
        try:
            # This would typically aggregate from prediction logs
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT 
                        model_name,
                        environment,
                        DATE(created_at) as date,
                        COUNT(*) as daily_predictions
                    FROM app_data.prediction_logs 
                    WHERE created_at >= :since
                    GROUP BY model_name, environment, DATE(created_at)
                    ORDER BY date
                """)
                
                results = conn.execute(query, {
                    "since": datetime.now() - timedelta(days=days)
                }).fetchall()
                
            return [
                {
                    "model_name": row.model_name,
                    "environment": row.environment,
                    "date": row.date.isoformat(),
                    "daily_predictions": row.daily_predictions
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get throughput trends: {e}")
            return []
            
    def _get_latency_trends(self, days: int) -> List[Dict[str, Any]]:
        """Get latency trends data."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT 
                        model_name,
                        environment,
                        DATE(created_at) as date,
                        AVG(response_time_ms) as avg_latency,
                        MIN(response_time_ms) as min_latency,
                        MAX(response_time_ms) as max_latency,
                        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_latency
                    FROM app_data.prediction_logs 
                    WHERE created_at >= :since
                    AND response_time_ms IS NOT NULL
                    GROUP BY model_name, environment, DATE(created_at)
                    ORDER BY date
                """)
                
                results = conn.execute(query, {
                    "since": datetime.now() - timedelta(days=days)
                }).fetchall()
                
            return [
                {
                    "model_name": row.model_name,
                    "environment": row.environment,
                    "date": row.date.isoformat(),
                    "avg_latency_ms": float(row.avg_latency),
                    "min_latency_ms": float(row.min_latency),
                    "max_latency_ms": float(row.max_latency),
                    "p95_latency_ms": float(row.p95_latency)
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get latency trends: {e}")
            return []
            
    def _get_alert_trends(self, days: int) -> List[Dict[str, Any]]:
        """Get alert trends over time."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT 
                        DATE(timestamp) as date,
                        severity,
                        alert_type,
                        COUNT(*) as alert_count
                    FROM monitoring.alerts 
                    WHERE timestamp >= :since
                    GROUP BY DATE(timestamp), severity, alert_type
                    ORDER BY date
                """)
                
                results = conn.execute(query, {
                    "since": datetime.now() - timedelta(days=days)
                }).fetchall()
                
            return [
                {
                    "date": row.date.isoformat(),
                    "severity": row.severity,
                    "alert_type": row.alert_type,
                    "alert_count": row.alert_count
                }
                for row in results
            ]
            
        except Exception as e:
            logger.error(f"Failed to get alert trends: {e}")
            return []