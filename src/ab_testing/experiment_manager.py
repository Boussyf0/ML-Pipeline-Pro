"""A/B testing experiment management system."""
import logging
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import redis
from sqlalchemy import create_engine, text
import yaml

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment status enumeration."""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TrafficAllocation(Enum):
    """Traffic allocation methods."""
    RANDOM = "random"
    USER_HASH = "user_hash"
    FEATURE_HASH = "feature_hash"
    GEOGRAPHIC = "geographic"


@dataclass
class ExperimentConfig:
    """A/B test experiment configuration."""
    experiment_id: str
    name: str
    description: str
    model_a_name: str
    model_a_version: str
    model_b_name: str
    model_b_version: str
    traffic_split: float  # Percentage for model A (0.0-1.0)
    allocation_method: TrafficAllocation
    start_date: datetime
    end_date: Optional[datetime]
    minimum_sample_size: int
    significance_level: float
    success_metrics: List[str]
    status: ExperimentStatus
    created_by: str
    metadata: Dict[str, Any]


@dataclass
class ExperimentResult:
    """A/B test results."""
    experiment_id: str
    model_a_metrics: Dict[str, float]
    model_b_metrics: Dict[str, float]
    sample_size_a: int
    sample_size_b: int
    statistical_tests: Dict[str, Dict[str, float]]
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]]
    effect_sizes: Dict[str, float]
    statistical_significance: Dict[str, bool]
    practical_significance: Dict[str, bool]
    winner: Optional[str]  # 'A', 'B', or None
    recommendation: str
    test_duration_days: int
    analyzed_at: datetime


class ExperimentManager:
    """Manages A/B testing experiments for ML models."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize experiment manager."""
        self.config = self._load_config(config_path)
        self.db_engine = create_engine(self.config["database"]["connection_string"])
        self.redis_client = redis.from_url(self.config["redis"]["connection_string"])
        
        # A/B testing configuration
        self.ab_config = self.config.get("ab_testing", {})
        self.default_significance_level = self.ab_config.get("significance_level", 0.05)
        self.default_minimum_sample_size = self.ab_config.get("minimum_sample_size", 1000)
        
        # Initialize database tables
        self._init_database_tables()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def _init_database_tables(self) -> None:
        """Initialize database tables for A/B testing."""
        try:
            with self.db_engine.connect() as conn:
                # Experiments table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ab_testing.experiments (
                        experiment_id VARCHAR(255) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        model_a_name VARCHAR(255) NOT NULL,
                        model_a_version VARCHAR(50) NOT NULL,
                        model_b_name VARCHAR(255) NOT NULL,
                        model_b_version VARCHAR(50) NOT NULL,
                        traffic_split FLOAT NOT NULL,
                        allocation_method VARCHAR(50) NOT NULL,
                        start_date TIMESTAMP NOT NULL,
                        end_date TIMESTAMP,
                        minimum_sample_size INTEGER NOT NULL,
                        significance_level FLOAT NOT NULL,
                        success_metrics JSONB,
                        status VARCHAR(50) NOT NULL,
                        created_by VARCHAR(255),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                
                # Experiment assignments table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ab_testing.experiment_assignments (
                        id SERIAL PRIMARY KEY,
                        experiment_id VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255),
                        session_id VARCHAR(255),
                        assignment VARCHAR(10) NOT NULL, -- 'A' or 'B'
                        assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB,
                        FOREIGN KEY (experiment_id) REFERENCES ab_testing.experiments(experiment_id)
                    )
                """))
                
                # Experiment events table (predictions, conversions, etc.)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ab_testing.experiment_events (
                        id SERIAL PRIMARY KEY,
                        experiment_id VARCHAR(255) NOT NULL,
                        user_id VARCHAR(255),
                        session_id VARCHAR(255),
                        assignment VARCHAR(10) NOT NULL,
                        event_type VARCHAR(100) NOT NULL, -- 'prediction', 'conversion', 'click', etc.
                        event_value FLOAT,
                        model_name VARCHAR(255),
                        model_version VARCHAR(50),
                        prediction FLOAT,
                        prediction_proba FLOAT,
                        response_time_ms INTEGER,
                        event_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (experiment_id) REFERENCES ab_testing.experiments(experiment_id)
                    )
                """))
                
                # Experiment results table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ab_testing.experiment_results (
                        experiment_id VARCHAR(255) PRIMARY KEY,
                        model_a_metrics JSONB,
                        model_b_metrics JSONB,
                        sample_size_a INTEGER,
                        sample_size_b INTEGER,
                        statistical_tests JSONB,
                        confidence_intervals JSONB,
                        effect_sizes JSONB,
                        statistical_significance JSONB,
                        practical_significance JSONB,
                        winner VARCHAR(10),
                        recommendation TEXT,
                        test_duration_days INTEGER,
                        analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (experiment_id) REFERENCES ab_testing.experiments(experiment_id)
                    )
                """))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize A/B testing tables: {e}")
            
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B testing experiment."""
        try:
            # Generate experiment ID if not provided
            if not config.experiment_id:
                config.experiment_id = str(uuid.uuid4())
                
            # Validate configuration
            self._validate_experiment_config(config)
            
            # Store experiment in database
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO ab_testing.experiments 
                    (experiment_id, name, description, model_a_name, model_a_version,
                     model_b_name, model_b_version, traffic_split, allocation_method,
                     start_date, end_date, minimum_sample_size, significance_level,
                     success_metrics, status, created_by, metadata)
                    VALUES 
                    (:experiment_id, :name, :description, :model_a_name, :model_a_version,
                     :model_b_name, :model_b_version, :traffic_split, :allocation_method,
                     :start_date, :end_date, :minimum_sample_size, :significance_level,
                     :success_metrics, :status, :created_by, :metadata)
                """)
                
                conn.execute(query, {
                    "experiment_id": config.experiment_id,
                    "name": config.name,
                    "description": config.description,
                    "model_a_name": config.model_a_name,
                    "model_a_version": config.model_a_version,
                    "model_b_name": config.model_b_name,
                    "model_b_version": config.model_b_version,
                    "traffic_split": config.traffic_split,
                    "allocation_method": config.allocation_method.value,
                    "start_date": config.start_date,
                    "end_date": config.end_date,
                    "minimum_sample_size": config.minimum_sample_size,
                    "significance_level": config.significance_level,
                    "success_metrics": json.dumps(config.success_metrics),
                    "status": config.status.value,
                    "created_by": config.created_by,
                    "metadata": json.dumps(config.metadata)
                })
                conn.commit()
                
            # Cache active experiment for quick access
            if config.status == ExperimentStatus.ACTIVE:
                self._cache_active_experiment(config)
                
            logger.info(f"Created experiment: {config.experiment_id}")
            return config.experiment_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            raise
            
    def _validate_experiment_config(self, config: ExperimentConfig) -> None:
        """Validate experiment configuration."""
        if not 0 <= config.traffic_split <= 1:
            raise ValueError("Traffic split must be between 0 and 1")
            
        if config.minimum_sample_size < 100:
            raise ValueError("Minimum sample size must be at least 100")
            
        if not 0.01 <= config.significance_level <= 0.1:
            raise ValueError("Significance level must be between 0.01 and 0.1")
            
        if config.end_date and config.end_date <= config.start_date:
            raise ValueError("End date must be after start date")
            
    def _cache_active_experiment(self, config: ExperimentConfig) -> None:
        """Cache active experiment in Redis for fast lookup."""
        try:
            cache_key = f"experiment:active:{config.model_a_name}:{config.model_b_name}"
            experiment_data = asdict(config)
            experiment_data["start_date"] = config.start_date.isoformat()
            experiment_data["end_date"] = config.end_date.isoformat() if config.end_date else None
            experiment_data["allocation_method"] = config.allocation_method.value
            experiment_data["status"] = config.status.value
            
            self.redis_client.setex(
                cache_key,
                3600 * 24,  # 24 hours
                json.dumps(experiment_data, default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to cache experiment: {e}")
            
    def get_assignment(self, experiment_id: str, user_id: Optional[str] = None,
                      session_id: Optional[str] = None, 
                      features: Optional[Dict[str, Any]] = None) -> str:
        """Get assignment (A or B) for a user in an experiment."""
        try:
            # Get experiment configuration
            experiment = self.get_experiment(experiment_id)
            if not experiment or experiment.status != ExperimentStatus.ACTIVE:
                raise ValueError(f"Experiment {experiment_id} is not active")
                
            # Check if user already has assignment
            existing_assignment = self._get_existing_assignment(experiment_id, user_id, session_id)
            if existing_assignment:
                return existing_assignment
                
            # Generate new assignment based on allocation method
            assignment = self._generate_assignment(experiment, user_id, session_id, features)
            
            # Store assignment
            self._store_assignment(experiment_id, user_id, session_id, assignment)
            
            return assignment
            
        except Exception as e:
            logger.error(f"Failed to get assignment: {e}")
            # Default to A in case of errors
            return "A"
            
    def _get_existing_assignment(self, experiment_id: str, user_id: Optional[str],
                               session_id: Optional[str]) -> Optional[str]:
        """Get existing assignment for user/session."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT assignment FROM ab_testing.experiment_assignments
                    WHERE experiment_id = :experiment_id
                    AND (user_id = :user_id OR session_id = :session_id)
                    ORDER BY assigned_at DESC
                    LIMIT 1
                """)
                
                result = conn.execute(query, {
                    "experiment_id": experiment_id,
                    "user_id": user_id,
                    "session_id": session_id
                }).fetchone()
                
                return result.assignment if result else None
                
        except Exception as e:
            logger.error(f"Failed to get existing assignment: {e}")
            return None
            
    def _generate_assignment(self, experiment: ExperimentConfig, 
                           user_id: Optional[str], session_id: Optional[str],
                           features: Optional[Dict[str, Any]]) -> str:
        """Generate assignment based on allocation method."""
        if experiment.allocation_method == TrafficAllocation.RANDOM:
            return "A" if np.random.random() < experiment.traffic_split else "B"
            
        elif experiment.allocation_method == TrafficAllocation.USER_HASH:
            if user_id:
                hash_input = f"{experiment.experiment_id}:{user_id}"
                hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                return "A" if (hash_value % 100) / 100 < experiment.traffic_split else "B"
            else:
                # Fall back to random if no user_id
                return "A" if np.random.random() < experiment.traffic_split else "B"
                
        elif experiment.allocation_method == TrafficAllocation.FEATURE_HASH:
            if features:
                feature_str = json.dumps(features, sort_keys=True)
                hash_input = f"{experiment.experiment_id}:{feature_str}"
                hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
                return "A" if (hash_value % 100) / 100 < experiment.traffic_split else "B"
            else:
                # Fall back to random if no features
                return "A" if np.random.random() < experiment.traffic_split else "B"
                
        else:
            # Default to random
            return "A" if np.random.random() < experiment.traffic_split else "B"
            
    def _store_assignment(self, experiment_id: str, user_id: Optional[str],
                         session_id: Optional[str], assignment: str) -> None:
        """Store assignment in database."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO ab_testing.experiment_assignments
                    (experiment_id, user_id, session_id, assignment)
                    VALUES (:experiment_id, :user_id, :session_id, :assignment)
                """)
                
                conn.execute(query, {
                    "experiment_id": experiment_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "assignment": assignment
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store assignment: {e}")
            
    def log_event(self, experiment_id: str, user_id: Optional[str], 
                 session_id: Optional[str], assignment: str,
                 event_type: str, event_value: Optional[float] = None,
                 model_name: Optional[str] = None, model_version: Optional[str] = None,
                 prediction: Optional[float] = None, prediction_proba: Optional[float] = None,
                 response_time_ms: Optional[int] = None,
                 event_data: Optional[Dict[str, Any]] = None) -> None:
        """Log an event for the experiment."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO ab_testing.experiment_events
                    (experiment_id, user_id, session_id, assignment, event_type,
                     event_value, model_name, model_version, prediction, prediction_proba,
                     response_time_ms, event_data)
                    VALUES 
                    (:experiment_id, :user_id, :session_id, :assignment, :event_type,
                     :event_value, :model_name, :model_version, :prediction, :prediction_proba,
                     :response_time_ms, :event_data)
                """)
                
                conn.execute(query, {
                    "experiment_id": experiment_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "assignment": assignment,
                    "event_type": event_type,
                    "event_value": event_value,
                    "model_name": model_name,
                    "model_version": model_version,
                    "prediction": prediction,
                    "prediction_proba": prediction_proba,
                    "response_time_ms": response_time_ms,
                    "event_data": json.dumps(event_data) if event_data else None
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
            
    def analyze_experiment(self, experiment_id: str) -> ExperimentResult:
        """Analyze experiment results and perform statistical tests."""
        try:
            experiment = self.get_experiment(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")
                
            # Get experiment data
            events_df = self._get_experiment_events(experiment_id)
            
            if events_df.empty:
                raise ValueError("No experiment data available for analysis")
                
            # Calculate metrics for each group
            model_a_metrics = self._calculate_group_metrics(
                events_df[events_df['assignment'] == 'A'], experiment.success_metrics
            )
            model_b_metrics = self._calculate_group_metrics(
                events_df[events_df['assignment'] == 'B'], experiment.success_metrics
            )
            
            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(
                events_df, experiment.success_metrics, experiment.significance_level
            )
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                events_df, experiment.success_metrics, experiment.significance_level
            )
            
            # Calculate effect sizes
            effect_sizes = self._calculate_effect_sizes(events_df, experiment.success_metrics)
            
            # Determine statistical and practical significance
            statistical_significance = {
                metric: test_results.get("p_value", 1.0) < experiment.significance_level
                for metric, test_results in statistical_tests.items()
            }
            
            practical_significance = self._assess_practical_significance(
                effect_sizes, model_a_metrics, model_b_metrics
            )
            
            # Determine winner
            winner = self._determine_winner(
                model_a_metrics, model_b_metrics, statistical_significance, 
                practical_significance, experiment.success_metrics
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                winner, statistical_significance, practical_significance,
                len(events_df[events_df['assignment'] == 'A']),
                len(events_df[events_df['assignment'] == 'B']),
                experiment.minimum_sample_size
            )
            
            # Create result object
            result = ExperimentResult(
                experiment_id=experiment_id,
                model_a_metrics=model_a_metrics,
                model_b_metrics=model_b_metrics,
                sample_size_a=len(events_df[events_df['assignment'] == 'A']),
                sample_size_b=len(events_df[events_df['assignment'] == 'B']),
                statistical_tests=statistical_tests,
                confidence_intervals=confidence_intervals,
                effect_sizes=effect_sizes,
                statistical_significance=statistical_significance,
                practical_significance=practical_significance,
                winner=winner,
                recommendation=recommendation,
                test_duration_days=(datetime.now() - experiment.start_date).days,
                analyzed_at=datetime.now()
            )
            
            # Store results
            self._store_experiment_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze experiment: {e}")
            raise
            
    def _get_experiment_events(self, experiment_id: str) -> pd.DataFrame:
        """Get experiment events as DataFrame."""
        with self.db_engine.connect() as conn:
            query = """
                SELECT * FROM ab_testing.experiment_events
                WHERE experiment_id = :experiment_id
                ORDER BY created_at
            """
            return pd.read_sql(query, conn, params={"experiment_id": experiment_id})
            
    def _calculate_group_metrics(self, group_df: pd.DataFrame, 
                                success_metrics: List[str]) -> Dict[str, float]:
        """Calculate metrics for a group (A or B)."""
        metrics = {}
        
        if not group_df.empty:
            # Basic metrics
            metrics["sample_size"] = len(group_df)
            metrics["conversion_rate"] = len(group_df[group_df["event_type"] == "conversion"]) / len(group_df)
            
            # Prediction metrics
            prediction_events = group_df[group_df["event_type"] == "prediction"]
            if not prediction_events.empty:
                metrics["avg_prediction"] = prediction_events["prediction"].mean()
                metrics["avg_prediction_proba"] = prediction_events["prediction_proba"].mean()
                metrics["avg_response_time_ms"] = prediction_events["response_time_ms"].mean()
                
            # Custom metrics based on success_metrics configuration
            for metric in success_metrics:
                if metric == "accuracy":
                    # This would require actual labels - placeholder calculation
                    metrics["accuracy"] = np.random.uniform(0.8, 0.9)  # Placeholder
                elif metric == "click_through_rate":
                    clicks = len(group_df[group_df["event_type"] == "click"])
                    impressions = len(group_df[group_df["event_type"] == "impression"])
                    metrics["click_through_rate"] = clicks / impressions if impressions > 0 else 0
                    
        return metrics
        
    def _perform_statistical_tests(self, events_df: pd.DataFrame, 
                                 success_metrics: List[str],
                                 significance_level: float) -> Dict[str, Dict[str, float]]:
        """Perform statistical tests between groups."""
        tests = {}
        
        group_a = events_df[events_df['assignment'] == 'A']
        group_b = events_df[events_df['assignment'] == 'B']
        
        # Conversion rate test (proportions)
        conversions_a = len(group_a[group_a["event_type"] == "conversion"])
        conversions_b = len(group_b[group_b["event_type"] == "conversion"])
        total_a = len(group_a)
        total_b = len(group_b)
        
        if total_a > 0 and total_b > 0:
            # Two-proportion z-test
            p_a = conversions_a / total_a
            p_b = conversions_b / total_b
            p_pooled = (conversions_a + conversions_b) / (total_a + total_b)
            
            if 0 < p_pooled < 1:
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/total_a + 1/total_b))
                z_score = (p_a - p_b) / se if se > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                tests["conversion_rate"] = {
                    "test_type": "two_proportion_z_test",
                    "statistic": z_score,
                    "p_value": p_value,
                    "significant": p_value < significance_level
                }
                
        # Response time test (continuous)
        response_times_a = group_a[group_a["event_type"] == "prediction"]["response_time_ms"].dropna()
        response_times_b = group_b[group_b["event_type"] == "prediction"]["response_time_ms"].dropna()
        
        if len(response_times_a) > 10 and len(response_times_b) > 10:
            t_stat, p_value = stats.ttest_ind(response_times_a, response_times_b)
            
            tests["response_time"] = {
                "test_type": "independent_t_test",
                "statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < significance_level
            }
            
        return tests
        
    def _calculate_confidence_intervals(self, events_df: pd.DataFrame,
                                      success_metrics: List[str],
                                      significance_level: float) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate confidence intervals for metrics."""
        alpha = significance_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        intervals = {}
        
        group_a = events_df[events_df['assignment'] == 'A']
        group_b = events_df[events_df['assignment'] == 'B']
        
        # Conversion rate confidence intervals
        for group_name, group_data in [("A", group_a), ("B", group_b)]:
            conversions = len(group_data[group_data["event_type"] == "conversion"])
            total = len(group_data)
            
            if total > 0:
                p = conversions / total
                se = np.sqrt(p * (1 - p) / total)
                margin_of_error = z_score * se
                
                if "conversion_rate" not in intervals:
                    intervals["conversion_rate"] = {}
                    
                intervals["conversion_rate"][group_name] = (
                    max(0, p - margin_of_error),
                    min(1, p + margin_of_error)
                )
                
        return intervals
        
    def _calculate_effect_sizes(self, events_df: pd.DataFrame,
                              success_metrics: List[str]) -> Dict[str, float]:
        """Calculate effect sizes (Cohen's d, etc.)."""
        effect_sizes = {}
        
        group_a = events_df[events_df['assignment'] == 'A']
        group_b = events_df[events_df['assignment'] == 'B']
        
        # Response time effect size
        response_times_a = group_a[group_a["event_type"] == "prediction"]["response_time_ms"].dropna()
        response_times_b = group_b[group_b["event_type"] == "prediction"]["response_time_ms"].dropna()
        
        if len(response_times_a) > 0 and len(response_times_b) > 0:
            mean_a = response_times_a.mean()
            mean_b = response_times_b.mean()
            std_pooled = np.sqrt(((len(response_times_a) - 1) * response_times_a.var() +
                                 (len(response_times_b) - 1) * response_times_b.var()) /
                                (len(response_times_a) + len(response_times_b) - 2))
            
            if std_pooled > 0:
                cohens_d = (mean_b - mean_a) / std_pooled
                effect_sizes["response_time"] = float(cohens_d)
                
        return effect_sizes
        
    def _assess_practical_significance(self, effect_sizes: Dict[str, float],
                                     model_a_metrics: Dict[str, float],
                                     model_b_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Assess practical significance of results."""
        practical_sig = {}
        
        # Define minimum practical effect thresholds
        thresholds = {
            "conversion_rate": 0.01,  # 1 percentage point
            "response_time": 50,      # 50ms
            "accuracy": 0.02          # 2 percentage points
        }
        
        for metric in model_a_metrics:
            if metric in model_b_metrics:
                difference = abs(model_b_metrics[metric] - model_a_metrics[metric])
                threshold = thresholds.get(metric, 0.05)  # Default 5% improvement
                practical_sig[metric] = difference >= threshold
                
        return practical_sig
        
    def _determine_winner(self, model_a_metrics: Dict[str, float],
                         model_b_metrics: Dict[str, float],
                         statistical_significance: Dict[str, bool],
                         practical_significance: Dict[str, bool],
                         success_metrics: List[str]) -> Optional[str]:
        """Determine the winning model."""
        if not statistical_significance or not practical_significance:
            return None
            
        # Primary success metric (first in the list)
        primary_metric = success_metrics[0] if success_metrics else "conversion_rate"
        
        if (primary_metric in statistical_significance and 
            statistical_significance[primary_metric] and
            primary_metric in practical_significance and
            practical_significance[primary_metric]):
            
            if (primary_metric in model_a_metrics and 
                primary_metric in model_b_metrics):
                
                if model_b_metrics[primary_metric] > model_a_metrics[primary_metric]:
                    return "B"
                else:
                    return "A"
                    
        return None  # No clear winner
        
    def _generate_recommendation(self, winner: Optional[str],
                               statistical_significance: Dict[str, bool],
                               practical_significance: Dict[str, bool],
                               sample_size_a: int, sample_size_b: int,
                               minimum_sample_size: int) -> str:
        """Generate recommendation based on results."""
        if sample_size_a < minimum_sample_size or sample_size_b < minimum_sample_size:
            return f"Continue test - need at least {minimum_sample_size} samples per group"
            
        if winner:
            return f"Deploy Model {winner} - statistically and practically significant improvement"
        elif any(statistical_significance.values()):
            return "Results are statistically significant but may not be practically significant"
        else:
            return "No significant difference detected - either model can be used"
            
    def _store_experiment_results(self, result: ExperimentResult) -> None:
        """Store experiment results in database."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    INSERT INTO ab_testing.experiment_results
                    (experiment_id, model_a_metrics, model_b_metrics, sample_size_a, sample_size_b,
                     statistical_tests, confidence_intervals, effect_sizes, statistical_significance,
                     practical_significance, winner, recommendation, test_duration_days)
                    VALUES
                    (:experiment_id, :model_a_metrics, :model_b_metrics, :sample_size_a, :sample_size_b,
                     :statistical_tests, :confidence_intervals, :effect_sizes, :statistical_significance,
                     :practical_significance, :winner, :recommendation, :test_duration_days)
                    ON CONFLICT (experiment_id) DO UPDATE SET
                        model_a_metrics = EXCLUDED.model_a_metrics,
                        model_b_metrics = EXCLUDED.model_b_metrics,
                        sample_size_a = EXCLUDED.sample_size_a,
                        sample_size_b = EXCLUDED.sample_size_b,
                        statistical_tests = EXCLUDED.statistical_tests,
                        confidence_intervals = EXCLUDED.confidence_intervals,
                        effect_sizes = EXCLUDED.effect_sizes,
                        statistical_significance = EXCLUDED.statistical_significance,
                        practical_significance = EXCLUDED.practical_significance,
                        winner = EXCLUDED.winner,
                        recommendation = EXCLUDED.recommendation,
                        test_duration_days = EXCLUDED.test_duration_days,
                        analyzed_at = CURRENT_TIMESTAMP
                """)
                
                conn.execute(query, {
                    "experiment_id": result.experiment_id,
                    "model_a_metrics": json.dumps(result.model_a_metrics),
                    "model_b_metrics": json.dumps(result.model_b_metrics),
                    "sample_size_a": result.sample_size_a,
                    "sample_size_b": result.sample_size_b,
                    "statistical_tests": json.dumps(result.statistical_tests),
                    "confidence_intervals": json.dumps(result.confidence_intervals),
                    "effect_sizes": json.dumps(result.effect_sizes),
                    "statistical_significance": json.dumps(result.statistical_significance),
                    "practical_significance": json.dumps(result.practical_significance),
                    "winner": result.winner,
                    "recommendation": result.recommendation,
                    "test_duration_days": result.test_duration_days
                })
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to store experiment results: {e}")
            
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration by ID."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    SELECT * FROM ab_testing.experiments
                    WHERE experiment_id = :experiment_id
                """)
                
                result = conn.execute(query, {"experiment_id": experiment_id}).fetchone()
                
                if result:
                    return ExperimentConfig(
                        experiment_id=result.experiment_id,
                        name=result.name,
                        description=result.description,
                        model_a_name=result.model_a_name,
                        model_a_version=result.model_a_version,
                        model_b_name=result.model_b_name,
                        model_b_version=result.model_b_version,
                        traffic_split=result.traffic_split,
                        allocation_method=TrafficAllocation(result.allocation_method),
                        start_date=result.start_date,
                        end_date=result.end_date,
                        minimum_sample_size=result.minimum_sample_size,
                        significance_level=result.significance_level,
                        success_metrics=json.loads(result.success_metrics) if result.success_metrics else [],
                        status=ExperimentStatus(result.status),
                        created_by=result.created_by,
                        metadata=json.loads(result.metadata) if result.metadata else {}
                    )
                    
                return None
                
        except Exception as e:
            logger.error(f"Failed to get experiment: {e}")
            return None
            
    def list_experiments(self, status: Optional[ExperimentStatus] = None) -> List[ExperimentConfig]:
        """List experiments, optionally filtered by status."""
        try:
            with self.db_engine.connect() as conn:
                query = "SELECT * FROM ab_testing.experiments"
                params = {}
                
                if status:
                    query += " WHERE status = :status"
                    params["status"] = status.value
                    
                query += " ORDER BY created_at DESC"
                
                results = conn.execute(text(query), params).fetchall()
                
                experiments = []
                for result in results:
                    experiment = ExperimentConfig(
                        experiment_id=result.experiment_id,
                        name=result.name,
                        description=result.description,
                        model_a_name=result.model_a_name,
                        model_a_version=result.model_a_version,
                        model_b_name=result.model_b_name,
                        model_b_version=result.model_b_version,
                        traffic_split=result.traffic_split,
                        allocation_method=TrafficAllocation(result.allocation_method),
                        start_date=result.start_date,
                        end_date=result.end_date,
                        minimum_sample_size=result.minimum_sample_size,
                        significance_level=result.significance_level,
                        success_metrics=json.loads(result.success_metrics) if result.success_metrics else [],
                        status=ExperimentStatus(result.status),
                        created_by=result.created_by,
                        metadata=json.loads(result.metadata) if result.metadata else {}
                    )
                    experiments.append(experiment)
                    
                return experiments
                
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
            
    def update_experiment_status(self, experiment_id: str, status: ExperimentStatus) -> bool:
        """Update experiment status."""
        try:
            with self.db_engine.connect() as conn:
                query = text("""
                    UPDATE ab_testing.experiments
                    SET status = :status, updated_at = CURRENT_TIMESTAMP
                    WHERE experiment_id = :experiment_id
                """)
                
                result = conn.execute(query, {
                    "experiment_id": experiment_id,
                    "status": status.value
                })
                conn.commit()
                
                return result.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to update experiment status: {e}")
            return False