"""Advanced data drift detection and monitoring."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
# wasserstein_distance not available in this scipy version, using alternative
try:
    from scipy.spatial.distance import wasserstein_distance
except ImportError:
    # Fallback implementation
    def wasserstein_distance(u_values, v_values):
        """Simple approximation for Wasserstein distance."""
        return abs(np.mean(u_values) - np.mean(v_values))
import yaml
from pathlib import Path
import json
import pickle
from sqlalchemy import create_engine, text
import redis


logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    """Data drift detection result."""
    feature_name: str
    drift_score: float
    drift_detected: bool
    test_statistic: float
    p_value: float
    test_type: str
    reference_stats: Dict[str, float]
    current_stats: Dict[str, float]
    timestamp: datetime


class DriftDetector:
    """Advanced data drift detection system."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize drift detector."""
        self.config = self._load_config(config_path)
        self.db_engine = create_engine(self.config["database"]["connection_string"])
        self.redis_client = redis.from_url(self.config["redis"]["connection_string"])
        
        # Drift detection configuration
        self.drift_config = self.config["monitoring"]["drift_detection"]
        self.thresholds = self.drift_config["thresholds"]
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def detect_drift(self, reference_data: pd.DataFrame, 
                    current_data: pd.DataFrame,
                    feature_columns: Optional[List[str]] = None) -> Dict[str, DriftResult]:
        """Detect drift between reference and current datasets."""
        logger.info("Starting drift detection analysis")
        
        if feature_columns is None:
            # Use numerical features from config
            feature_columns = self.config["training"]["features"]["numerical_features"]
            
        drift_results = {}
        
        for feature in feature_columns:
            if feature not in reference_data.columns or feature not in current_data.columns:
                logger.warning(f"Feature {feature} not found in data, skipping")
                continue
                
            try:
                result = self._detect_feature_drift(
                    reference_data[feature], 
                    current_data[feature], 
                    feature
                )
                drift_results[feature] = result
                
                # Log result
                status = "DRIFT DETECTED" if result.drift_detected else "NO DRIFT"
                logger.info(f"{feature}: {status} (score: {result.drift_score:.4f})")
                
            except Exception as e:
                logger.error(f"Error detecting drift for feature {feature}: {e}")
                
        # Store results in database
        self._store_drift_results(list(drift_results.values()))
        
        return drift_results
        
    def _detect_feature_drift(self, reference_series: pd.Series, 
                             current_series: pd.Series, 
                             feature_name: str) -> DriftResult:
        """Detect drift for a single feature."""
        # Clean data
        ref_clean = reference_series.dropna()
        curr_clean = current_series.dropna()
        
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return DriftResult(
                feature_name=feature_name,
                drift_score=1.0,
                drift_detected=True,
                test_statistic=0.0,
                p_value=0.0,
                test_type="empty_data",
                reference_stats={},
                current_stats={},
                timestamp=datetime.now()
            )
            
        # Calculate statistics
        ref_stats = self._calculate_feature_stats(ref_clean)
        curr_stats = self._calculate_feature_stats(curr_clean)
        
        # Perform statistical tests
        drift_scores = {}
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_p_value = stats.ks_2samp(ref_clean, curr_clean)
        drift_scores['ks'] = ks_statistic
        
        # Population Stability Index
        psi_score = self._calculate_psi(ref_clean, curr_clean)
        drift_scores['psi'] = psi_score
        
        # Wasserstein distance
        wasserstein_dist = wasserstein_distance(ref_clean, curr_clean)
        # Normalize by reference std
        normalized_wasserstein = wasserstein_dist / (ref_stats['std'] + 1e-8)
        drift_scores['wasserstein'] = normalized_wasserstein
        
        # Jensen-Shannon divergence
        js_div = self._calculate_js_divergence(ref_clean, curr_clean)
        drift_scores['js_divergence'] = js_div
        
        # Energy distance
        energy_dist = self._calculate_energy_distance(ref_clean, curr_clean)
        drift_scores['energy_distance'] = energy_dist
        
        # Determine overall drift based on multiple tests
        drift_detected = self._is_drift_detected(drift_scores)
        
        # Use KS statistic as primary drift score
        primary_score = ks_statistic
        
        return DriftResult(
            feature_name=feature_name,
            drift_score=primary_score,
            drift_detected=drift_detected,
            test_statistic=ks_statistic,
            p_value=ks_p_value,
            test_type="ks_test",
            reference_stats=ref_stats,
            current_stats=curr_stats,
            timestamp=datetime.now()
        )
        
    def _calculate_feature_stats(self, series: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive statistics for a feature."""
        return {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'median': float(series.median()),
            'q25': float(series.quantile(0.25)),
            'q75': float(series.quantile(0.75)),
            'skewness': float(series.skew()),
            'kurtosis': float(series.kurtosis()),
            'count': len(series)
        }
        
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, 
                      buckets: int = 10) -> float:
        """Calculate Population Stability Index."""
        try:
            # Create bins based on reference data quantiles
            _, bin_edges = np.histogram(reference, bins=buckets)
            
            # Ensure we have proper bin edges
            if len(bin_edges) < 2:
                return float('inf')
                
            # Get counts for each bin
            ref_counts = np.histogram(reference, bins=bin_edges)[0]
            current_counts = np.histogram(current, bins=bin_edges)[0]
            
            # Calculate percentages
            ref_pct = ref_counts / len(reference)
            current_pct = current_counts / len(current)
            
            # Avoid division by zero and log(0)
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            current_pct = np.where(current_pct == 0, 0.0001, current_pct)
            
            # Calculate PSI
            psi = np.sum((current_pct - ref_pct) * np.log(current_pct / ref_pct))
            
            return float(psi)
            
        except Exception as e:
            logger.error(f"PSI calculation failed: {e}")
            return float('inf')
            
    def _calculate_js_divergence(self, reference: pd.Series, current: pd.Series,
                                bins: int = 50) -> float:
        """Calculate Jensen-Shannon divergence."""
        try:
            # Create common bin edges
            combined_data = np.concatenate([reference, current])
            bin_edges = np.histogram_bin_edges(combined_data, bins=bins)
            
            # Get normalized histograms
            ref_hist = np.histogram(reference, bins=bin_edges)[0]
            curr_hist = np.histogram(current, bins=bin_edges)[0]
            
            # Normalize to probabilities
            ref_prob = ref_hist / np.sum(ref_hist)
            curr_prob = curr_hist / np.sum(curr_hist)
            
            # Avoid log(0)
            ref_prob = np.where(ref_prob == 0, 1e-8, ref_prob)
            curr_prob = np.where(curr_prob == 0, 1e-8, curr_prob)
            
            # Calculate JS divergence
            m = 0.5 * (ref_prob + curr_prob)
            js_div = 0.5 * stats.entropy(ref_prob, m) + 0.5 * stats.entropy(curr_prob, m)
            
            return float(js_div)
            
        except Exception as e:
            logger.error(f"JS divergence calculation failed: {e}")
            return float('inf')
            
    def _calculate_energy_distance(self, reference: pd.Series, current: pd.Series) -> float:
        """Calculate energy distance between distributions."""
        try:
            from scipy.stats import energy_distance
            return float(energy_distance(reference, current))
        except ImportError:
            # Fallback implementation
            ref_vals = reference.values
            curr_vals = current.values
            
            # Calculate pairwise distances
            ref_ref = np.mean([abs(x - y) for x in ref_vals for y in ref_vals])
            curr_curr = np.mean([abs(x - y) for x in curr_vals for y in curr_vals])
            ref_curr = np.mean([abs(x - y) for x in ref_vals for y in curr_vals])
            
            energy_dist = 2 * ref_curr - ref_ref - curr_curr
            return float(energy_dist)
        except Exception as e:
            logger.error(f"Energy distance calculation failed: {e}")
            return float('inf')
            
    def _is_drift_detected(self, drift_scores: Dict[str, float]) -> bool:
        """Determine if drift is detected based on multiple test thresholds."""
        # Get thresholds
        thresholds = self.thresholds
        
        # Check each test against its threshold
        tests_results = []
        
        if 'ks' in drift_scores and 'ks' in thresholds:
            tests_results.append(drift_scores['ks'] > thresholds['ks'])
            
        if 'psi' in drift_scores and 'psi' in thresholds:
            tests_results.append(drift_scores['psi'] > thresholds['psi'])
            
        if 'wasserstein' in drift_scores and 'wasserstein' in thresholds:
            tests_results.append(drift_scores['wasserstein'] > thresholds['wasserstein'])
            
        if 'js_divergence' in drift_scores:
            # JS divergence threshold (typically 0.1-0.3)
            tests_results.append(drift_scores['js_divergence'] > 0.1)
            
        # Drift detected if any test exceeds threshold
        return any(tests_results) if tests_results else False
        
    def _store_drift_results(self, drift_results: List[DriftResult]) -> None:
        """Store drift detection results in database."""
        try:
            with self.db_engine.connect() as conn:
                for result in drift_results:
                    query = text("""
                        INSERT INTO monitoring.data_drift 
                        (feature_name, drift_score, drift_detected, 
                         reference_period_start, reference_period_end,
                         current_period_start, current_period_end, created_at)
                        VALUES 
                        (:feature_name, :drift_score, :drift_detected,
                         :ref_start, :ref_end, :curr_start, :curr_end, :created_at)
                    """)
                    
                    # For this example, using dummy periods
                    now = datetime.now()
                    ref_start = now - timedelta(days=30)
                    ref_end = now - timedelta(days=7)
                    curr_start = now - timedelta(days=7)
                    
                    conn.execute(query, {
                        "feature_name": result.feature_name,
                        "drift_score": result.drift_score,
                        "drift_detected": result.drift_detected,
                        "ref_start": ref_start,
                        "ref_end": ref_end,
                        "curr_start": curr_start,
                        "curr_end": now,
                        "created_at": result.timestamp
                    })
                    
                conn.commit()
                logger.info(f"Stored {len(drift_results)} drift results in database")
                
        except Exception as e:
            logger.error(f"Failed to store drift results: {e}")
            
    def get_drift_history(self, feature_name: Optional[str] = None, 
                         days: int = 30) -> pd.DataFrame:
        """Get drift detection history."""
        try:
            query = """
                SELECT feature_name, drift_score, drift_detected, created_at
                FROM monitoring.data_drift 
                WHERE created_at >= :since_date
            """
            
            params = {"since_date": datetime.now() - timedelta(days=days)}
            
            if feature_name:
                query += " AND feature_name = :feature_name"
                params["feature_name"] = feature_name
                
            query += " ORDER BY created_at DESC"
            
            with self.db_engine.connect() as conn:
                return pd.read_sql(query, conn, params=params)
                
        except Exception as e:
            logger.error(f"Failed to get drift history: {e}")
            return pd.DataFrame()
            
    def get_drift_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get drift detection summary."""
        try:
            query = text("""
                SELECT 
                    feature_name,
                    COUNT(*) as total_checks,
                    SUM(CASE WHEN drift_detected THEN 1 ELSE 0 END) as drift_count,
                    AVG(drift_score) as avg_drift_score,
                    MAX(drift_score) as max_drift_score,
                    MAX(created_at) as last_check
                FROM monitoring.data_drift 
                WHERE created_at >= :since_date
                GROUP BY feature_name
                ORDER BY drift_count DESC, max_drift_score DESC
            """)
            
            with self.db_engine.connect() as conn:
                results = conn.execute(query, {
                    "since_date": datetime.now() - timedelta(days=days)
                }).fetchall()
                
            summary = {
                "period_days": days,
                "total_features_monitored": len(results),
                "features_with_drift": sum(1 for r in results if r.drift_count > 0),
                "features": []
            }
            
            for row in results:
                summary["features"].append({
                    "feature_name": row.feature_name,
                    "total_checks": row.total_checks,
                    "drift_count": row.drift_count,
                    "drift_rate": row.drift_count / row.total_checks if row.total_checks > 0 else 0,
                    "avg_drift_score": float(row.avg_drift_score) if row.avg_drift_score else 0,
                    "max_drift_score": float(row.max_drift_score) if row.max_drift_score else 0,
                    "last_check": row.last_check.isoformat() if row.last_check else None
                })
                
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get drift summary: {e}")
            return {"error": str(e)}
            
    def create_drift_alert(self, drift_results: Dict[str, DriftResult]) -> None:
        """Create alerts for detected drift."""
        drifted_features = [
            name for name, result in drift_results.items() 
            if result.drift_detected
        ]
        
        if not drifted_features:
            logger.info("No drift detected, no alerts created")
            return
            
        # Create alert payload
        alert = {
            "alert_type": "data_drift_detected",
            "timestamp": datetime.now().isoformat(),
            "affected_features": drifted_features,
            "feature_details": {
                name: {
                    "drift_score": result.drift_score,
                    "test_type": result.test_type,
                    "p_value": result.p_value
                }
                for name, result in drift_results.items()
                if result.drift_detected
            },
            "severity": self._calculate_alert_severity(drift_results)
        }
        
        # Store alert for processing
        try:
            self.redis_client.lpush("drift_alerts", json.dumps(alert, default=str))
            logger.info(f"Created drift alert for {len(drifted_features)} features")
        except Exception as e:
            logger.error(f"Failed to create drift alert: {e}")
            
    def _calculate_alert_severity(self, drift_results: Dict[str, DriftResult]) -> str:
        """Calculate alert severity based on drift scores."""
        max_score = max(
            result.drift_score for result in drift_results.values()
            if result.drift_detected
        )
        
        if max_score > 0.5:
            return "high"
        elif max_score > 0.2:
            return "medium"
        else:
            return "low"