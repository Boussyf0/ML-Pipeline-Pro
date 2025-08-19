"""Traffic splitting and routing for A/B testing."""
import logging
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import redis
import yaml
from .experiment_manager import ExperimentManager, ExperimentStatus

logger = logging.getLogger(__name__)


@dataclass
class TrafficSplit:
    """Traffic split configuration."""
    experiment_id: str
    model_a: str
    model_a_version: str
    model_b: str 
    model_b_version: str
    split_ratio: float  # Percentage going to model A
    allocation_key: str  # user_id, session_id, or feature_hash


class TrafficSplitter:
    """Handles traffic splitting for A/B testing experiments."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize traffic splitter."""
        self.config = self._load_config(config_path)
        self.redis_client = redis.from_url(self.config["redis"]["connection_string"])
        self.experiment_manager = ExperimentManager(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def get_model_assignment(self, model_name: str, user_id: Optional[str] = None,
                           session_id: Optional[str] = None,
                           features: Optional[Dict[str, Any]] = None) -> Tuple[str, str, Optional[str]]:
        """
        Get model assignment for a request.
        
        Returns:
            Tuple of (model_name, model_version, experiment_group)
        """
        try:
            # Check for active experiments involving this model
            active_experiment = self._get_active_experiment(model_name)
            
            if not active_experiment:
                # No active experiment, return production model
                return self._get_production_model(model_name)
                
            # Get assignment for the experiment
            assignment = self.experiment_manager.get_assignment(
                active_experiment.experiment_id, user_id, session_id, features
            )
            
            if assignment == "A":
                return (
                    active_experiment.model_a_name,
                    active_experiment.model_a_version,
                    "A"
                )
            else:
                return (
                    active_experiment.model_b_name,
                    active_experiment.model_b_version,
                    "B"
                )
                
        except Exception as e:
            logger.error(f"Error in model assignment: {e}")
            # Fall back to production model
            return self._get_production_model(model_name)
            
    def _get_active_experiment(self, model_name: str):
        """Get active experiment for a model."""
        try:
            # Check Redis cache first
            cache_key = f"active_experiments:{model_name}"
            cached_experiment = self.redis_client.get(cache_key)
            
            if cached_experiment:
                experiment_id = cached_experiment.decode()
                return self.experiment_manager.get_experiment(experiment_id)
                
            # Query database for active experiments
            experiments = self.experiment_manager.list_experiments(ExperimentStatus.ACTIVE)
            
            for experiment in experiments:
                if (experiment.model_a_name == model_name or 
                    experiment.model_b_name == model_name):
                    
                    # Cache for quick lookup
                    self.redis_client.setex(cache_key, 300, experiment.experiment_id)  # 5 min cache
                    return experiment
                    
            return None
            
        except Exception as e:
            logger.error(f"Error getting active experiment: {e}")
            return None
            
    def _get_production_model(self, model_name: str) -> Tuple[str, str, Optional[str]]:
        """Get production model version."""
        # This would typically query MLflow registry for production version
        # For now, return a default
        return (model_name, "latest", None)
        
    def setup_traffic_split(self, experiment_id: str, model_a: str, model_a_version: str,
                          model_b: str, model_b_version: str, split_ratio: float) -> bool:
        """Setup traffic splitting for an experiment."""
        try:
            traffic_split = TrafficSplit(
                experiment_id=experiment_id,
                model_a=model_a,
                model_a_version=model_a_version,
                model_b=model_b,
                model_b_version=model_b_version,
                split_ratio=split_ratio,
                allocation_key="user_hash"  # Default allocation method
            )
            
            # Store in Redis for quick access
            cache_key = f"traffic_split:{experiment_id}"
            self.redis_client.setex(
                cache_key,
                86400,  # 24 hours
                json.dumps({
                    "experiment_id": traffic_split.experiment_id,
                    "model_a": traffic_split.model_a,
                    "model_a_version": traffic_split.model_a_version,
                    "model_b": traffic_split.model_b,
                    "model_b_version": traffic_split.model_b_version,
                    "split_ratio": traffic_split.split_ratio,
                    "allocation_key": traffic_split.allocation_key
                })
            )
            
            logger.info(f"Setup traffic split for experiment {experiment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup traffic split: {e}")
            return False
            
    def get_assignment_with_sticky_session(self, experiment_id: str, 
                                         user_id: Optional[str] = None,
                                         session_id: Optional[str] = None) -> str:
        """Get assignment with sticky session support."""
        try:
            # Create a consistent identifier
            identifier = user_id or session_id or "anonymous"
            
            # Check if assignment already exists
            assignment_key = f"assignment:{experiment_id}:{identifier}"
            existing_assignment = self.redis_client.get(assignment_key)
            
            if existing_assignment:
                return existing_assignment.decode()
                
            # Get experiment configuration
            experiment = self.experiment_manager.get_experiment(experiment_id)
            if not experiment:
                return "A"  # Default fallback
                
            # Generate new assignment
            assignment = self._generate_consistent_assignment(
                experiment.experiment_id, identifier, experiment.traffic_split
            )
            
            # Store assignment with expiry (experiment duration or 30 days)
            expiry = 86400 * 30  # 30 days
            if experiment.end_date:
                expiry = min(expiry, int((experiment.end_date - datetime.now()).total_seconds()))
                
            self.redis_client.setex(assignment_key, expiry, assignment)
            
            return assignment
            
        except Exception as e:
            logger.error(f"Error getting sticky assignment: {e}")
            return "A"  # Default fallback
            
    def _generate_consistent_assignment(self, experiment_id: str, 
                                      identifier: str, traffic_split: float) -> str:
        """Generate consistent assignment based on hash."""
        # Create hash input
        hash_input = f"{experiment_id}:{identifier}"
        
        # Generate hash
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        
        # Convert to number between 0 and 1
        hash_number = int(hash_value, 16) % 10000 / 10000.0
        
        # Assign based on traffic split
        return "A" if hash_number < traffic_split else "B"
        
    def get_traffic_allocation(self, experiment_id: str) -> Dict[str, int]:
        """Get current traffic allocation for an experiment."""
        try:
            # Get assignment counts from Redis
            assignment_pattern = f"assignment:{experiment_id}:*"
            assignment_keys = self.redis_client.keys(assignment_pattern)
            
            allocation = {"A": 0, "B": 0}
            
            for key in assignment_keys:
                assignment = self.redis_client.get(key)
                if assignment:
                    assignment_value = assignment.decode()
                    if assignment_value in allocation:
                        allocation[assignment_value] += 1
                        
            return allocation
            
        except Exception as e:
            logger.error(f"Error getting traffic allocation: {e}")
            return {"A": 0, "B": 0}
            
    def update_traffic_split(self, experiment_id: str, new_split_ratio: float) -> bool:
        """Update traffic split ratio for an experiment."""
        try:
            # Update experiment in database
            experiment = self.experiment_manager.get_experiment(experiment_id)
            if not experiment:
                return False
                
            experiment.traffic_split = new_split_ratio
            
            # Update cache
            cache_key = f"traffic_split:{experiment_id}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data.decode())
                data["split_ratio"] = new_split_ratio
                self.redis_client.setex(cache_key, 86400, json.dumps(data))
                
            logger.info(f"Updated traffic split for experiment {experiment_id} to {new_split_ratio}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update traffic split: {e}")
            return False
            
    def pause_experiment(self, experiment_id: str) -> bool:
        """Pause an experiment and route all traffic to control."""
        try:
            # Update experiment status
            success = self.experiment_manager.update_experiment_status(
                experiment_id, ExperimentStatus.PAUSED
            )
            
            if success:
                # Clear cache to force re-evaluation
                self.redis_client.delete(f"traffic_split:{experiment_id}")
                
                # Set temporary routing to control (model A)
                routing_key = f"experiment_routing:{experiment_id}"
                self.redis_client.setex(routing_key, 86400, "A")  # Route all to A
                
                logger.info(f"Paused experiment {experiment_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to pause experiment: {e}")
            return False
            
    def resume_experiment(self, experiment_id: str) -> bool:
        """Resume a paused experiment."""
        try:
            # Update experiment status
            success = self.experiment_manager.update_experiment_status(
                experiment_id, ExperimentStatus.ACTIVE
            )
            
            if success:
                # Clear forced routing
                self.redis_client.delete(f"experiment_routing:{experiment_id}")
                
                logger.info(f"Resumed experiment {experiment_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to resume experiment: {e}")
            return False
            
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get comprehensive experiment status."""
        try:
            experiment = self.experiment_manager.get_experiment(experiment_id)
            if not experiment:
                return {"error": "Experiment not found"}
                
            # Get traffic allocation
            allocation = self.get_traffic_allocation(experiment_id)
            
            # Get total assignments
            total_assignments = sum(allocation.values())
            
            # Calculate actual split ratio
            actual_split_a = allocation["A"] / total_assignments if total_assignments > 0 else 0
            actual_split_b = allocation["B"] / total_assignments if total_assignments > 0 else 0
            
            return {
                "experiment_id": experiment_id,
                "status": experiment.status.value,
                "configured_split": experiment.traffic_split,
                "actual_split": {
                    "A": actual_split_a,
                    "B": actual_split_b
                },
                "total_assignments": total_assignments,
                "allocation": allocation,
                "models": {
                    "A": f"{experiment.model_a_name}:{experiment.model_a_version}",
                    "B": f"{experiment.model_b_name}:{experiment.model_b_version}"
                },
                "duration_days": (datetime.now() - experiment.start_date).days,
                "is_paused": experiment.status == ExperimentStatus.PAUSED
            }
            
        except Exception as e:
            logger.error(f"Error getting experiment status: {e}")
            return {"error": str(e)}
            
    def cleanup_expired_assignments(self) -> int:
        """Clean up expired assignments and return count of cleaned items."""
        try:
            cleaned_count = 0
            
            # Get all assignment keys
            assignment_pattern = "assignment:*"
            assignment_keys = self.redis_client.keys(assignment_pattern)
            
            for key in assignment_keys:
                # Check if key has TTL
                ttl = self.redis_client.ttl(key)
                if ttl == -1:  # No expiration set
                    # Set default expiration of 30 days
                    self.redis_client.expire(key, 86400 * 30)
                elif ttl == -2:  # Key doesn't exist or expired
                    cleaned_count += 1
                    
            logger.info(f"Cleaned up {cleaned_count} expired assignments")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up assignments: {e}")
            return 0