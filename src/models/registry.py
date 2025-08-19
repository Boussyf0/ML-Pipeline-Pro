"""MLflow model registry management."""
import logging
from typing import Dict, Any, Optional, List
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import yaml
from pathlib import Path


logger = logging.getLogger(__name__)


class ModelRegistry:
    """MLflow model registry manager."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model registry."""
        self.config = self._load_config(config_path)
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        self.client = MlflowClient()
        self.registered_model_name = self.config["mlflow"]["registered_model_name"]
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def register_model(self, run_id: str, model_path: str, 
                      model_name: Optional[str] = None,
                      description: Optional[str] = None,
                      tags: Optional[Dict[str, str]] = None) -> str:
        """Register a model in MLflow model registry."""
        if model_name is None:
            model_name = self.registered_model_name
            
        try:
            # Create registered model if it doesn't exist
            try:
                self.client.get_registered_model(model_name)
            except MlflowException:
                self.client.create_registered_model(
                    name=model_name,
                    description=description or f"Registered model for {model_name}"
                )
                
            # Register model version
            model_uri = f"runs:/{run_id}/{model_path}"
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags=tags
            )
            
            logger.info(f"Model registered: {model_name} version {model_version.version}")
            return model_version.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
            
    def promote_model(self, model_name: str, version: str, stage: str) -> None:
        """Promote model to a specific stage (Staging, Production)."""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {model_name} version {version} promoted to {stage}")
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            raise
            
    def get_model_version(self, model_name: str, stage: str) -> Optional[str]:
        """Get model version for a specific stage."""
        try:
            versions = self.client.get_latest_versions(
                name=model_name,
                stages=[stage]
            )
            if versions:
                return versions[0].version
            return None
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None
            
    def load_model(self, model_name: str, version: Optional[str] = None, 
                   stage: Optional[str] = None):
        """Load model from registry."""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                # Get latest version
                latest_version = self.get_latest_version(model_name)
                if not latest_version:
                    raise ValueError(f"No versions found for model {model_name}")
                model_uri = f"models:/{model_name}/{latest_version}"
                
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
            
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            if versions:
                return str(max([int(v.version) for v in versions]))
            return None
        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return None
            
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        try:
            models = self.client.search_registered_models()
            return [
                {
                    "name": model.name,
                    "description": model.description,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "status": v.status
                        } for v in model.latest_versions
                    ]
                }
                for model in models
            ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
            
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a registered model."""
        try:
            model = self.client.get_registered_model(model_name)
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            return {
                "name": model.name,
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status,
                        "creation_timestamp": v.creation_timestamp,
                        "run_id": v.run_id,
                        "description": v.description
                    }
                    for v in sorted(versions, key=lambda x: int(x.version), reverse=True)
                ]
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
            
    def delete_model_version(self, model_name: str, version: str) -> None:
        """Delete a specific model version."""
        try:
            self.client.delete_model_version(
                name=model_name,
                version=version
            )
            logger.info(f"Deleted model {model_name} version {version}")
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            raise
            
    def archive_model(self, model_name: str, version: str) -> None:
        """Archive a model version."""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Archived"
            )
            logger.info(f"Archived model {model_name} version {version}")
        except Exception as e:
            logger.error(f"Failed to archive model: {e}")
            raise
            
    def add_model_description(self, model_name: str, version: str, 
                            description: str) -> None:
        """Add description to a model version."""
        try:
            self.client.update_model_version(
                name=model_name,
                version=version,
                description=description
            )
            logger.info(f"Updated description for {model_name} version {version}")
        except Exception as e:
            logger.error(f"Failed to update model description: {e}")
            raise
            
    def compare_models(self, model_name: str, version1: str, 
                      version2: str) -> Dict[str, Any]:
        """Compare two model versions."""
        try:
            # Get run information for both versions
            version1_info = self.client.get_model_version(model_name, version1)
            version2_info = self.client.get_model_version(model_name, version2)
            
            # Get metrics for both runs
            run1_metrics = self.client.get_run(version1_info.run_id).data.metrics
            run2_metrics = self.client.get_run(version2_info.run_id).data.metrics
            
            comparison = {
                "model_name": model_name,
                "version1": {
                    "version": version1,
                    "stage": version1_info.current_stage,
                    "metrics": run1_metrics
                },
                "version2": {
                    "version": version2,
                    "stage": version2_info.current_stage,
                    "metrics": run2_metrics
                },
                "metrics_comparison": {}
            }
            
            # Compare common metrics
            common_metrics = set(run1_metrics.keys()) & set(run2_metrics.keys())
            for metric in common_metrics:
                comparison["metrics_comparison"][metric] = {
                    "version1": run1_metrics[metric],
                    "version2": run2_metrics[metric],
                    "difference": run2_metrics[metric] - run1_metrics[metric],
                    "improvement": run2_metrics[metric] > run1_metrics[metric]
                }
                
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise