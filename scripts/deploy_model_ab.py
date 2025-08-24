#!/usr/bin/env python3
"""
A/B Testing Model Deployment Script
Deploys models with A/B testing configuration for production evaluation.
"""

import argparse
import logging
import sys
import json
import time
from typing import Dict, Any, Optional
import requests
import mlflow
from mlflow.tracking import MlflowClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ABTestModelDeployer:
    """A/B testing model deployment manager."""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.client = MlflowClient()
        
    def get_model_info(self, model_name: str, model_version: str = "latest") -> Optional[Dict[str, Any]]:
        """Get model information from MLflow registry."""
        try:
            if model_version == "latest":
                model_version = self.client.get_latest_versions(
                    model_name, 
                    stages=["Production"]
                )[0].version
                
            model_info = self.client.get_model_version(model_name, model_version)
            
            return {
                "name": model_info.name,
                "version": model_info.version,
                "stage": model_info.current_stage,
                "run_id": model_info.run_id,
                "model_uri": f"models:/{model_name}/{model_version}",
                "creation_timestamp": model_info.creation_timestamp,
                "last_updated_timestamp": model_info.last_updated_timestamp
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
            
    def create_ab_test_config(self, model_name: str, model_version: str, 
                             traffic_split: float, experiment_duration: int = 604800) -> Dict[str, Any]:
        """Create A/B test configuration."""
        config = {
            "experiment_id": f"ab_test_{model_name}_{model_version}_{int(time.time())}",
            "model_name": model_name,
            "control_model": {
                "version": "current",
                "traffic_percentage": round((1.0 - traffic_split) * 100, 1)
            },
            "treatment_model": {
                "version": model_version,
                "traffic_percentage": round(traffic_split * 100, 1)
            },
            "start_time": int(time.time()),
            "end_time": int(time.time()) + experiment_duration,
            "duration_hours": experiment_duration // 3600,
            "environment": self.environment,
            "status": "active",
            "success_metrics": [
                "prediction_accuracy",
                "response_time",
                "error_rate"
            ],
            "traffic_routing": {
                "method": "user_id_hash",  # Consistent user assignment
                "hash_salt": f"{model_name}_{model_version}_salt"
            }
        }
        
        return config
        
    def deploy_model_version(self, model_name: str, model_version: str) -> bool:
        """Deploy specific model version to staging area."""
        logger.info(f"Deploying model {model_name} version {model_version}")
        
        try:
            # Get model info
            model_info = self.get_model_info(model_name, model_version)
            if not model_info:
                logger.error("Failed to get model information")
                return False
                
            # Download model artifacts
            model_uri = model_info["model_uri"]
            logger.info(f"Downloading model from: {model_uri}")
            
            # In production, this would download the model to a shared storage
            # accessible by the API servers
            model_path = f"/models/{model_name}/versions/{model_version}"
            
            # Create deployment manifest for the new model version
            deployment_info = {
                "model_name": model_name,
                "model_version": model_version,
                "model_path": model_path,
                "deployment_time": time.time(),
                "status": "deployed"
            }
            
            # Save deployment info
            deployment_file = f"/tmp/{model_name}_v{model_version}_deployment.json"
            with open(deployment_file, 'w') as f:
                json.dump(deployment_info, f, indent=2)
                
            logger.info(f"Model deployment info saved to {deployment_file}")
            return True
            
        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return False
            
    def configure_traffic_split(self, config: Dict[str, Any]) -> bool:
        """Configure traffic splitting for A/B test."""
        logger.info(f"Configuring traffic split: {config['treatment_model']['traffic_percentage']}% to treatment")
        
        try:
            # This would typically configure a load balancer or service mesh
            # For this example, we'll save the configuration
            
            traffic_config = {
                "experiment_id": config["experiment_id"],
                "routing_rules": [
                    {
                        "condition": f"hash(user_id + '{config['traffic_routing']['hash_salt']}') % 100 < {config['treatment_model']['traffic_percentage']}",
                        "model_version": config["treatment_model"]["version"],
                        "percentage": config["treatment_model"]["traffic_percentage"]
                    },
                    {
                        "condition": "default",
                        "model_version": config["control_model"]["version"],
                        "percentage": config["control_model"]["traffic_percentage"]
                    }
                ],
                "created_at": time.time()
            }
            
            # Save traffic configuration
            traffic_file = f"/tmp/ab_test_traffic_config_{config['experiment_id']}.json"
            with open(traffic_file, 'w') as f:
                json.dump(traffic_config, f, indent=2)
                
            logger.info(f"Traffic configuration saved to {traffic_file}")
            
            # In production, this would update the actual routing configuration
            # via API calls to your service mesh, load balancer, or feature flag system
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to configure traffic split: {e}")
            return False
            
    def setup_monitoring(self, config: Dict[str, Any]) -> bool:
        """Setup monitoring for A/B test."""
        logger.info("Setting up A/B test monitoring")
        
        try:
            monitoring_config = {
                "experiment_id": config["experiment_id"],
                "metrics_to_track": [
                    {
                        "name": "prediction_latency",
                        "type": "histogram",
                        "labels": ["model_version", "endpoint"]
                    },
                    {
                        "name": "prediction_accuracy",
                        "type": "gauge",
                        "labels": ["model_version"]
                    },
                    {
                        "name": "error_rate",
                        "type": "counter",
                        "labels": ["model_version", "error_type"]
                    },
                    {
                        "name": "user_satisfaction",
                        "type": "histogram",
                        "labels": ["model_version"]
                    }
                ],
                "alert_rules": [
                    {
                        "name": "high_error_rate",
                        "condition": "error_rate > 0.05",
                        "action": "pause_experiment"
                    },
                    {
                        "name": "high_latency",
                        "condition": "avg(prediction_latency) > 2.0",
                        "action": "alert"
                    },
                    {
                        "name": "low_accuracy",
                        "condition": "prediction_accuracy < 0.70",
                        "action": "pause_experiment"
                    }
                ],
                "dashboard_url": f"https://monitoring.{self.environment}.mlops-pipeline.com/ab-tests/{config['experiment_id']}"
            }
            
            # Save monitoring configuration
            monitoring_file = f"/tmp/ab_test_monitoring_{config['experiment_id']}.json"
            with open(monitoring_file, 'w') as f:
                json.dump(monitoring_config, f, indent=2)
                
            logger.info(f"Monitoring configuration saved to {monitoring_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
            return False
            
    def validate_deployment(self, model_name: str, model_version: str) -> bool:
        """Validate model deployment."""
        logger.info(f"Validating deployment of {model_name} version {model_version}")
        
        try:
            # Check if model endpoints are healthy
            health_checks = [
                self.check_model_health(model_version),
                self.check_prediction_endpoint(model_version),
                self.check_metrics_collection(model_version)
            ]
            
            all_healthy = all(health_checks)
            
            if all_healthy:
                logger.info("All deployment validation checks passed")
            else:
                logger.error("Some deployment validation checks failed")
                
            return all_healthy
            
        except Exception as e:
            logger.error(f"Deployment validation failed: {e}")
            return False
            
    def check_model_health(self, model_version: str) -> bool:
        """Check model health."""
        # Simulate health check
        logger.info(f"Checking health for model version {model_version}")
        time.sleep(1)  # Simulate health check time
        return True
        
    def check_prediction_endpoint(self, model_version: str) -> bool:
        """Check prediction endpoint."""
        # Simulate prediction test
        logger.info(f"Testing prediction endpoint for model version {model_version}")
        time.sleep(1)  # Simulate prediction test
        return True
        
    def check_metrics_collection(self, model_version: str) -> bool:
        """Check metrics collection."""
        # Simulate metrics check
        logger.info(f"Checking metrics collection for model version {model_version}")
        time.sleep(1)  # Simulate metrics check
        return True
        
    def save_experiment_config(self, config: Dict[str, Any]) -> bool:
        """Save A/B test experiment configuration."""
        try:
            config_file = f"/tmp/ab_experiment_{config['experiment_id']}.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"A/B test configuration saved to {config_file}")
            
            # Also save to a central experiments registry
            experiments_file = f"/tmp/ab_experiments_registry.json"
            try:
                with open(experiments_file, 'r') as f:
                    experiments = json.load(f)
            except FileNotFoundError:
                experiments = {"experiments": []}
                
            experiments["experiments"].append({
                "experiment_id": config["experiment_id"],
                "model_name": config["model_name"],
                "treatment_version": config["treatment_model"]["version"],
                "traffic_split": config["treatment_model"]["traffic_percentage"],
                "start_time": config["start_time"],
                "status": config["status"],
                "config_file": config_file
            })
            
            with open(experiments_file, 'w') as f:
                json.dump(experiments, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to save experiment config: {e}")
            return False
            
    def deploy(self, model_name: str, model_version: str = "latest", 
              traffic_split: float = 0.1, experiment_duration: int = 604800) -> bool:
        """Deploy model with A/B testing configuration."""
        logger.info(f"Starting A/B test deployment for {model_name}")
        
        try:
            # Validate inputs
            if not (0.0 < traffic_split < 1.0):
                logger.error("Traffic split must be between 0.0 and 1.0")
                return False
                
            # Step 1: Get model information
            model_info = self.get_model_info(model_name, model_version)
            if not model_info:
                return False
                
            actual_version = model_info["version"]
            logger.info(f"Deploying model version: {actual_version}")
            
            # Step 2: Create A/B test configuration
            ab_config = self.create_ab_test_config(
                model_name, actual_version, traffic_split, experiment_duration
            )
            
            # Step 3: Deploy model version
            if not self.deploy_model_version(model_name, actual_version):
                return False
                
            # Step 4: Validate deployment
            if not self.validate_deployment(model_name, actual_version):
                return False
                
            # Step 5: Configure traffic splitting
            if not self.configure_traffic_split(ab_config):
                return False
                
            # Step 6: Setup monitoring
            if not self.setup_monitoring(ab_config):
                return False
                
            # Step 7: Save experiment configuration
            if not self.save_experiment_config(ab_config):
                return False
                
            logger.info("A/B test deployment completed successfully")
            logger.info(f"Experiment ID: {ab_config['experiment_id']}")
            logger.info(f"Traffic split: {traffic_split*100:.1f}% to treatment model")
            logger.info(f"Experiment duration: {experiment_duration//3600} hours")
            
            return True
            
        except Exception as e:
            logger.error(f"A/B test deployment failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Deploy model with A/B testing')
    parser.add_argument('--model-name', required=True, help='Name of the model to deploy')
    parser.add_argument('--model-version', default='latest', help='Version of the model to deploy')
    parser.add_argument('--environment', default='production', help='Deployment environment')
    parser.add_argument('--traffic-split', type=float, default=0.1,
                       help='Percentage of traffic to send to new model (0.0-1.0)')
    parser.add_argument('--experiment-duration', type=int, default=604800,
                       help='Experiment duration in seconds (default: 7 days)')
    parser.add_argument('--mlflow-uri', help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    try:
        # Set MLflow tracking URI if provided
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
            
        deployer = ABTestModelDeployer(environment=args.environment)
        success = deployer.deploy(
            args.model_name,
            args.model_version,
            args.traffic_split,
            args.experiment_duration
        )
        
        if success:
            logger.info("A/B test deployment completed successfully")
            sys.exit(0)
        else:
            logger.error("A/B test deployment failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Deployment script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()