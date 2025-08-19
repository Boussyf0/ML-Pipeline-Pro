#!/usr/bin/env python3
"""Model deployment script for CI/CD pipeline."""
import argparse
import logging
import sys
import os
from pathlib import Path
import yaml
import requests
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.model_manager import ModelManager
from models.registry import ModelRegistry


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def deploy_model(model_name: str, environment: str, version: str = None, 
                promote_from_registry: bool = False) -> bool:
    """Deploy model to specified environment."""
    try:
        model_manager = ModelManager()
        registry = ModelRegistry()
        
        # Get model version to deploy
        if promote_from_registry:
            # Get latest version from registry
            if not version:
                version = registry.get_latest_version(model_name)
                if not version:
                    logger.error(f"No versions found for model {model_name}")
                    return False
                    
            logger.info(f"Promoting model {model_name} version {version} to {environment}")
        else:
            if not version:
                logger.error("Version must be specified when not promoting from registry")
                return False
                
        # Deploy model
        deployment_result = model_manager.deploy_model(model_name, version, environment)
        
        logger.info(f"Deployment result: {deployment_result}")
        
        # Verify deployment
        health = model_manager.get_model_health(model_name)
        env_status = health.get("environments", {}).get(environment, {})
        
        if env_status.get("deployed") and env_status.get("version") == version:
            logger.info(f"✓ Model {model_name} v{version} successfully deployed to {environment}")
            return True
        else:
            logger.error(f"✗ Deployment verification failed for {model_name} v{version}")
            return False
            
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return False


def run_health_checks(api_url: str) -> bool:
    """Run health checks on deployed API."""
    try:
        # Health endpoint
        health_url = f"{api_url}/health"
        response = requests.get(health_url, timeout=30)
        response.raise_for_status()
        
        health_data = response.json()
        if health_data.get("status") == "healthy":
            logger.info("✓ API health check passed")
        else:
            logger.error(f"✗ API health check failed: {health_data}")
            return False
            
        # Model endpoint test
        predict_url = f"{api_url}/predict"
        test_payload = {
            "features": {
                "tenure": 12,
                "monthly_charges": 65.0,
                "total_charges": 780.0,
                "contract": "Month-to-month",
                "payment_method": "Electronic check"
            }
        }
        
        response = requests.post(predict_url, json=test_payload, timeout=30)
        response.raise_for_status()
        
        prediction_data = response.json()
        if "prediction" in prediction_data:
            logger.info(f"✓ Prediction endpoint test passed: {prediction_data}")
        else:
            logger.error(f"✗ Prediction endpoint test failed: {prediction_data}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Health checks failed: {e}")
        return False


def setup_monitoring_alerts(model_name: str, environment: str) -> bool:
    """Setup monitoring alerts for deployed model."""
    try:
        # This would integrate with your monitoring system (Prometheus, Grafana, etc.)
        logger.info(f"Setting up monitoring alerts for {model_name} in {environment}")
        
        # Example: Create Prometheus alerts
        alerts_config = {
            "groups": [
                {
                    "name": f"{model_name}_alerts",
                    "rules": [
                        {
                            "alert": f"{model_name}_HighErrorRate",
                            "expr": f'rate(model_prediction_errors_total{{model="{model_name}",environment="{environment}"}}[5m]) > 0.05',
                            "for": "2m",
                            "labels": {
                                "severity": "warning",
                                "model": model_name,
                                "environment": environment
                            },
                            "annotations": {
                                "summary": f"High error rate for {model_name}",
                                "description": f"Error rate for {model_name} in {environment} is above 5%"
                            }
                        },
                        {
                            "alert": f"{model_name}_HighLatency",
                            "expr": f'histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket{{model="{model_name}",environment="{environment}"}}[5m])) > 1.0',
                            "for": "5m",
                            "labels": {
                                "severity": "warning",
                                "model": model_name,
                                "environment": environment
                            },
                            "annotations": {
                                "summary": f"High latency for {model_name}",
                                "description": f"95th percentile latency for {model_name} in {environment} is above 1 second"
                            }
                        }
                    ]
                }
            ]
        }
        
        # Save alerts configuration
        alerts_file = Path(f"config/alerts/{model_name}_{environment}_alerts.yml")
        alerts_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(alerts_file, 'w') as f:
            yaml.dump(alerts_config, f)
            
        logger.info(f"✓ Monitoring alerts configured: {alerts_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup monitoring alerts: {e}")
        return False


def wait_for_deployment(model_name: str, environment: str, timeout: int = 300) -> bool:
    """Wait for deployment to be ready."""
    logger.info(f"Waiting for {model_name} deployment in {environment} (timeout: {timeout}s)")
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            model_manager = ModelManager()
            health = model_manager.get_model_health(model_name)
            
            env_status = health.get("environments", {}).get(environment, {})
            
            if env_status.get("deployed") and env_status.get("status") == "healthy":
                logger.info(f"✓ Deployment ready after {int(time.time() - start_time)}s")
                return True
                
            logger.info(f"Deployment not ready yet, retrying... ({int(time.time() - start_time)}s)")
            time.sleep(10)
            
        except Exception as e:
            logger.warning(f"Error checking deployment status: {e}")
            time.sleep(10)
            
    logger.error(f"✗ Deployment timeout after {timeout}s")
    return False


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy ML model")
    parser.add_argument("--model-name", required=True, help="Name of the model to deploy")
    parser.add_argument("--environment", required=True, choices=["staging", "production"], 
                       help="Target environment")
    parser.add_argument("--version", help="Model version to deploy")
    parser.add_argument("--promote-from-registry", action="store_true", 
                       help="Promote latest version from registry")
    parser.add_argument("--api-url", help="API URL for health checks")
    parser.add_argument("--skip-health-checks", action="store_true", 
                       help="Skip health checks")
    parser.add_argument("--setup-monitoring", action="store_true", 
                       help="Setup monitoring alerts")
    parser.add_argument("--timeout", type=int, default=300, 
                       help="Deployment timeout in seconds")
    
    args = parser.parse_args()
    
    logger.info(f"Starting deployment: {args.model_name} to {args.environment}")
    
    try:
        # Deploy model
        success = deploy_model(
            args.model_name, 
            args.environment, 
            args.version, 
            args.promote_from_registry
        )
        
        if not success:
            logger.error("Model deployment failed")
            sys.exit(1)
            
        # Wait for deployment to be ready
        if not wait_for_deployment(args.model_name, args.environment, args.timeout):
            logger.error("Deployment readiness check failed")
            sys.exit(1)
            
        # Run health checks
        if args.api_url and not args.skip_health_checks:
            if not run_health_checks(args.api_url):
                logger.error("Health checks failed")
                sys.exit(1)
                
        # Setup monitoring
        if args.setup_monitoring:
            if not setup_monitoring_alerts(args.model_name, args.environment):
                logger.warning("Failed to setup monitoring alerts")
                
        logger.info(f"✓ Deployment completed successfully: {args.model_name} to {args.environment}")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()