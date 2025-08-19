#!/usr/bin/env python3
"""Blue-Green deployment script for zero-downtime deployments."""
import argparse
import logging
import sys
import subprocess
import time
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BlueGreenDeployer:
    """Blue-Green deployment manager."""
    
    def __init__(self, namespace: str = "production"):
        self.namespace = namespace
        self.app_name = "mlops-api"
        
    def get_current_environment(self) -> str:
        """Get currently active environment (blue or green)."""
        try:
            # Check which environment is receiving traffic
            cmd = [
                "kubectl", "get", "service", f"{self.app_name}-service",
                "-n", self.namespace, "-o", "jsonpath={.spec.selector.environment}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            current_env = result.stdout.strip()
            
            if current_env in ["blue", "green"]:
                logger.info(f"Current active environment: {current_env}")
                return current_env
            else:
                # Default to blue if not set
                logger.info("No current environment found, defaulting to blue")
                return "blue"
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get current environment: {e}")
            return "blue"
            
    def get_target_environment(self, current_env: str) -> str:
        """Get target environment for deployment."""
        return "green" if current_env == "blue" else "blue"
        
    def deploy_to_environment(self, image: str, target_env: str) -> bool:
        """Deploy new image to target environment."""
        try:
            logger.info(f"Deploying {image} to {target_env} environment")
            
            # Update deployment with new image
            cmd = [
                "kubectl", "set", "image", 
                f"deployment/{self.app_name}-{target_env}",
                f"{self.app_name}={image}",
                "-n", self.namespace
            ]
            
            subprocess.run(cmd, check=True)
            
            # Wait for rollout to complete
            cmd = [
                "kubectl", "rollout", "status", 
                f"deployment/{self.app_name}-{target_env}",
                "-n", self.namespace,
                "--timeout=600s"
            ]
            
            subprocess.run(cmd, check=True)
            
            logger.info(f"✓ Deployment to {target_env} completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Deployment to {target_env} failed: {e}")
            return False
            
    def run_health_checks(self, target_env: str) -> bool:
        """Run health checks on target environment."""
        try:
            logger.info(f"Running health checks on {target_env} environment")
            
            # Get pod IP for direct health check
            cmd = [
                "kubectl", "get", "pods", 
                "-l", f"app={self.app_name},environment={target_env}",
                "-n", self.namespace,
                "-o", "jsonpath={.items[0].status.podIP}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            pod_ip = result.stdout.strip()
            
            if not pod_ip:
                logger.error("Could not get pod IP for health check")
                return False
                
            # Run health check using kubectl port-forward
            logger.info(f"Testing health endpoint on {pod_ip}")
            
            # Port forward to pod
            port_forward_cmd = [
                "kubectl", "port-forward", 
                f"pod/{self.app_name}-{target_env}",
                "8000:8000",
                "-n", self.namespace
            ]
            
            # Start port-forward in background
            port_forward_proc = subprocess.Popen(
                port_forward_cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            
            # Wait for port-forward to establish
            time.sleep(5)
            
            # Run health check
            health_check_success = self._test_health_endpoint()
            
            # Clean up port-forward
            port_forward_proc.terminate()
            port_forward_proc.wait()
            
            if health_check_success:
                logger.info(f"✓ Health checks passed for {target_env}")
                return True
            else:
                logger.error(f"✗ Health checks failed for {target_env}")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Health check failed: {e}")
            return False
            
    def _test_health_endpoint(self) -> bool:
        """Test health endpoint."""
        try:
            import requests
            
            # Test health endpoint
            response = requests.get("http://localhost:8000/health", timeout=10)
            response.raise_for_status()
            
            health_data = response.json()
            if health_data.get("status") == "healthy":
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Health endpoint test failed: {e}")
            return False
            
    def switch_traffic(self, target_env: str) -> bool:
        """Switch traffic to target environment."""
        try:
            logger.info(f"Switching traffic to {target_env} environment")
            
            # Update service selector to point to target environment
            cmd = [
                "kubectl", "patch", "service", f"{self.app_name}-service",
                "-n", self.namespace,
                "--type", "merge",
                "-p", f'{{"spec":{{"selector":{{"environment":"{target_env}"}}}}}}'
            ]
            
            subprocess.run(cmd, check=True)
            
            # Verify traffic switch
            time.sleep(5)
            current_env = self.get_current_environment()
            
            if current_env == target_env:
                logger.info(f"✓ Traffic successfully switched to {target_env}")
                return True
            else:
                logger.error(f"✗ Traffic switch verification failed")
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Traffic switch failed: {e}")
            return False
            
    def cleanup_old_environment(self, old_env: str) -> bool:
        """Cleanup old environment (optional)."""
        try:
            logger.info(f"Cleaning up {old_env} environment")
            
            # Scale down old deployment
            cmd = [
                "kubectl", "scale", "deployment", f"{self.app_name}-{old_env}",
                "--replicas=0",
                "-n", self.namespace
            ]
            
            subprocess.run(cmd, check=True)
            
            logger.info(f"✓ {old_env} environment scaled down")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Cleanup failed: {e}")
            return False
            
    def rollback(self, rollback_env: str) -> bool:
        """Rollback to previous environment."""
        try:
            logger.info(f"Rolling back to {rollback_env} environment")
            
            # Scale up rollback environment
            cmd = [
                "kubectl", "scale", "deployment", f"{self.app_name}-{rollback_env}",
                "--replicas=3",
                "-n", self.namespace
            ]
            
            subprocess.run(cmd, check=True)
            
            # Wait for rollback environment to be ready
            cmd = [
                "kubectl", "rollout", "status", 
                f"deployment/{self.app_name}-{rollback_env}",
                "-n", self.namespace,
                "--timeout=300s"
            ]
            
            subprocess.run(cmd, check=True)
            
            # Switch traffic back
            return self.switch_traffic(rollback_env)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Rollback failed: {e}")
            return False
            
    def deploy(self, image: str) -> bool:
        """Execute complete blue-green deployment."""
        try:
            # Get current and target environments
            current_env = self.get_current_environment()
            target_env = self.get_target_environment(current_env)
            
            logger.info(f"Blue-Green Deployment Plan:")
            logger.info(f"  Current environment: {current_env}")
            logger.info(f"  Target environment: {target_env}")
            logger.info(f"  New image: {image}")
            
            # Step 1: Deploy to target environment
            if not self.deploy_to_environment(image, target_env):
                logger.error("Deployment failed, aborting")
                return False
                
            # Step 2: Run health checks
            if not self.run_health_checks(target_env):
                logger.error("Health checks failed, aborting deployment")
                return False
                
            # Step 3: Switch traffic
            if not self.switch_traffic(target_env):
                logger.error("Traffic switch failed, attempting rollback")
                self.rollback(current_env)
                return False
                
            # Step 4: Monitor new environment briefly
            logger.info("Monitoring new environment for 60 seconds...")
            time.sleep(60)
            
            # Final health check
            if not self.run_health_checks(target_env):
                logger.error("Post-deployment health check failed, rolling back")
                self.rollback(current_env)
                return False
                
            # Step 5: Cleanup old environment (optional, keep for quick rollback)
            logger.info("Keeping old environment for quick rollback capability")
            # self.cleanup_old_environment(current_env)
            
            logger.info(f"✓ Blue-Green deployment completed successfully!")
            logger.info(f"  Active environment: {target_env}")
            logger.info(f"  Standby environment: {current_env}")
            
            return True
            
        except Exception as e:
            logger.error(f"Blue-Green deployment failed: {e}")
            return False


def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Blue-Green deployment")
    parser.add_argument("--image", required=True, help="Docker image to deploy")
    parser.add_argument("--namespace", default="production", help="Kubernetes namespace")
    parser.add_argument("--rollback", action="store_true", help="Rollback to previous environment")
    
    args = parser.parse_args()
    
    deployer = BlueGreenDeployer(args.namespace)
    
    if args.rollback:
        logger.info("Initiating rollback...")
        current_env = deployer.get_current_environment()
        rollback_env = deployer.get_target_environment(current_env)
        
        success = deployer.rollback(rollback_env)
    else:
        logger.info(f"Starting Blue-Green deployment of {args.image}")
        success = deployer.deploy(args.image)
        
    if success:
        logger.info("Operation completed successfully")
        sys.exit(0)
    else:
        logger.error("Operation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()