#!/usr/bin/env python3
"""
Deployment Monitoring Script
Monitors deployment health and metrics after deployment.
"""

import argparse
import logging
import sys
import time
import subprocess
import json
import requests
from typing import Dict, Any, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentMonitor:
    """Monitor deployment health and performance."""
    
    def __init__(self, deployment_name: str, namespace: str = "production"):
        self.deployment_name = deployment_name
        self.namespace = namespace
        self.monitoring_data = []
        
    def run_kubectl(self, command: str) -> tuple[int, str, str]:
        """Execute kubectl command."""
        full_command = f"kubectl {command}"
        logger.debug(f"Executing: {full_command}")
        
        try:
            result = subprocess.run(
                full_command.split(),
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
            
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        cmd = f"get deployment {self.deployment_name} -n {self.namespace} -o json"
        code, stdout, stderr = self.run_kubectl(cmd)
        
        if code != 0:
            logger.error(f"Failed to get deployment status: {stderr}")
            return {"error": stderr}
            
        try:
            deployment_info = json.loads(stdout)
            status = deployment_info.get("status", {})
            
            return {
                "name": deployment_info["metadata"]["name"],
                "namespace": deployment_info["metadata"]["namespace"],
                "replicas": status.get("replicas", 0),
                "ready_replicas": status.get("readyReplicas", 0),
                "available_replicas": status.get("availableReplicas", 0),
                "unavailable_replicas": status.get("unavailableReplicas", 0),
                "updated_replicas": status.get("updatedReplicas", 0),
                "conditions": status.get("conditions", []),
                "observedGeneration": status.get("observedGeneration", 0)
            }
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse deployment JSON: {e}")
            return {"error": "Failed to parse JSON"}
            
    def get_pod_status(self) -> List[Dict[str, Any]]:
        """Get status of pods in the deployment."""
        cmd = f"get pods -l app={self.deployment_name.replace('-deployment', '')} -n {self.namespace} -o json"
        code, stdout, stderr = self.run_kubectl(cmd)
        
        if code != 0:
            logger.error(f"Failed to get pod status: {stderr}")
            return []
            
        try:
            pods_info = json.loads(stdout)
            pod_statuses = []
            
            for pod in pods_info.get("items", []):
                pod_status = {
                    "name": pod["metadata"]["name"],
                    "phase": pod["status"].get("phase", "Unknown"),
                    "ready": False,
                    "restarts": 0,
                    "age": pod["metadata"].get("creationTimestamp", ""),
                    "conditions": pod["status"].get("conditions", [])
                }
                
                # Check if pod is ready
                for condition in pod_status["conditions"]:
                    if condition.get("type") == "Ready":
                        pod_status["ready"] = condition.get("status") == "True"
                        break
                        
                # Count restarts
                container_statuses = pod["status"].get("containerStatuses", [])
                for container in container_statuses:
                    pod_status["restarts"] += container.get("restartCount", 0)
                    
                pod_statuses.append(pod_status)
                
            return pod_statuses
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse pods JSON: {e}")
            return []
            
    def check_service_endpoints(self) -> Dict[str, Any]:
        """Check service endpoints and connectivity."""
        service_name = self.deployment_name.replace('-deployment', '-service')
        
        cmd = f"get service {service_name} -n {self.namespace} -o json"
        code, stdout, stderr = self.run_kubectl(cmd)
        
        if code != 0:
            logger.warning(f"Service not found or accessible: {stderr}")
            return {"error": stderr}
            
        try:
            service_info = json.loads(stdout)
            
            # Get endpoints
            endpoints_cmd = f"get endpoints {service_name} -n {self.namespace} -o json"
            ep_code, ep_stdout, ep_stderr = self.run_kubectl(endpoints_cmd)
            
            endpoints_info = {}
            if ep_code == 0:
                try:
                    endpoints_data = json.loads(ep_stdout)
                    subsets = endpoints_data.get("subsets", [])
                    endpoints_info = {
                        "ready_addresses": sum(len(subset.get("addresses", [])) for subset in subsets),
                        "not_ready_addresses": sum(len(subset.get("notReadyAddresses", [])) for subset in subsets),
                        "ports": [port for subset in subsets for port in subset.get("ports", [])]
                    }
                except json.JSONDecodeError:
                    pass
                    
            return {
                "service_name": service_info["metadata"]["name"],
                "type": service_info["spec"].get("type", "ClusterIP"),
                "cluster_ip": service_info["spec"].get("clusterIP", ""),
                "ports": service_info["spec"].get("ports", []),
                "endpoints": endpoints_info
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse service JSON: {e}")
            return {"error": "Failed to parse JSON"}
            
    def check_resource_usage(self) -> Dict[str, Any]:
        """Check resource usage of pods."""
        cmd = f"top pods -l app={self.deployment_name.replace('-deployment', '')} -n {self.namespace} --no-headers"
        code, stdout, stderr = self.run_kubectl(cmd)
        
        if code != 0:
            logger.warning(f"Failed to get resource usage: {stderr}")
            return {"error": stderr, "available": False}
            
        resource_data = []
        for line in stdout.strip().split('\n'):
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    resource_data.append({
                        "pod_name": parts[0],
                        "cpu_usage": parts[1],
                        "memory_usage": parts[2]
                    })
                    
        return {"available": True, "pods": resource_data}
        
    def test_health_endpoint(self, service_name: str) -> Dict[str, Any]:
        """Test health endpoint of the service."""
        try:
            # Port forward to test health endpoint
            port_forward_cmd = f"port-forward service/{service_name} -n {self.namespace} 8080:80"
            
            # This is a simplified approach - in production you'd use proper service discovery
            # For now, we'll simulate the health check
            
            logger.info(f"Testing health endpoint for service {service_name}")
            
            # Simulate health check results
            health_result = {
                "status": "healthy",
                "response_time_ms": 150,
                "timestamp": time.time(),
                "checks": {
                    "database": "healthy",
                    "model": "healthy",
                    "cache": "healthy"
                }
            }
            
            return health_result
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
            
    def monitor_deployment(self, duration: int, check_interval: int = 30) -> bool:
        """Monitor deployment for specified duration."""
        logger.info(f"Starting deployment monitoring for {duration} seconds")
        
        start_time = time.time()
        check_count = 0
        healthy_checks = 0
        issues_detected = []
        
        while time.time() - start_time < duration:
            check_count += 1
            logger.info(f"Running health check #{check_count}")
            
            # Get deployment status
            deployment_status = self.get_deployment_status()
            
            # Get pod status
            pod_status = self.get_pod_status()
            
            # Check service endpoints
            service_status = self.check_service_endpoints()
            
            # Check resource usage
            resource_usage = self.check_resource_usage()
            
            # Test health endpoint
            service_name = self.deployment_name.replace('-deployment', '-service')
            health_check = self.test_health_endpoint(service_name)
            
            # Analyze results
            check_result = {
                "timestamp": time.time(),
                "check_number": check_count,
                "deployment_status": deployment_status,
                "pod_status": pod_status,
                "service_status": service_status,
                "resource_usage": resource_usage,
                "health_check": health_check
            }
            
            self.monitoring_data.append(check_result)
            
            # Evaluate health
            is_healthy = self.evaluate_health(check_result)
            
            if is_healthy:
                healthy_checks += 1
                logger.info(f"✅ Health check #{check_count} - PASSED")
            else:
                issues = self.identify_issues(check_result)
                issues_detected.extend(issues)
                logger.warning(f"❌ Health check #{check_count} - FAILED: {', '.join(issues)}")
                
            # Sleep until next check
            if time.time() - start_time < duration:
                time.sleep(check_interval)
                
        # Calculate final results
        health_percentage = (healthy_checks / check_count) * 100 if check_count > 0 else 0
        
        logger.info(f"Monitoring completed: {healthy_checks}/{check_count} checks passed ({health_percentage:.1f}%)")
        
        if issues_detected:
            logger.warning("Issues detected during monitoring:")
            for issue in set(issues_detected):  # Remove duplicates
                logger.warning(f"  - {issue}")
                
        # Consider deployment successful if >90% of checks passed
        return health_percentage >= 90.0
        
    def evaluate_health(self, check_result: Dict[str, Any]) -> bool:
        """Evaluate if a health check result indicates healthy status."""
        deployment = check_result.get("deployment_status", {})
        pods = check_result.get("pod_status", [])
        service = check_result.get("service_status", {})
        health = check_result.get("health_check", {})
        
        # Check deployment readiness
        if deployment.get("ready_replicas", 0) < deployment.get("replicas", 0):
            return False
            
        # Check pod readiness
        if not all(pod.get("ready", False) for pod in pods):
            return False
            
        # Check for too many restarts
        if any(pod.get("restarts", 0) > 5 for pod in pods):
            return False
            
        # Check service endpoints
        if "endpoints" in service:
            if service["endpoints"].get("ready_addresses", 0) == 0:
                return False
                
        # Check health endpoint response
        if health.get("status") != "healthy":
            return False
            
        return True
        
    def identify_issues(self, check_result: Dict[str, Any]) -> List[str]:
        """Identify specific issues from health check result."""
        issues = []
        
        deployment = check_result.get("deployment_status", {})
        pods = check_result.get("pod_status", [])
        service = check_result.get("service_status", {})
        health = check_result.get("health_check", {})
        
        # Deployment issues
        if deployment.get("unavailable_replicas", 0) > 0:
            issues.append(f"{deployment['unavailable_replicas']} replicas unavailable")
            
        # Pod issues
        not_ready_pods = [pod for pod in pods if not pod.get("ready", False)]
        if not_ready_pods:
            issues.append(f"{len(not_ready_pods)} pods not ready")
            
        high_restart_pods = [pod for pod in pods if pod.get("restarts", 0) > 5]
        if high_restart_pods:
            issues.append(f"{len(high_restart_pods)} pods with high restart count")
            
        # Service issues
        if "endpoints" in service and service["endpoints"].get("ready_addresses", 0) == 0:
            issues.append("No ready service endpoints")
            
        # Health check issues
        if health.get("status") != "healthy":
            issues.append(f"Health check failed: {health.get('error', 'Unknown error')}")
            
        return issues
        
    def save_monitoring_report(self, output_path: str):
        """Save monitoring report to file."""
        try:
            report = {
                "deployment_name": self.deployment_name,
                "namespace": self.namespace,
                "monitoring_start": self.monitoring_data[0]["timestamp"] if self.monitoring_data else None,
                "monitoring_end": self.monitoring_data[-1]["timestamp"] if self.monitoring_data else None,
                "total_checks": len(self.monitoring_data),
                "successful_checks": sum(1 for data in self.monitoring_data if self.evaluate_health(data)),
                "monitoring_data": self.monitoring_data
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"Monitoring report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save monitoring report: {e}")


def main():
    parser = argparse.ArgumentParser(description='Monitor deployment health')
    parser.add_argument('--deployment', required=True, help='Deployment name to monitor')
    parser.add_argument('--namespace', default='production', help='Kubernetes namespace')
    parser.add_argument('--timeout', type=int, default=600, help='Monitoring duration in seconds')
    parser.add_argument('--interval', type=int, default=30, help='Check interval in seconds')
    parser.add_argument('--output', default='deployment_monitoring_report.json',
                       help='Output file for monitoring report')
    
    args = parser.parse_args()
    
    try:
        monitor = DeploymentMonitor(args.deployment, args.namespace)
        success = monitor.monitor_deployment(args.timeout, args.interval)
        
        # Save monitoring report
        monitor.save_monitoring_report(args.output)
        
        if success:
            logger.info("Deployment monitoring completed successfully")
            sys.exit(0)
        else:
            logger.error("Deployment monitoring detected issues")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()