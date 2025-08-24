#!/usr/bin/env python3
"""
Alert Creation Script
Creates monitoring alerts for MLOps pipeline.
"""

import argparse
import logging
import sys
import json
import subprocess
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertManager:
    """Manage monitoring alerts for MLOps pipeline."""
    
    def __init__(self, model_name: str, environment: str = "production"):
        self.model_name = model_name
        self.environment = environment
        self.monitoring_namespace = f"monitoring-{environment}"
        
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
            
    def generate_alert_rules(self) -> Dict[str, Any]:
        """Generate Prometheus alert rules for the model."""
        alert_rules = {
            "groups": [
                {
                    "name": f"{self.model_name}_alerts",
                    "rules": [
                        {
                            "alert": "HighAPIErrorRate",
                            "expr": f"rate(http_requests_total{{job=\"mlops-api\",status=~\"5..\"}}[5m]) / rate(http_requests_total{{job=\"mlops-api\"}}[5m]) > 0.05",
                            "for": "2m",
                            "labels": {
                                "severity": "critical",
                                "service": "mlops-api",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "High API error rate detected",
                                "description": f"API error rate is above 5% for model {self.model_name} in {self.environment}"
                            }
                        },
                        {
                            "alert": "HighPredictionLatency",
                            "expr": f"histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket{{model_name=\"{self.model_name}\"}}[5m])) > 2.0",
                            "for": "5m",
                            "labels": {
                                "severity": "warning",
                                "service": "mlops-api",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "High prediction latency detected",
                                "description": f"95th percentile prediction latency is above 2 seconds for model {self.model_name}"
                            }
                        },
                        {
                            "alert": "ModelAccuracyDrop",
                            "expr": f"model_accuracy_score{{model_name=\"{self.model_name}\"}} < 0.70",
                            "for": "10m",
                            "labels": {
                                "severity": "critical",
                                "service": "mlops-api",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "Model accuracy has dropped significantly",
                                "description": f"Model accuracy for {self.model_name} has dropped below 70%"
                            }
                        },
                        {
                            "alert": "PodCrashLooping",
                            "expr": f"rate(kube_pod_container_status_restarts_total{{namespace=\"{self.environment}\",pod=~\".*mlops-api.*\"}}[15m]) > 0",
                            "for": "5m",
                            "labels": {
                                "severity": "warning",
                                "service": "mlops-api",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "Pod is crash looping",
                                "description": f"MLOps API pod is restarting frequently in {self.environment}"
                            }
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": f"container_memory_usage_bytes{{namespace=\"{self.environment}\",pod=~\".*mlops-api.*\"}} / container_spec_memory_limit_bytes * 100 > 80",
                            "for": "10m",
                            "labels": {
                                "severity": "warning",
                                "service": "mlops-api",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "High memory usage detected",
                                "description": f"Memory usage is above 80% for MLOps API pods in {self.environment}"
                            }
                        },
                        {
                            "alert": "HighCPUUsage",
                            "expr": f"rate(container_cpu_usage_seconds_total{{namespace=\"{self.environment}\",pod=~\".*mlops-api.*\"}}[5m]) * 100 > 80",
                            "for": "10m",
                            "labels": {
                                "severity": "warning",
                                "service": "mlops-api",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "High CPU usage detected",
                                "description": f"CPU usage is above 80% for MLOps API pods in {self.environment}"
                            }
                        },
                        {
                            "alert": "DataDriftDetected",
                            "expr": f"data_drift_score{{model_name=\"{self.model_name}\"}} > 0.1",
                            "for": "15m",
                            "labels": {
                                "severity": "warning",
                                "service": "mlops-api",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "Data drift detected",
                                "description": f"Significant data drift detected for model {self.model_name}"
                            }
                        },
                        {
                            "alert": "LowRequestVolume",
                            "expr": f"rate(prediction_requests_total{{model_name=\"{self.model_name}\"}}[1h]) < 1",
                            "for": "30m",
                            "labels": {
                                "severity": "info",
                                "service": "mlops-api",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "Low request volume",
                                "description": f"Unusually low request volume for model {self.model_name}"
                            }
                        },
                        {
                            "alert": "ModelServingDown",
                            "expr": f"up{{job=\"mlops-api\"}} == 0",
                            "for": "1m",
                            "labels": {
                                "severity": "critical",
                                "service": "mlops-api",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "Model serving is down",
                                "description": f"MLOps API is not responding in {self.environment}"
                            }
                        },
                        {
                            "alert": "DiskSpaceRunningLow",
                            "expr": f"(node_filesystem_avail_bytes{{fstype!=\"tmpfs\"}} / node_filesystem_size_bytes) * 100 < 20",
                            "for": "5m",
                            "labels": {
                                "severity": "warning",
                                "service": "infrastructure",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "Disk space running low",
                                "description": "Available disk space is below 20%"
                            }
                        }
                    ]
                },
                {
                    "name": f"{self.model_name}_business_metrics",
                    "rules": [
                        {
                            "alert": "HighChurnPredictionRate",
                            "expr": f"(sum(rate(prediction_requests_total{{model_name=\"{self.model_name}\",prediction=\"1\"}}[1h])) / sum(rate(prediction_requests_total{{model_name=\"{self.model_name}\"}}[1h]))) > 0.5",
                            "for": "30m",
                            "labels": {
                                "severity": "info",
                                "service": "business-metrics",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "High churn prediction rate",
                                "description": f"Churn prediction rate is above 50% for model {self.model_name}"
                            }
                        },
                        {
                            "alert": "PredictionAccuracyTrend",
                            "expr": f"rate(model_correct_predictions_total{{model_name=\"{self.model_name}\"}}[24h]) / rate(prediction_requests_total{{model_name=\"{self.model_name}\"}}[24h]) < 0.75",
                            "for": "1h",
                            "labels": {
                                "severity": "warning",
                                "service": "business-metrics",
                                "model": self.model_name,
                                "environment": self.environment
                            },
                            "annotations": {
                                "summary": "Prediction accuracy trend declining",
                                "description": f"24-hour prediction accuracy is below 75% for model {self.model_name}"
                            }
                        }
                    ]
                }
            ]
        }
        
        return alert_rules
        
    def create_alert_rules_configmap(self) -> bool:
        """Create ConfigMap with alert rules."""
        logger.info("Creating alert rules ConfigMap")
        
        alert_rules = self.generate_alert_rules()
        
        configmap_manifest = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: {self.model_name}-alert-rules
  namespace: {self.monitoring_namespace}
  labels:
    app: prometheus
    model: {self.model_name}
    environment: {self.environment}
data:
  alert_rules.yml: |
{self.yaml_dump(alert_rules, indent=4)}
"""
        
        # Write to temporary file and apply
        manifest_path = f"/tmp/{self.model_name}-alert-rules.yaml"
        try:
            with open(manifest_path, 'w') as f:
                f.write(configmap_manifest)
                
            cmd = f"apply -f {manifest_path}"
            code, stdout, stderr = self.run_kubectl(cmd)
            
            if code != 0:
                logger.error(f"Failed to create alert rules ConfigMap: {stderr}")
                return False
                
            logger.info(f"Alert rules ConfigMap created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating alert rules: {e}")
            return False
            
    def yaml_dump(self, data: Dict[str, Any], indent: int = 2) -> str:
        """Simple YAML dumper for alert rules."""
        def dump_value(value, level=0):
            spaces = " " * (level * indent)
            if isinstance(value, dict):
                result = []
                for k, v in value.items():
                    if isinstance(v, (dict, list)):
                        result.append(f"{spaces}{k}:")
                        result.append(dump_value(v, level + 1))
                    else:
                        result.append(f"{spaces}{k}: {self.format_value(v)}")
                return "\n".join(result)
            elif isinstance(value, list):
                result = []
                for item in value:
                    if isinstance(item, (dict, list)):
                        result.append(f"{spaces}-")
                        result.append(dump_value(item, level + 1))
                    else:
                        result.append(f"{spaces}- {self.format_value(item)}")
                return "\n".join(result)
            else:
                return f"{spaces}{self.format_value(value)}"
                
        return dump_value(data)
        
    def format_value(self, value) -> str:
        """Format value for YAML output."""
        if isinstance(value, str):
            if any(char in value for char in ['"', "'", '\n', ':', '#']):
                return f'"{value}"'
            return value
        return str(value)
        
    def update_prometheus_config(self) -> bool:
        """Update Prometheus configuration to include alert rules."""
        logger.info("Updating Prometheus configuration")
        
        # Get current Prometheus ConfigMap
        cmd = f"get configmap prometheus-config -n {self.monitoring_namespace} -o json"
        code, stdout, stderr = self.run_kubectl(cmd)
        
        if code != 0:
            logger.error(f"Failed to get Prometheus config: {stderr}")
            return False
            
        try:
            config_data = json.loads(stdout)
            current_config = config_data["data"]["prometheus.yml"]
            
            # Add rule files reference if not present
            if f"{self.model_name}-alert-rules" not in current_config:
                # Add the new rule file to the rule_files section
                updated_config = current_config.replace(
                    'rule_files:\n      - "alert_rules.yml"',
                    f'rule_files:\n      - "alert_rules.yml"\n      - "/etc/prometheus/rules/{self.model_name}-alert-rules/alert_rules.yml"'
                )
                
                # Create updated ConfigMap
                updated_manifest = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: {self.monitoring_namespace}
data:
  prometheus.yml: |
{updated_config}
"""
                
                # Apply updated configuration
                manifest_path = f"/tmp/prometheus-config-updated.yaml"
                with open(manifest_path, 'w') as f:
                    f.write(updated_manifest)
                    
                cmd = f"apply -f {manifest_path}"
                code, stdout, stderr = self.run_kubectl(cmd)
                
                if code != 0:
                    logger.error(f"Failed to update Prometheus config: {stderr}")
                    return False
                    
            logger.info("Prometheus configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating Prometheus config: {e}")
            return False
            
    def restart_prometheus(self) -> bool:
        """Restart Prometheus to reload configuration."""
        logger.info("Restarting Prometheus to reload configuration")
        
        cmd = f"rollout restart deployment/prometheus -n {self.monitoring_namespace}"
        code, stdout, stderr = self.run_kubectl(cmd)
        
        if code != 0:
            logger.error(f"Failed to restart Prometheus: {stderr}")
            return False
            
        # Wait for rollout to complete
        cmd = f"rollout status deployment/prometheus -n {self.monitoring_namespace} --timeout=300s"
        code, stdout, stderr = self.run_kubectl(cmd)
        
        if code != 0:
            logger.error(f"Prometheus restart timed out: {stderr}")
            return False
            
        logger.info("Prometheus restarted successfully")
        return True
        
    def create_notification_channels(self) -> bool:
        """Create notification channels for alerts."""
        logger.info("Creating notification channels")
        
        # Create webhook service for alert notifications
        webhook_manifest = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: alert-webhook-config
  namespace: {self.monitoring_namespace}
data:
  config.json: |
    {{
      "slack_webhook": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
      "email_smtp": {{
        "server": "smtp.gmail.com",
        "port": 587,
        "username": "alerts@mlops-pipeline.com",
        "recipients": [
          "ops-team@mlops-pipeline.com",
          "dev-team@mlops-pipeline.com"
        ]
      }},
      "pagerduty": {{
        "integration_key": "YOUR_PAGERDUTY_INTEGRATION_KEY"
      }}
    }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alert-webhook
  namespace: {self.monitoring_namespace}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alert-webhook
  template:
    metadata:
      labels:
        app: alert-webhook
    spec:
      containers:
      - name: webhook
        image: nginx:alpine
        ports:
        - containerPort: 80
        volumeMounts:
        - name: config
          mountPath: /etc/webhook
      volumes:
      - name: config
        configMap:
          name: alert-webhook-config
---
apiVersion: v1
kind: Service
metadata:
  name: webhook-service
  namespace: {self.monitoring_namespace}
spec:
  selector:
    app: alert-webhook
  ports:
  - port: 8080
    targetPort: 80
"""
        
        # Apply webhook configuration
        manifest_path = f"/tmp/alert-webhook.yaml"
        try:
            with open(manifest_path, 'w') as f:
                f.write(webhook_manifest)
                
            cmd = f"apply -f {manifest_path}"
            code, stdout, stderr = self.run_kubectl(cmd)
            
            if code != 0:
                logger.warning(f"Failed to create webhook service: {stderr}")
                # This is not critical, continue
                
            logger.info("Notification channels configured")
            return True
            
        except Exception as e:
            logger.warning(f"Error creating notification channels: {e}")
            return True  # Not critical
            
    def create_alerts(self) -> bool:
        """Create all monitoring alerts."""
        logger.info(f"Creating alerts for model: {self.model_name}")
        
        try:
            # Step 1: Create alert rules ConfigMap
            if not self.create_alert_rules_configmap():
                return False
                
            # Step 2: Update Prometheus configuration
            if not self.update_prometheus_config():
                return False
                
            # Step 3: Create notification channels
            if not self.create_notification_channels():
                return False
                
            # Step 4: Restart Prometheus to reload rules
            if not self.restart_prometheus():
                return False
                
            logger.info("All alerts created successfully")
            
            # Print summary
            self.print_alert_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create alerts: {e}")
            return False
            
    def print_alert_summary(self):
        """Print summary of created alerts."""
        print(f"\n{'='*60}")
        print(f"ALERTS CREATED FOR MODEL: {self.model_name}")
        print(f"{'='*60}")
        print(f"Environment: {self.environment}")
        print(f"Monitoring Namespace: {self.monitoring_namespace}")
        print(f"\nAlert Categories:")
        print(f"  • Infrastructure Alerts (CPU, Memory, Disk)")
        print(f"  • Application Alerts (API errors, latency)")
        print(f"  • Model Performance Alerts (accuracy, drift)")
        print(f"  • Business Metrics Alerts (churn rate, trends)")
        print(f"\nAlert Severities:")
        print(f"  • CRITICAL: Immediate attention required")
        print(f"  • WARNING: Should be addressed soon")
        print(f"  • INFO: Informational, monitor trends")
        print(f"\nTo view alerts:")
        print(f"  kubectl port-forward -n {self.monitoring_namespace} service/prometheus 9090:9090")
        print(f"  Open: http://localhost:9090/alerts")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Create monitoring alerts')
    parser.add_argument('--model-name', required=True, help='Name of the model to create alerts for')
    parser.add_argument('--environment', default='production', help='Environment (production, staging)')
    
    args = parser.parse_args()
    
    try:
        alert_manager = AlertManager(args.model_name, args.environment)
        success = alert_manager.create_alerts()
        
        if success:
            logger.info("Alert creation completed successfully")
            sys.exit(0)
        else:
            logger.error("Alert creation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Alert creation script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()