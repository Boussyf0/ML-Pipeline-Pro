#!/usr/bin/env python3
"""
Monitoring Setup Script
Sets up monitoring infrastructure for MLOps pipeline.
"""

import argparse
import logging
import sys
import json
import subprocess
import time
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringSetup:
    """Setup monitoring infrastructure."""
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.namespace = f"monitoring-{environment}"
        
    def run_kubectl(self, command: str) -> tuple[int, str, str]:
        """Execute kubectl command."""
        full_command = f"kubectl {command}"
        logger.debug(f"Executing: {full_command}")
        
        try:
            result = subprocess.run(
                full_command.split(),
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", "Command timed out"
        except Exception as e:
            return 1, "", str(e)
            
    def create_monitoring_namespace(self) -> bool:
        """Create monitoring namespace if it doesn't exist."""
        logger.info(f"Creating monitoring namespace: {self.namespace}")
        
        # Check if namespace exists
        cmd = f"get namespace {self.namespace}"
        code, stdout, stderr = self.run_kubectl(cmd)
        
        if code == 0:
            logger.info(f"Namespace {self.namespace} already exists")
            return True
            
        # Create namespace
        cmd = f"create namespace {self.namespace}"
        code, stdout, stderr = self.run_kubectl(cmd)
        
        if code != 0:
            logger.error(f"Failed to create namespace: {stderr}")
            return False
            
        logger.info(f"Namespace {self.namespace} created successfully")
        return True
        
    def setup_prometheus(self) -> bool:
        """Setup Prometheus monitoring."""
        logger.info("Setting up Prometheus")
        
        prometheus_config = self.generate_prometheus_config()
        
        # Create ConfigMap for Prometheus configuration
        config_manifest = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: {self.namespace}
data:
  prometheus.yml: |
{prometheus_config}
"""
        
        # Apply ConfigMap
        if not self.apply_manifest(config_manifest, "prometheus-config.yaml"):
            return False
            
        # Deploy Prometheus
        prometheus_deployment = self.generate_prometheus_deployment()
        
        if not self.apply_manifest(prometheus_deployment, "prometheus-deployment.yaml"):
            return False
            
        logger.info("Prometheus setup completed")
        return True
        
    def generate_prometheus_config(self) -> str:
        """Generate Prometheus configuration."""
        config = f"""
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
      - "alert_rules.yml"
    
    alerting:
      alertmanagers:
        - static_configs:
            - targets:
              - alertmanager:9093
    
    scrape_configs:
      - job_name: 'mlops-api'
        static_configs:
          - targets: ['mlops-api-service.{self.environment}:8000']
        metrics_path: '/metrics'
        scrape_interval: 15s
        
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
            namespaces:
              names:
                - {self.environment}
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
            action: replace
            target_label: __metrics_path__
            regex: (.+)
          - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
            action: replace
            regex: ([^:]+)(?::\d+)?;(\d+)
            replacement: $1:$2
            target_label: __address__
            
      - job_name: 'model-metrics'
        static_configs:
          - targets: ['mlops-api-service.{self.environment}:8000']
        metrics_path: '/model/metrics'
        scrape_interval: 30s
"""
        return config.strip()
        
    def generate_prometheus_deployment(self) -> str:
        """Generate Prometheus deployment manifest."""
        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: {self.namespace}
  labels:
    app: prometheus
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:v2.45.0
        ports:
        - containerPort: 9090
        args:
          - '--config.file=/etc/prometheus/prometheus.yml'
          - '--storage.tsdb.path=/prometheus'
          - '--web.console.libraries=/etc/prometheus/console_libraries'
          - '--web.console.templates=/etc/prometheus/consoles'
          - '--storage.tsdb.retention.time=15d'
          - '--web.enable-lifecycle'
        volumeMounts:
        - name: config-volume
          mountPath: /etc/prometheus
        - name: storage-volume
          mountPath: /prometheus
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: config-volume
        configMap:
          name: prometheus-config
      - name: storage-volume
        emptyDir: {{}}
---
apiVersion: v1
kind: Service
metadata:
  name: prometheus
  namespace: {self.namespace}
  labels:
    app: prometheus
spec:
  type: ClusterIP
  ports:
  - port: 9090
    targetPort: 9090
    name: web
  selector:
    app: prometheus
"""
        return manifest.strip()
        
    def setup_grafana(self) -> bool:
        """Setup Grafana dashboards."""
        logger.info("Setting up Grafana")
        
        grafana_deployment = self.generate_grafana_deployment()
        
        if not self.apply_manifest(grafana_deployment, "grafana-deployment.yaml"):
            return False
            
        # Setup dashboards
        if not self.create_grafana_dashboards():
            return False
            
        logger.info("Grafana setup completed")
        return True
        
    def generate_grafana_deployment(self) -> str:
        """Generate Grafana deployment manifest."""
        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafana
  namespace: {self.namespace}
  labels:
    app: grafana
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafana
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - name: grafana
        image: grafana/grafana:10.0.0
        ports:
        - containerPort: 3000
        env:
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: "mlops-admin-password"
        - name: GF_INSTALL_PLUGINS
          value: "grafana-piechart-panel"
        volumeMounts:
        - name: grafana-storage
          mountPath: /var/lib/grafana
        - name: grafana-config
          mountPath: /etc/grafana/provisioning
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
      volumes:
      - name: grafana-storage
        emptyDir: {{}}
      - name: grafana-config
        configMap:
          name: grafana-config
---
apiVersion: v1
kind: Service
metadata:
  name: grafana
  namespace: {self.namespace}
  labels:
    app: grafana
spec:
  type: LoadBalancer
  ports:
  - port: 3000
    targetPort: 3000
    name: web
  selector:
    app: grafana
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-config
  namespace: {self.namespace}
data:
  datasources.yml: |
    apiVersion: 1
    datasources:
      - name: Prometheus
        type: prometheus
        access: proxy
        url: http://prometheus:9090
        isDefault: true
  dashboards.yml: |
    apiVersion: 1
    providers:
      - name: 'default'
        orgId: 1
        folder: ''
        type: file
        disableDeletion: false
        updateIntervalSeconds: 10
        options:
          path: /var/lib/grafana/dashboards
"""
        return manifest.strip()
        
    def create_grafana_dashboards(self) -> bool:
        """Create Grafana dashboards for MLOps monitoring."""
        logger.info("Creating Grafana dashboards")
        
        # Main MLOps dashboard
        mlops_dashboard = self.generate_mlops_dashboard()
        
        dashboard_manifest = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlops-dashboards
  namespace: {self.namespace}
data:
  mlops-overview.json: |
{mlops_dashboard}
"""
        
        return self.apply_manifest(dashboard_manifest, "grafana-dashboards.yaml")
        
    def generate_mlops_dashboard(self) -> str:
        """Generate MLOps overview dashboard JSON."""
        dashboard = {
            "dashboard": {
                "id": None,
                "title": "MLOps Pipeline Overview",
                "tags": ["mlops", "production"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "API Request Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total[5m])",
                                "legendFormat": "Requests/sec"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                    },
                    {
                        "id": 2,
                        "title": "Prediction Latency",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, rate(model_prediction_duration_seconds_bucket[5m]))",
                                "legendFormat": "95th percentile"
                            }
                        ],
                        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                    },
                    {
                        "id": 3,
                        "title": "Model Accuracy",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "model_accuracy_score",
                                "legendFormat": "Current Accuracy"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 0, "y": 8}
                    },
                    {
                        "id": 4,
                        "title": "Error Rate",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])",
                                "legendFormat": "Error Rate"
                            }
                        ],
                        "gridPos": {"h": 4, "w": 6, "x": 6, "y": 8}
                    }
                ],
                "time": {"from": "now-1h", "to": "now"},
                "refresh": "30s"
            }
        }
        
        return json.dumps(dashboard, indent=4)
        
    def setup_alertmanager(self) -> bool:
        """Setup Alertmanager for alerts."""
        logger.info("Setting up Alertmanager")
        
        alertmanager_config = self.generate_alertmanager_config()
        alertmanager_deployment = self.generate_alertmanager_deployment()
        
        # Create ConfigMap
        config_manifest = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: alertmanager-config
  namespace: {self.namespace}
data:
  alertmanager.yml: |
{alertmanager_config}
"""
        
        if not self.apply_manifest(config_manifest, "alertmanager-config.yaml"):
            return False
            
        # Deploy Alertmanager
        if not self.apply_manifest(alertmanager_deployment, "alertmanager-deployment.yaml"):
            return False
            
        logger.info("Alertmanager setup completed")
        return True
        
    def generate_alertmanager_config(self) -> str:
        """Generate Alertmanager configuration."""
        config = f"""
    global:
      smtp_smarthost: 'localhost:587'
      smtp_from: 'alerts@mlops-pipeline.com'
    
    route:
      group_by: ['alertname']
      group_wait: 10s
      group_interval: 10s
      repeat_interval: 1h
      receiver: 'web.hook'
      routes:
      - match:
          severity: critical
        receiver: 'critical-alerts'
      - match:
          severity: warning
        receiver: 'warning-alerts'
    
    receivers:
    - name: 'web.hook'
      webhook_configs:
      - url: 'http://webhook-service:8080/webhook'
    
    - name: 'critical-alerts'
      email_configs:
      - to: 'ops-team@mlops-pipeline.com'
        subject: 'CRITICAL: MLOps Pipeline Alert'
        body: |
          Alert: {{{{ .GroupLabels.alertname }}}}
          Environment: {self.environment}
          Severity: {{{{ .GroupLabels.severity }}}}
          Description: {{{{ range .Alerts }}}}{{{{ .Annotations.description }}}}{{{{ end }}}}
    
    - name: 'warning-alerts'
      email_configs:
      - to: 'dev-team@mlops-pipeline.com'
        subject: 'WARNING: MLOps Pipeline Alert'
        body: |
          Alert: {{{{ .GroupLabels.alertname }}}}
          Environment: {self.environment}
          Severity: {{{{ .GroupLabels.severity }}}}
          Description: {{{{ range .Alerts }}}}{{{{ .Annotations.description }}}}{{{{ end }}}}
"""
        return config.strip()
        
    def generate_alertmanager_deployment(self) -> str:
        """Generate Alertmanager deployment manifest."""
        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: alertmanager
  namespace: {self.namespace}
  labels:
    app: alertmanager
spec:
  replicas: 1
  selector:
    matchLabels:
      app: alertmanager
  template:
    metadata:
      labels:
        app: alertmanager
    spec:
      containers:
      - name: alertmanager
        image: prom/alertmanager:v0.25.0
        ports:
        - containerPort: 9093
        args:
          - '--config.file=/etc/alertmanager/alertmanager.yml'
          - '--storage.path=/alertmanager'
        volumeMounts:
        - name: config-volume
          mountPath: /etc/alertmanager
        - name: storage-volume
          mountPath: /alertmanager
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "200m"
      volumes:
      - name: config-volume
        configMap:
          name: alertmanager-config
      - name: storage-volume
        emptyDir: {{}}
---
apiVersion: v1
kind: Service
metadata:
  name: alertmanager
  namespace: {self.namespace}
  labels:
    app: alertmanager
spec:
  type: ClusterIP
  ports:
  - port: 9093
    targetPort: 9093
    name: web
  selector:
    app: alertmanager
"""
        return manifest.strip()
        
    def apply_manifest(self, manifest: str, filename: str) -> bool:
        """Apply Kubernetes manifest."""
        try:
            # Write manifest to temporary file
            manifest_path = f"/tmp/{filename}"
            with open(manifest_path, 'w') as f:
                f.write(manifest)
                
            # Apply manifest
            cmd = f"apply -f {manifest_path}"
            code, stdout, stderr = self.run_kubectl(cmd)
            
            if code != 0:
                logger.error(f"Failed to apply {filename}: {stderr}")
                return False
                
            logger.info(f"Successfully applied {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying manifest {filename}: {e}")
            return False
            
    def wait_for_deployments(self) -> bool:
        """Wait for monitoring deployments to be ready."""
        logger.info("Waiting for monitoring deployments to be ready...")
        
        deployments = ["prometheus", "grafana", "alertmanager"]
        
        for deployment in deployments:
            logger.info(f"Waiting for {deployment} deployment...")
            
            cmd = f"wait --for=condition=available --timeout=300s deployment/{deployment} -n {self.namespace}"
            code, stdout, stderr = self.run_kubectl(cmd)
            
            if code != 0:
                logger.error(f"Deployment {deployment} failed to become ready: {stderr}")
                return False
                
            logger.info(f"Deployment {deployment} is ready")
            
        return True
        
    def setup_monitoring(self) -> bool:
        """Setup complete monitoring stack."""
        logger.info(f"Setting up monitoring for environment: {self.environment}")
        
        try:
            # Step 1: Create monitoring namespace
            if not self.create_monitoring_namespace():
                return False
                
            # Step 2: Setup Prometheus
            if not self.setup_prometheus():
                return False
                
            # Step 3: Setup Grafana
            if not self.setup_grafana():
                return False
                
            # Step 4: Setup Alertmanager
            if not self.setup_alertmanager():
                return False
                
            # Step 5: Wait for deployments
            if not self.wait_for_deployments():
                return False
                
            logger.info("Monitoring setup completed successfully")
            
            # Print access information
            self.print_access_info()
            
            return True
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return False
            
    def print_access_info(self):
        """Print monitoring access information."""
        print("\n" + "="*60)
        print("MONITORING ACCESS INFORMATION")
        print("="*60)
        print(f"Environment: {self.environment}")
        print(f"Namespace: {self.namespace}")
        print(f"\nGrafana Dashboard:")
        print(f"  kubectl port-forward -n {self.namespace} service/grafana 3000:3000")
        print(f"  Then access: http://localhost:3000")
        print(f"  Username: admin")
        print(f"  Password: mlops-admin-password")
        print(f"\nPrometheus:")
        print(f"  kubectl port-forward -n {self.namespace} service/prometheus 9090:9090")
        print(f"  Then access: http://localhost:9090")
        print(f"\nAlertmanager:")
        print(f"  kubectl port-forward -n {self.namespace} service/alertmanager 9093:9093")
        print(f"  Then access: http://localhost:9093")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Setup monitoring infrastructure')
    parser.add_argument('--environment', default='production', help='Environment to setup monitoring for')
    
    args = parser.parse_args()
    
    try:
        setup = MonitoringSetup(environment=args.environment)
        success = setup.setup_monitoring()
        
        if success:
            logger.info("Monitoring setup completed successfully")
            sys.exit(0)
        else:
            logger.error("Monitoring setup failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Setup script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()