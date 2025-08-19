# 🚀 Deployment Guide

Production deployment strategies for the MLOps pipeline with Kubernetes, Docker, and cloud infrastructure.

## 🏗️ Deployment Architecture

```
Development → Staging → Production
     ↓          ↓         ↓
   Local     K8s Test   K8s Prod
  Docker      Env       Cluster
   Compose     ↓         ↓
     ↓       Load       Auto
  CI/CD      Testing     Scaling
 Pipeline      ↓         ↓
     ↓      Security   Monitoring
  Build      Scans     & Alerting
  Images       ↓         ↓
     ↓      Manual     Blue/Green
  Registry   Approval   Deployment
```

## 🐳 Container Strategy

### Production Dockerfile

```dockerfile
# docker/api.Dockerfile
FROM python:3.9-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --uid 1001 mlops

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Set ownership
RUN chown -R mlops:mlops /app

USER mlops

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Multi-stage Build for Training

```dockerfile
# docker/training.Dockerfile
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt requirements-training.txt ./
RUN pip install --no-cache-dir -r requirements-training.txt

FROM python:3.9-slim AS runtime

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --uid 1001 mlops

WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

RUN chown -R mlops:mlops /app
USER mlops

CMD ["python", "src/models/train.py"]
```

## ☸️ Kubernetes Deployment

### Namespace Setup

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: mlops-production
  labels:
    name: mlops-production
    environment: production
```

### ConfigMaps and Secrets

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlops-config
  namespace: mlops-production
data:
  ENVIRONMENT: "production"
  MLFLOW_TRACKING_URI: "http://mlflow-service:5000"
  LOG_LEVEL: "info"
  API_WORKERS: "4"

---
apiVersion: v1
kind: Secret
metadata:
  name: mlops-secrets
  namespace: mlops-production
type: Opaque
data:
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  API_SECRET_KEY: <base64-encoded-secret>
```

### API Deployment

```yaml
# k8s/api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlops-api
  namespace: mlops-production
  labels:
    app: mlops-api
    version: v1.2.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mlops-api
  template:
    metadata:
      labels:
        app: mlops-api
        version: v1.2.0
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        fsGroup: 1001
      containers:
      - name: api
        image: your-registry/mlops-api:v1.2.0
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: mlops-secrets
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: mlops-secrets
              key: REDIS_URL
        envFrom:
        - configMapRef:
            name: mlops-config
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          runAsNonRoot: true
          runAsUser: 1001
          capabilities:
            drop:
            - ALL
```

### Service and Ingress

```yaml
# k8s/api-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: mlops-api-service
  namespace: mlops-production
spec:
  selector:
    app: mlops-api
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: mlops-api-ingress
  namespace: mlops-production
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.yourcompany.com
    secretName: mlops-api-tls
  rules:
  - host: api.yourcompany.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: mlops-api-service
            port:
              number: 80
```

### Horizontal Pod Autoscaler

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: mlops-api-hpa
  namespace: mlops-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mlops-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔄 Blue-Green Deployment

### Deployment Strategy

```bash
#!/bin/bash
# scripts/blue_green_deploy.sh

NAMESPACE="mlops-production"
NEW_VERSION=$1
CURRENT_VERSION=$(kubectl get deployment mlops-api -n $NAMESPACE -o jsonpath='{.metadata.labels.version}')

echo "Current version: $CURRENT_VERSION"
echo "Deploying version: $NEW_VERSION"

# Deploy green environment
kubectl apply -f k8s/production/ -n $NAMESPACE

# Update deployment with new image
kubectl set image deployment/mlops-api-green api=your-registry/mlops-api:$NEW_VERSION -n $NAMESPACE

# Wait for rollout
kubectl rollout status deployment/mlops-api-green -n $NAMESPACE

# Run health checks
echo "Running health checks on green environment..."
GREEN_POD=$(kubectl get pods -n $NAMESPACE -l app=mlops-api-green -o jsonpath='{.items[0].metadata.name}')
kubectl exec $GREEN_POD -n $NAMESPACE -- curl -f http://localhost:8000/health

if [ $? -eq 0 ]; then
    echo "Health check passed. Switching traffic..."
    
    # Update service selector to point to green
    kubectl patch service mlops-api-service -n $NAMESPACE -p '{"spec":{"selector":{"version":"'$NEW_VERSION'"}}}'
    
    echo "Traffic switched to version $NEW_VERSION"
    
    # Wait and verify
    sleep 30
    
    # Clean up old blue deployment
    kubectl delete deployment mlops-api -n $NAMESPACE
    kubectl patch deployment mlops-api-green -n $NAMESPACE -p '{"metadata":{"name":"mlops-api"}}'
    
    echo "Blue-green deployment completed successfully"
else
    echo "Health check failed. Rolling back..."
    kubectl delete deployment mlops-api-green -n $NAMESPACE
    exit 1
fi
```

### Canary Deployment

```yaml
# k8s/canary-deployment.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: mlops-api-rollout
  namespace: mlops-production
spec:
  replicas: 5
  strategy:
    canary:
      steps:
      - setWeight: 10
      - pause: {duration: 5m}
      - setWeight: 30
      - pause: {duration: 10m}
      - setWeight: 60
      - pause: {duration: 10m}
      - setWeight: 100
      analysis:
        templates:
        - templateName: success-rate
        args:
        - name: service-name
          value: mlops-api-service
  selector:
    matchLabels:
      app: mlops-api
  template:
    metadata:
      labels:
        app: mlops-api
    spec:
      containers:
      - name: api
        image: your-registry/mlops-api:v1.2.0
        ports:
        - containerPort: 8000
```

## ☁️ Cloud Deployment

### AWS EKS Deployment

```yaml
# terraform/eks-cluster.tf
resource "aws_eks_cluster" "mlops_cluster" {
  name     = "mlops-production"
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = "1.27"

  vpc_config {
    subnet_ids              = aws_subnet.eks_subnet[*].id
    endpoint_private_access = true
    endpoint_public_access  = true
    public_access_cidrs     = ["0.0.0.0/0"]
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_AmazonEKSClusterPolicy,
  ]
}

resource "aws_eks_node_group" "mlops_nodes" {
  cluster_name    = aws_eks_cluster.mlops_cluster.name
  node_group_name = "mlops-nodes"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = aws_subnet.eks_subnet[*].id

  capacity_type  = "ON_DEMAND"
  instance_types = ["t3.medium", "t3.large"]

  scaling_config {
    desired_size = 3
    max_size     = 10
    min_size     = 1
  }

  update_config {
    max_unavailable = 1
  }
}
```

### Google Cloud GKE Deployment

```yaml
# terraform/gke-cluster.tf
resource "google_container_cluster" "mlops_cluster" {
  name     = "mlops-production"
  location = var.gcp_zone

  remove_default_node_pool = true
  initial_node_count       = 1

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  network_policy {
    enabled = true
  }
}

resource "google_container_node_pool" "mlops_nodes" {
  name       = "mlops-node-pool"
  location   = var.gcp_zone
  cluster    = google_container_cluster.mlops_cluster.name
  node_count = 3

  node_config {
    preemptible  = false
    machine_type = "e2-medium"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }
}
```

### Azure AKS Deployment

```yaml
# terraform/aks-cluster.tf
resource "azurerm_kubernetes_cluster" "mlops_cluster" {
  name                = "mlops-production"
  location            = azurerm_resource_group.mlops.location
  resource_group_name = azurerm_resource_group.mlops.name
  dns_prefix          = "mlops"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_D2_v2"
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "azure"
  }
}
```

## 🔒 Security Configuration

### Network Policies

```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mlops-api-netpol
  namespace: mlops-production
spec:
  podSelector:
    matchLabels:
      app: mlops-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - namespaceSelector:
        matchLabels:
          name: redis
    ports:
    - protocol: TCP
      port: 6379
```

### Pod Security Standards

```yaml
# k8s/pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: mlops-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
```

### RBAC Configuration

```yaml
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mlops-api-sa
  namespace: mlops-production

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: mlops-api-role
  namespace: mlops-production
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: mlops-api-rolebinding
  namespace: mlops-production
subjects:
- kind: ServiceAccount
  name: mlops-api-sa
  namespace: mlops-production
roleRef:
  kind: Role
  name: mlops-api-role
  apiGroup: rbac.authorization.k8s.io
```

## 📊 Monitoring in Production

### Prometheus ServiceMonitor

```yaml
# k8s/servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: mlops-api-metrics
  namespace: mlops-production
  labels:
    app: mlops-api
spec:
  selector:
    matchLabels:
      app: mlops-api
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
```

### Grafana Dashboard Deployment

```yaml
# k8s/grafana-dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mlops-dashboard
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
data:
  mlops-production.json: |
    {
      "dashboard": {
        "title": "MLOps Production Dashboard",
        "panels": [...]
      }
    }
```

## 🚀 CI/CD Pipeline

### GitHub Actions Deployment

```yaml
# .github/workflows/deploy-production.yml
name: Deploy to Production

on:
  push:
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
      
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-west-2
        
    - name: Login to ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1
      
    - name: Build and push image
      env:
        ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        ECR_REPOSITORY: mlops-api
        IMAGE_TAG: ${{ github.ref_name }}
      run: |
        docker build -f docker/api.Dockerfile -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
        docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
        
    - name: Deploy to EKS
      env:
        IMAGE_TAG: ${{ github.ref_name }}
      run: |
        aws eks update-kubeconfig --name mlops-production
        
        # Update deployment with new image
        kubectl set image deployment/mlops-api \
          api=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
          -n mlops-production
          
        # Wait for rollout
        kubectl rollout status deployment/mlops-api -n mlops-production
        
        # Verify deployment
        kubectl get pods -n mlops-production -l app=mlops-api
```

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy-staging
  - deploy-production

variables:
  DOCKER_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  KUBECONFIG: /tmp/kubeconfig

build:
  stage: build
  script:
    - docker build -f docker/api.Dockerfile -t $DOCKER_IMAGE .
    - docker push $DOCKER_IMAGE

deploy-production:
  stage: deploy-production
  script:
    - echo $KUBE_CONFIG | base64 -d > $KUBECONFIG
    - kubectl set image deployment/mlops-api api=$DOCKER_IMAGE -n mlops-production
    - kubectl rollout status deployment/mlops-api -n mlops-production
  only:
    - tags
  environment:
    name: production
    url: https://api.yourcompany.com
```

## 📈 Performance Optimization

### Resource Management

```yaml
# k8s/vertical-pod-autoscaler.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: mlops-api-vpa
  namespace: mlops-production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: mlops-api
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: api
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 128Mi
```

### Caching Strategy

```yaml
# k8s/redis-cluster.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: mlops-production
spec:
  serviceName: redis-cluster
  replicas: 3
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## 🔍 Troubleshooting

### Debugging Failed Deployments

```bash
# Check deployment status
kubectl get deployments -n mlops-production

# View deployment events
kubectl describe deployment mlops-api -n mlops-production

# Check pod logs
kubectl logs -l app=mlops-api -n mlops-production --tail=100

# Check pod resource usage
kubectl top pods -n mlops-production

# Debug failing pods
kubectl describe pod <pod-name> -n mlops-production
```

### Health Check Issues

```bash
# Test health endpoint directly
kubectl port-forward svc/mlops-api-service 8080:80 -n mlops-production
curl http://localhost:8080/health

# Check service endpoints
kubectl get endpoints mlops-api-service -n mlops-production

# Verify ingress configuration
kubectl describe ingress mlops-api-ingress -n mlops-production
```

### Performance Issues

```bash
# Check resource utilization
kubectl top nodes
kubectl top pods -n mlops-production

# View HPA status
kubectl get hpa -n mlops-production

# Check for resource constraints
kubectl describe nodes | grep -A 5 "Allocated resources"
```

## 📋 Production Checklist

### Pre-deployment

- [ ] Security scan completed
- [ ] Load testing passed
- [ ] Database migrations ready
- [ ] Monitoring dashboards configured
- [ ] Alert rules defined
- [ ] Rollback procedure documented
- [ ] Team notifications sent

### Deployment

- [ ] Blue-green deployment executed
- [ ] Health checks passing
- [ ] API responding correctly
- [ ] Database connectivity verified
- [ ] Cache warming completed
- [ ] Metrics collection active

### Post-deployment

- [ ] Application performance verified
- [ ] Error rates within acceptable limits
- [ ] Resource utilization normal
- [ ] Customer impact assessment
- [ ] Documentation updated
- [ ] Incident response team notified

## 📚 Next Steps

After successful deployment:

1. **[Monitoring Guide](monitoring.md)**: Set up comprehensive monitoring
2. **[A/B Testing Guide](ab_testing.md)**: Compare model versions
3. **[API Documentation](api.md)**: Understand serving endpoints
4. **[Training Pipeline](training.md)**: Automated model training

---

**Need help?** Check deployment logs with `kubectl logs` or contact the DevOps team.