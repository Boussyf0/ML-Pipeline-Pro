FROM python:3.11-slim

WORKDIR /mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and dependencies
RUN pip install \
    mlflow==2.9.2 \
    psycopg2-binary==2.9.9 \
    boto3

# Create artifacts directory
RUN mkdir -p /mlflow/artifacts

# Expose MLflow port
EXPOSE 5000

# Run MLflow server
CMD mlflow server \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root ${ARTIFACT_ROOT} \
    --host 0.0.0.0 \
    --port ${MLFLOW_PORT}