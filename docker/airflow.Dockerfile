FROM apache/airflow:2.8.1-python3.9

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements and install Python packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Install additional packages for MLOps
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    scikit-learn==1.3.2 \
    xgboost==2.0.3 \
    lightgbm==4.1.0 \
    optuna==3.5.0 \
    evidently==0.4.11 \
    great-expectations==0.18.8

# Set the PYTHONPATH to include our source directory
ENV PYTHONPATH="/opt/airflow/src:$PYTHONPATH"

# Initialize Airflow database
RUN airflow db init

# Create admin user
RUN airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin