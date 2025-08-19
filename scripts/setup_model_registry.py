#!/usr/bin/env python3
"""Setup script for MLflow model registry and infrastructure."""
import logging
import sys
import os
import subprocess
from pathlib import Path
import time
import requests
import yaml
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.mlflow_setup import MLflowSetup
from models.registry import ModelRegistry
from models.model_manager import ModelManager


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def wait_for_service(host: str, port: int, service_name: str, max_retries: int = 30) -> bool:
    """Wait for a service to be available."""
    logger.info(f"Waiting for {service_name} at {host}:{port}")
    
    for i in range(max_retries):
        try:
            if service_name.lower() == "postgres":
                conn = psycopg2.connect(
                    host=host, port=port, user="mlops_user", 
                    password="mlops_password", database="postgres"
                )
                conn.close()
            elif service_name.lower() == "redis":
                import redis
                client = redis.Redis(host=host, port=port, db=0)
                client.ping()
            elif service_name.lower() == "mlflow":
                response = requests.get(f"http://{host}:{port}/health", timeout=5)
                response.raise_for_status()
            else:
                # Generic TCP check
                import socket
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                sock.close()
                if result != 0:
                    raise ConnectionError("Service not available")
                    
            logger.info(f"✓ {service_name} is available")
            return True
            
        except Exception as e:
            if i == max_retries - 1:
                logger.error(f"✗ {service_name} is not available after {max_retries} retries: {e}")
                return False
            time.sleep(2)
            
    return False


def setup_database():
    """Setup database for MLOps."""
    logger.info("Setting up database...")
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host="localhost", port=5432, user="mlops_user", 
            password="mlops_password", database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute("SELECT 1 FROM pg_database WHERE datname='mlops_db'")
        if not cursor.fetchone():
            cursor.execute("CREATE DATABASE mlops_db")
            logger.info("✓ Created mlops_db database")
        else:
            logger.info("✓ Database mlops_db already exists")
            
        cursor.close()
        conn.close()
        
        # Run initialization script
        init_script = Path(__file__).parent.parent / "docker" / "init-db.sql"
        if init_script.exists():
            subprocess.run([
                "psql", 
                "-h", "localhost", 
                "-p", "5432", 
                "-U", "mlops_user", 
                "-d", "mlops_db", 
                "-f", str(init_script)
            ], env={**os.environ, "PGPASSWORD": "mlops_password"}, check=True)
            logger.info("✓ Database schema initialized")
        else:
            logger.warning("Database initialization script not found")
            
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False
        
    return True


def setup_mlflow():
    """Setup MLflow tracking server and model registry."""
    logger.info("Setting up MLflow...")
    
    try:
        mlflow_setup = MLflowSetup()
        results = mlflow_setup.setup_all()
        
        success_count = sum(1 for v in results.values() if v)
        total_count = len(results)
        
        logger.info(f"MLflow setup: {success_count}/{total_count} components successful")
        
        for component, status in results.items():
            status_str = "✓" if status else "✗"
            logger.info(f"  {status_str} {component}")
            
        return success_count == total_count
        
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        return False


def test_model_registry():
    """Test model registry functionality."""
    logger.info("Testing model registry...")
    
    try:
        registry = ModelRegistry()
        
        # List models
        models = registry.list_models()
        logger.info(f"✓ Found {len(models)} registered models")
        
        # Test model manager
        model_manager = ModelManager()
        
        # Test health check
        health = model_manager.get_model_health("churn-predictor")
        logger.info(f"✓ Model health check completed: {health.get('model_name', 'unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Model registry test failed: {e}")
        return False


def create_sample_model():
    """Create a sample model for testing."""
    logger.info("Creating sample model for testing...")
    
    try:
        import mlflow
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Generate sample data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log model with MLflow
        mlflow.set_tracking_uri("http://localhost:5000")
        
        with mlflow.start_run(run_name="sample_model_setup"):
            # Log parameters
            mlflow.log_param("n_estimators", 10)
            mlflow.log_param("random_state", 42)
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="churn-predictor"
            )
            
        logger.info(f"✓ Sample model created with accuracy: {accuracy:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"Sample model creation failed: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Starting MLOps Model Registry Setup")
    logger.info("=" * 50)
    
    # Check if we're running in Docker environment
    in_docker = os.path.exists("/.dockerenv")
    host = "localhost" if not in_docker else "postgres"
    
    setup_steps = [
        ("PostgreSQL", lambda: wait_for_service(host, 5432, "postgres")),
        ("Redis", lambda: wait_for_service(host, 6379, "redis")),
        ("Database Setup", setup_database),
        ("MLflow Setup", setup_mlflow),
        ("MLflow Server", lambda: wait_for_service("localhost", 5000, "mlflow")),
        ("Model Registry Test", test_model_registry),
        ("Sample Model Creation", create_sample_model),
    ]
    
    results = {}
    
    for step_name, step_func in setup_steps:
        logger.info(f"\n[{step_name}] Starting...")
        try:
            result = step_func()
            results[step_name] = result
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"[{step_name}] {status}")
        except Exception as e:
            results[step_name] = False
            logger.error(f"[{step_name}] ✗ FAILED: {e}")
            
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SETUP SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for step_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{step_name.ljust(25)}: {status}")
        
    logger.info(f"\nOverall: {passed}/{total} steps completed successfully")
    
    if passed == total:
        logger.info("🎉 Model Registry setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Access MLflow UI at: http://localhost:5000")
        logger.info("2. Run model training: python src/models/train.py --data-path data/sample.csv")
        logger.info("3. Deploy models using the ModelManager API")
    else:
        logger.error("⚠️  Setup completed with errors. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()