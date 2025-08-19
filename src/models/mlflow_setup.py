"""MLflow tracking server setup and configuration."""
import logging
import os
import subprocess
import yaml
from pathlib import Path
from typing import Dict, Any
import mlflow
from mlflow.tracking import MlflowClient
import psycopg2
from sqlalchemy import create_engine


logger = logging.getLogger(__name__)


class MLflowSetup:
    """Setup and configure MLflow tracking server."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize MLflow setup."""
        self.config = self._load_config(config_path)
        self.client = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def setup_database(self) -> None:
        """Setup PostgreSQL database for MLflow backend store."""
        db_config = self.config["database"]
        
        try:
            # Create database if it doesn't exist
            engine = create_engine(
                f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/postgres"
            )
            
            with engine.connect() as conn:
                # Check if database exists
                result = conn.execute(
                    f"SELECT 1 FROM pg_database WHERE datname='{db_config['name']}'"
                )
                
                if not result.fetchone():
                    # Create database
                    conn.execute("COMMIT")
                    conn.execute(f"CREATE DATABASE {db_config['name']}")
                    logger.info(f"Created database: {db_config['name']}")
                else:
                    logger.info(f"Database {db_config['name']} already exists")
                    
        except Exception as e:
            logger.error(f"Failed to setup database: {e}")
            raise
            
    def initialize_mlflow_server(self) -> None:
        """Initialize MLflow tracking server."""
        mlflow_config = self.config["mlflow"]
        db_config = self.config["database"]
        
        # Set environment variables
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_config["tracking_uri"]
        os.environ["MLFLOW_BACKEND_STORE_URI"] = db_config["connection_string"]
        os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = mlflow_config["artifact_location"]
        
        # Create artifact directory
        artifact_path = Path(mlflow_config["artifact_location"])
        artifact_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("MLflow server environment configured")
        
    def create_experiments(self) -> None:
        """Create default experiments."""
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        client = MlflowClient()
        
        experiments_to_create = [
            {
                "name": self.config["mlflow"]["experiment_name"],
                "artifact_location": self.config["mlflow"]["artifact_location"]
            },
            {
                "name": "model_validation",
                "artifact_location": f"{self.config['mlflow']['artifact_location']}/validation"
            },
            {
                "name": "hyperparameter_optimization",
                "artifact_location": f"{self.config['mlflow']['artifact_location']}/optimization"
            }
        ]
        
        for exp_config in experiments_to_create:
            try:
                experiment = mlflow.get_experiment_by_name(exp_config["name"])
                if experiment:
                    logger.info(f"Experiment '{exp_config['name']}' already exists")
                else:
                    exp_id = mlflow.create_experiment(
                        name=exp_config["name"],
                        artifact_location=exp_config["artifact_location"]
                    )
                    logger.info(f"Created experiment '{exp_config['name']}' with ID: {exp_id}")
            except Exception as e:
                logger.error(f"Failed to create experiment '{exp_config['name']}': {e}")
                
    def setup_model_registry(self) -> None:
        """Setup MLflow model registry."""
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        client = MlflowClient()
        
        # Create registered model
        model_name = self.config["mlflow"]["registered_model_name"]
        
        try:
            # Check if model already exists
            registered_model = client.get_registered_model(model_name)
            logger.info(f"Registered model '{model_name}' already exists")
        except Exception:
            # Create registered model
            client.create_registered_model(
                name=model_name,
                description="Customer churn prediction model with automated MLOps pipeline"
            )
            logger.info(f"Created registered model: {model_name}")
            
    def validate_setup(self) -> Dict[str, bool]:
        """Validate MLflow setup."""
        results = {}
        
        try:
            # Test tracking server connection
            mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
            client = MlflowClient()
            
            # Test experiment creation
            test_exp_name = "mlflow_setup_test"
            try:
                exp_id = mlflow.create_experiment(test_exp_name)
                client.delete_experiment(exp_id)
                results["tracking_server"] = True
            except Exception as e:
                logger.error(f"Tracking server test failed: {e}")
                results["tracking_server"] = False
                
            # Test model registry
            try:
                models = client.search_registered_models()
                results["model_registry"] = True
            except Exception as e:
                logger.error(f"Model registry test failed: {e}")
                results["model_registry"] = False
                
            # Test artifact store
            try:
                artifact_path = Path(self.config["mlflow"]["artifact_location"])
                test_file = artifact_path / "test.txt"
                test_file.write_text("test")
                test_file.unlink()
                results["artifact_store"] = True
            except Exception as e:
                logger.error(f"Artifact store test failed: {e}")
                results["artifact_store"] = False
                
            # Test database connection
            try:
                db_config = self.config["database"]
                engine = create_engine(db_config["connection_string"])
                with engine.connect():
                    pass
                results["database"] = True
            except Exception as e:
                logger.error(f"Database test failed: {e}")
                results["database"] = False
                
        except Exception as e:
            logger.error(f"Setup validation failed: {e}")
            results["general"] = False
            
        return results
        
    def start_mlflow_server(self, host: str = "0.0.0.0", port: int = 5000) -> None:
        """Start MLflow tracking server."""
        db_config = self.config["database"]
        mlflow_config = self.config["mlflow"]
        
        cmd = [
            "mlflow", "server",
            "--backend-store-uri", db_config["connection_string"],
            "--default-artifact-root", mlflow_config["artifact_location"],
            "--host", host,
            "--port", str(port)
        ]
        
        logger.info(f"Starting MLflow server: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start MLflow server: {e}")
            raise
            
    def setup_all(self) -> Dict[str, bool]:
        """Run complete MLflow setup."""
        logger.info("Starting complete MLflow setup")
        
        setup_results = {}
        
        try:
            # Setup database
            self.setup_database()
            setup_results["database_setup"] = True
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            setup_results["database_setup"] = False
            
        try:
            # Initialize MLflow server
            self.initialize_mlflow_server()
            setup_results["server_init"] = True
        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            setup_results["server_init"] = False
            
        try:
            # Create experiments
            self.create_experiments()
            setup_results["experiments_created"] = True
        except Exception as e:
            logger.error(f"Experiment creation failed: {e}")
            setup_results["experiments_created"] = False
            
        try:
            # Setup model registry
            self.setup_model_registry()
            setup_results["model_registry_setup"] = True
        except Exception as e:
            logger.error(f"Model registry setup failed: {e}")
            setup_results["model_registry_setup"] = False
            
        # Validate setup
        validation_results = self.validate_setup()
        setup_results.update(validation_results)
        
        logger.info("MLflow setup completed")
        return setup_results


def main():
    """Main setup function."""
    setup = MLflowSetup()
    results = setup.setup_all()
    
    print("MLflow Setup Results:")
    print("=" * 50)
    for component, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{component.ljust(30)}: {status_str}")
        
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All components setup successfully!")
    else:
        print("\n⚠️  Some components failed setup. Check logs for details.")


if __name__ == "__main__":
    main()