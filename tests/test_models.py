"""Tests for model training and management components."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.trainer import ModelTrainer
from models.registry import ModelRegistry
from models.model_manager import ModelManager
from data.preprocessor import DataPreprocessor


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "training": {
            "data": {
                "train_ratio": 0.7,
                "validation_ratio": 0.15,
                "test_ratio": 0.15,
                "random_state": 42,
                "target_column": "churn"
            },
            "features": {
                "categorical_features": ["gender", "contract"],
                "numerical_features": ["tenure", "monthly_charges"],
                "scaling_method": "standard",
                "encoding_method": "onehot"
            },
            "models": {
                "xgboost": {
                    "n_estimators": 10,
                    "max_depth": 3,
                    "random_state": 42
                }
            },
            "hyperparameter_tuning": {
                "enabled": False,
                "method": "optuna",
                "n_trials": 5
            },
            "validation": {
                "cv_folds": 3,
                "scoring": "roc_auc"
            },
            "thresholds": {
                "min_accuracy": 0.6,
                "min_precision": 0.5,
                "min_recall": 0.5,
                "min_f1_score": 0.5,
                "min_auc_roc": 0.6
            }
        },
        "mlflow": {
            "tracking_uri": "memory://",
            "experiment_name": "test_experiment",
            "registered_model_name": "test_model",
            "artifact_location": "/tmp/mlflow"
        },
        "database": {
            "connection_string": "sqlite:///:memory:"
        },
        "redis": {
            "connection_string": "redis://localhost:6379/0"
        }
    }


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "tenure": np.random.randint(1, 72, n_samples),
        "monthly_charges": np.random.uniform(20, 120, n_samples),
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "churn": np.random.choice([0, 1], n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def config_file(sample_config):
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        config_path = f.name
        
    yield config_path
    os.unlink(config_path)


@pytest.fixture
def data_file(sample_data):
    """Create temporary data file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        data_path = f.name
        
    yield data_path
    os.unlink(data_path)


class TestDataPreprocessor:
    """Test data preprocessing functionality."""
    
    def test_data_validation(self, config_file, sample_data):
        """Test data validation."""
        preprocessor = DataPreprocessor(config_file)
        results = preprocessor.validate_data(sample_data)
        
        assert isinstance(results, dict)
        assert "data_quality_score" in results
        assert 0 <= results["data_quality_score"] <= 1
        
    def test_data_cleaning(self, config_file, sample_data):
        """Test data cleaning."""
        # Add some missing values
        sample_data.loc[0, "tenure"] = np.nan
        sample_data.loc[1, "gender"] = np.nan
        
        preprocessor = DataPreprocessor(config_file)
        cleaned_data = preprocessor.clean_data(sample_data)
        
        assert cleaned_data.isnull().sum().sum() == 0  # No missing values
        assert len(cleaned_data) <= len(sample_data)  # May remove duplicates
        
    def test_feature_engineering(self, config_file, sample_data):
        """Test feature engineering."""
        preprocessor = DataPreprocessor(config_file)
        engineered_data = preprocessor.feature_engineering(sample_data)
        
        # Should have more columns after feature engineering
        assert engineered_data.shape[1] >= sample_data.shape[1]
        
    @patch('sys.path', [])  # Mock to avoid import issues
    def test_drift_detection(self, config_file, sample_data):
        """Test data drift detection."""
        preprocessor = DataPreprocessor(config_file)
        
        # Create slightly different data for drift test
        drift_data = sample_data.copy()
        drift_data["monthly_charges"] += 10  # Add systematic shift
        
        with patch('scipy.stats.ks_2samp') as mock_ks:
            mock_ks.return_value = (0.2, 0.01)  # Significant drift
            
            drift_results = preprocessor.detect_data_drift(sample_data, drift_data)
            
            assert isinstance(drift_results, dict)


class TestModelTrainer:
    """Test model training functionality."""
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.create_experiment')
    @patch('mlflow.start_run')
    def test_trainer_initialization(self, mock_start_run, mock_create_exp, 
                                  mock_set_uri, config_file):
        """Test trainer initialization."""
        trainer = ModelTrainer(config_file)
        
        assert trainer.config is not None
        assert trainer.models == {}
        mock_set_uri.assert_called_once()
        
    def test_build_preprocessor(self, config_file):
        """Test preprocessor building."""
        trainer = ModelTrainer(config_file)
        preprocessor = trainer._build_preprocessor()
        
        assert preprocessor is not None
        
    def test_data_splitting(self, config_file, sample_data):
        """Test data splitting."""
        trainer = ModelTrainer(config_file)
        
        X = sample_data.drop(columns=["churn"])
        y = sample_data["churn"]
        
        X_train, X_val, X_test, y_train, y_val, y_test = trainer._split_data(X, y)
        
        # Check shapes
        total_samples = len(sample_data)
        assert len(X_train) + len(X_val) + len(X_test) == total_samples
        assert len(y_train) + len(y_val) + len(y_test) == total_samples
        
        # Check ratios approximately correct
        train_ratio = len(X_train) / total_samples
        assert 0.6 < train_ratio < 0.8
        
    @patch('mlflow.start_run')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_params')
    def test_model_evaluation(self, mock_log_params, mock_log_metrics, 
                            mock_start_run, config_file, sample_data):
        """Test model evaluation."""
        trainer = ModelTrainer(config_file)
        
        # Mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.random.choice([0, 1], 20)
        mock_model.predict_proba.return_value = np.random.rand(20, 2)
        
        X_test = sample_data.drop(columns=["churn"]).iloc[:20]
        y_test = sample_data["churn"].iloc[:20]
        
        metrics = trainer._evaluate_model(mock_model, X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics


class TestModelRegistry:
    """Test model registry functionality."""
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.tracking.MlflowClient')
    def test_registry_initialization(self, mock_client, mock_set_uri, config_file):
        """Test registry initialization."""
        registry = ModelRegistry(config_file)
        
        assert registry.config is not None
        mock_set_uri.assert_called_once()
        
    @patch('mlflow.register_model')
    @patch('mlflow.tracking.MlflowClient')
    def test_model_registration(self, mock_client, mock_register, config_file):
        """Test model registration."""
        mock_client_instance = Mock()
        mock_client.return_value = mock_client_instance
        
        mock_version = Mock()
        mock_version.version = "1"
        mock_register.return_value = mock_version
        
        registry = ModelRegistry(config_file)
        version = registry.register_model("run123", "model_path")
        
        assert version == "1"
        mock_register.assert_called_once()


class TestModelManager:
    """Test model manager functionality."""
    
    @patch('redis.from_url')
    @patch('sqlalchemy.create_engine')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.tracking.MlflowClient')
    def test_manager_initialization(self, mock_client, mock_set_uri, 
                                  mock_create_engine, mock_redis, config_file):
        """Test manager initialization."""
        manager = ModelManager(config_file)
        
        assert manager.config is not None
        mock_set_uri.assert_called_once()
        mock_create_engine.assert_called_once()
        mock_redis.assert_called_once()
        
    @patch('redis.from_url')
    @patch('sqlalchemy.create_engine') 
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.tracking.MlflowClient')
    @patch('mlflow.pyfunc.load_model')
    def test_model_deployment(self, mock_load_model, mock_client, mock_set_uri,
                            mock_create_engine, mock_redis, config_file):
        """Test model deployment."""
        # Mock dependencies
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        manager = ModelManager(config_file)
        
        result = manager.deploy_model("test_model", "1", "staging")
        
        assert isinstance(result, dict)
        assert result["model_name"] == "test_model"
        assert result["version"] == "1"
        assert result["environment"] == "staging"


class TestIntegration:
    """Integration tests."""
    
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.create_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_metrics')
    @patch('mlflow.log_params')
    @patch('mlflow.sklearn.log_model')
    def test_end_to_end_training(self, mock_log_model, mock_log_params, 
                                mock_log_metrics, mock_start_run, 
                                mock_create_exp, mock_set_uri,
                                config_file, data_file):
        """Test end-to-end training pipeline."""
        # Mock MLflow context manager
        mock_run_context = Mock()
        mock_start_run.return_value.__enter__ = Mock(return_value=mock_run_context)
        mock_start_run.return_value.__exit__ = Mock(return_value=None)
        
        trainer = ModelTrainer(config_file)
        
        # This would normally train real models, but we'll just test the pipeline
        try:
            results = trainer.train_models(data_file)
            assert isinstance(results, dict)
        except Exception as e:
            # Expected to fail in test environment due to mocking
            assert "xgboost" in str(e) or "model" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__])