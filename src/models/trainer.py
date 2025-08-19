"""Model training pipeline implementation."""
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import optuna
# Temporarily commented out due to pydantic compatibility issues
# from evidently import Report
# from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import joblib


logger = logging.getLogger(__name__)


class ModelTrainer:
    """MLOps model training pipeline with experiment tracking."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        self.models = {}
        self.preprocessor = None
        self.experiment_id = None
        
        # Initialize MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        self.experiment_name = self.config["mlflow"]["experiment_name"]
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def _create_or_get_experiment(self) -> str:
        """Create MLflow experiment if it doesn't exist."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment:
                return experiment.experiment_id
        except Exception:
            pass
            
        experiment_id = mlflow.create_experiment(
            name=self.experiment_name,
            artifact_location=self.config["mlflow"]["artifact_location"]
        )
        return experiment_id
        
    def _build_preprocessor(self) -> ColumnTransformer:
        """Build preprocessing pipeline."""
        config = self.config["training"]
        
        # Get feature lists
        categorical_features = config["features"]["categorical_features"]
        numerical_features = config["features"]["numerical_features"]
        
        # Scaling method
        scaling_method = config["features"]["scaling_method"]
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
            
        # Encoding method
        encoding_method = config["features"]["encoding_method"]
        if encoding_method == "onehot":
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        elif encoding_method == "label":
            encoder = LabelEncoder()
        else:
            raise ValueError(f"Unknown encoding method: {encoding_method}")
            
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[('scaler', scaler)])
        categorical_transformer = Pipeline(steps=[('encoder', encoder)])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        return preprocessor
        
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, ...]:
        """Split data into train, validation, and test sets."""
        config = self.config["training"]["data"]
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=config["test_ratio"],
            random_state=config["random_state"],
            stratify=y
        )
        
        # Second split: train vs val
        val_ratio_adjusted = config["validation_ratio"] / (1 - config["test_ratio"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=config["random_state"],
            stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def _get_model(self, model_name: str) -> Any:
        """Get model instance based on configuration."""
        model_config = self.config["training"]["models"][model_name]
        
        if model_name == "xgboost":
            return xgb.XGBClassifier(**model_config)
        elif model_name == "lightgbm":
            return lgb.LGBMClassifier(**model_config)
        elif model_name == "random_forest":
            return RandomForestClassifier(**model_config)
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
    def _evaluate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average='binary'),
            "recall": recall_score(y, y_pred, average='binary'),
            "f1_score": f1_score(y, y_pred, average='binary'),
        }
        
        if y_pred_proba is not None:
            metrics["auc_roc"] = roc_auc_score(y, y_pred_proba)
            
        return metrics
        
    def _cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Perform cross-validation."""
        config = self.config["training"]["validation"]
        cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=42)
        
        scoring = config["scoring"]
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            f"cv_{scoring}_mean": scores.mean(),
            f"cv_{scoring}_std": scores.std(),
        }
        
    def _check_performance_thresholds(self, metrics: Dict[str, float]) -> bool:
        """Check if model meets performance thresholds."""
        thresholds = self.config["training"]["thresholds"]
        
        checks = [
            metrics.get("accuracy", 0) >= thresholds["min_accuracy"],
            metrics.get("precision", 0) >= thresholds["min_precision"],
            metrics.get("recall", 0) >= thresholds["min_recall"],
            metrics.get("f1_score", 0) >= thresholds["min_f1_score"],
            metrics.get("auc_roc", 0) >= thresholds["min_auc_roc"],
        ]
        
        return all(checks)
        
    def optimize_hyperparameters(self, model_name: str, X_train: pd.DataFrame, 
                                y_train: pd.Series, X_val: pd.DataFrame, 
                                y_val: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        config = self.config["training"]["hyperparameter_tuning"]
        
        def objective(trial):
            # Define hyperparameter search space based on model
            if model_name == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "random_state": 42
                }
                model = xgb.XGBClassifier(**params)
            elif model_name == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "random_state": 42
                }
                model = lgb.LGBMClassifier(**params)
            elif model_name == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                    "random_state": 42
                }
                model = RandomForestClassifier(**params)
            else:
                raise ValueError(f"Hyperparameter optimization not supported for {model_name}")
                
            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, y_pred_proba)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=config["n_trials"],
            timeout=config["timeout"]
        )
        
        return study.best_params
        
    def train_models(self, data_path: str) -> Dict[str, Any]:
        """Train all models and track experiments."""
        logger.info("Starting model training pipeline")
        
        # Load data
        data = pd.read_csv(data_path)
        target_column = self.config["training"]["data"]["target_column"]
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self._split_data(X, y)
        
        # Build preprocessor
        self.preprocessor = self._build_preprocessor()
        
        # Fit preprocessor and transform data
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_val_transformed = self.preprocessor.transform(X_val)
        X_test_transformed = self.preprocessor.transform(X_test)
        
        # Convert to DataFrame for consistency
        feature_names = self.preprocessor.get_feature_names_out()
        X_train_transformed = pd.DataFrame(X_train_transformed, columns=feature_names)
        X_val_transformed = pd.DataFrame(X_val_transformed, columns=feature_names)
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
        
        # Create experiment
        self.experiment_id = self._create_or_get_experiment()
        
        results = {}
        
        # Train each model
        for model_name in self.config["training"]["models"].keys():
            logger.info(f"Training {model_name}")
            
            with mlflow.start_run(experiment_id=self.experiment_id, run_name=f"{model_name}_training"):
                # Get base model
                model = self._get_model(model_name)
                
                # Hyperparameter optimization if enabled
                if self.config["training"]["hyperparameter_tuning"]["enabled"]:
                    logger.info(f"Optimizing hyperparameters for {model_name}")
                    best_params = self.optimize_hyperparameters(
                        model_name, X_train_transformed, y_train, X_val_transformed, y_val
                    )
                    mlflow.log_params(best_params)
                    
                    # Create model with best parameters
                    if model_name == "xgboost":
                        model = xgb.XGBClassifier(**best_params)
                    elif model_name == "lightgbm":
                        model = lgb.LGBMClassifier(**best_params)
                    elif model_name == "random_forest":
                        model = RandomForestClassifier(**best_params)
                
                # Train model
                model.fit(X_train_transformed, y_train)
                
                # Evaluate on validation set
                val_metrics = self._evaluate_model(model, X_val_transformed, y_val)
                
                # Cross-validation
                cv_metrics = self._cross_validate_model(model, X_train_transformed, y_train)
                
                # Combine metrics
                all_metrics = {**val_metrics, **cv_metrics}
                
                # Log metrics
                mlflow.log_metrics(all_metrics)
                
                # Log model parameters
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())
                
                # Check performance thresholds
                meets_threshold = self._check_performance_thresholds(all_metrics)
                mlflow.log_param("meets_threshold", meets_threshold)
                
                if meets_threshold:
                    # Evaluate on test set
                    test_metrics = self._evaluate_model(model, X_test_transformed, y_test)
                    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
                    mlflow.log_metrics(test_metrics)
                    
                    # Log model
                    if model_name == "xgboost":
                        mlflow.xgboost.log_model(model, f"{model_name}_model")
                    elif model_name == "lightgbm":
                        mlflow.lightgbm.log_model(model, f"{model_name}_model")
                    else:
                        mlflow.sklearn.log_model(model, f"{model_name}_model")
                    
                    # Save preprocessor
                    mlflow.sklearn.log_model(self.preprocessor, "preprocessor")
                    
                    self.models[model_name] = {
                        "model": model,
                        "metrics": {**all_metrics, **test_metrics},
                        "meets_threshold": meets_threshold
                    }
                    
                    logger.info(f"{model_name} training completed successfully")
                else:
                    logger.warning(f"{model_name} does not meet performance thresholds")
                    self.models[model_name] = {
                        "model": model,
                        "metrics": all_metrics,
                        "meets_threshold": meets_threshold
                    }
                
                results[model_name] = self.models[model_name]
                
        logger.info("Model training pipeline completed")
        return results
        
    def save_models(self, output_dir: str = "models/artifacts") -> None:
        """Save trained models and preprocessor."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save preprocessor
        joblib.dump(self.preprocessor, output_path / "preprocessor.joblib")
        
        # Save models
        for name, model_info in self.models.items():
            if model_info["meets_threshold"]:
                model_path = output_path / f"{name}_model.joblib"
                joblib.dump(model_info["model"], model_path)
                
                # Save metrics
                metrics_path = output_path / f"{name}_metrics.yaml"
                with open(metrics_path, 'w') as f:
                    yaml.dump(model_info["metrics"], f)
                    
        logger.info(f"Models saved to {output_dir}")
        
    def get_best_model(self) -> Tuple[str, Any, Dict[str, float]]:
        """Get the best performing model based on validation AUC."""
        best_model_name = None
        best_score = 0
        
        for name, model_info in self.models.items():
            if model_info["meets_threshold"]:
                score = model_info["metrics"].get("auc_roc", 0)
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    
        if best_model_name:
            return (
                best_model_name,
                self.models[best_model_name]["model"],
                self.models[best_model_name]["metrics"]
            )
        else:
            raise ValueError("No models meet the performance thresholds")