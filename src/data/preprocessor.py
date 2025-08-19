"""Data preprocessing and validation utilities."""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import great_expectations as ge
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.data_context import FileDataContext


logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing pipeline with validation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize preprocessor with configuration."""
        self.config = self._load_config(config_path)
        self.preprocessor = None
        self.feature_names = None
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
            
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality using Great Expectations."""
        try:
            # Create expectations
            gx_df = ge.from_pandas(df)
            
            # Basic data quality checks
            validation_results = {}
            
            # Check for missing values
            for column in df.columns:
                expectation = gx_df.expect_column_values_to_not_be_null(column)
                validation_results[f"{column}_not_null"] = expectation.success
                
            # Check data types
            config = self.config["training"]["features"]
            
            # Validate categorical features
            for col in config["categorical_features"]:
                if col in df.columns:
                    expectation = gx_df.expect_column_values_to_be_of_type(col, "object")
                    validation_results[f"{col}_categorical"] = expectation.success
                    
            # Validate numerical features
            for col in config["numerical_features"]:
                if col in df.columns:
                    expectation = gx_df.expect_column_values_to_be_of_type(col, ["int64", "float64"])
                    validation_results[f"{col}_numerical"] = expectation.success
                    
                    # Check for outliers (values outside 3 standard deviations)
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    lower_bound = mean_val - 3 * std_val
                    upper_bound = mean_val + 3 * std_val
                    
                    expectation = gx_df.expect_column_values_to_be_between(
                        col, min_value=lower_bound, max_value=upper_bound, mostly=0.95
                    )
                    validation_results[f"{col}_outliers"] = expectation.success
                    
            # Check target column if exists
            target_col = self.config["training"]["data"]["target_column"]
            if target_col in df.columns:
                unique_values = df[target_col].nunique()
                validation_results["target_binary"] = unique_values == 2
                validation_results["target_distribution"] = df[target_col].value_counts().to_dict()
                
            # Overall data quality score
            passed_checks = sum(1 for v in validation_results.values() if isinstance(v, bool) and v)
            total_checks = sum(1 for v in validation_results.values() if isinstance(v, bool))
            validation_results["data_quality_score"] = passed_checks / total_checks if total_checks > 0 else 0
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            return {"validation_error": str(e)}
            
    def detect_data_drift(self, reference_df: pd.DataFrame, 
                         current_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and current datasets."""
        try:
            from scipy import stats
            from scipy.spatial.distance import wasserstein_distance
            
            drift_results = {}
            numerical_features = self.config["training"]["features"]["numerical_features"]
            
            for feature in numerical_features:
                if feature in reference_df.columns and feature in current_df.columns:
                    ref_data = reference_df[feature].dropna()
                    current_data = current_df[feature].dropna()
                    
                    if len(ref_data) > 0 and len(current_data) > 0:
                        # Kolmogorov-Smirnov test
                        ks_statistic, ks_p_value = stats.ks_2samp(ref_data, current_data)
                        
                        # Wasserstein distance
                        wasserstein_dist = wasserstein_distance(ref_data, current_data)
                        
                        # Population Stability Index (PSI)
                        psi_score = self._calculate_psi(ref_data, current_data)
                        
                        drift_results[feature] = {
                            "ks_statistic": ks_statistic,
                            "ks_p_value": ks_p_value,
                            "wasserstein_distance": wasserstein_dist,
                            "psi_score": psi_score,
                            "drift_detected": self._is_drift_detected(
                                ks_statistic, wasserstein_dist, psi_score
                            )
                        }
                        
            return drift_results
            
        except Exception as e:
            logger.error(f"Data drift detection failed: {e}")
            return {"drift_error": str(e)}
            
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, 
                      buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=buckets)
            
            # Get counts for each bin
            ref_counts = np.histogram(reference, bins=bin_edges)[0]
            current_counts = np.histogram(current, bins=bin_edges)[0]
            
            # Calculate percentages
            ref_pct = ref_counts / len(reference)
            current_pct = current_counts / len(current)
            
            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            current_pct = np.where(current_pct == 0, 0.0001, current_pct)
            
            # Calculate PSI
            psi = np.sum((current_pct - ref_pct) * np.log(current_pct / ref_pct))
            
            return psi
            
        except Exception:
            return float('inf')
            
    def _is_drift_detected(self, ks_stat: float, wasserstein_dist: float, 
                          psi_score: float) -> bool:
        """Determine if drift is detected based on thresholds."""
        config = self.config["monitoring"]["drift_detection"]
        thresholds = config["thresholds"]
        
        return (
            ks_stat > thresholds["ks"] or
            wasserstein_dist > thresholds["wasserstein"] or
            psi_score > thresholds["psi"]
        )
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for training."""
        logger.info("Starting data cleaning")
        
        # Make a copy
        df_clean = df.copy()
        
        # Handle missing values
        numerical_features = self.config["training"]["features"]["numerical_features"]
        categorical_features = self.config["training"]["features"]["categorical_features"]
        
        # Fill numerical missing values with median
        for col in numerical_features:
            if col in df_clean.columns:
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                
        # Fill categorical missing values with mode
        for col in categorical_features:
            if col in df_clean.columns:
                mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
                
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        removed_duplicates = initial_rows - len(df_clean)
        
        if removed_duplicates > 0:
            logger.info(f"Removed {removed_duplicates} duplicate rows")
            
        # Handle outliers for numerical features
        for col in numerical_features:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                
        logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
        return df_clean
        
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        logger.info("Starting feature engineering")
        
        df_engineered = df.copy()
        
        # Example feature engineering for customer churn
        if 'tenure' in df_engineered.columns and 'monthly_charges' in df_engineered.columns:
            # Total amount spent
            df_engineered['total_amount'] = df_engineered['tenure'] * df_engineered['monthly_charges']
            
            # Average monthly charges per tenure month
            df_engineered['avg_monthly_charges'] = df_engineered['monthly_charges'] / (df_engineered['tenure'] + 1)
            
            # Tenure categories
            df_engineered['tenure_category'] = pd.cut(
                df_engineered['tenure'],
                bins=[0, 12, 24, 48, float('inf')],
                labels=['New', 'Medium', 'Long', 'Very_Long']
            )
            
            # Monthly charges categories
            df_engineered['charges_category'] = pd.cut(
                df_engineered['monthly_charges'],
                bins=[0, 35, 65, float('inf')],
                labels=['Low', 'Medium', 'High']
            )
            
        # Add these new features to categorical list for processing
        new_categorical = ['tenure_category', 'charges_category']
        existing_categorical = self.config["training"]["features"]["categorical_features"]
        
        # Update feature lists in config (for this session)
        self.config["training"]["features"]["categorical_features"] = existing_categorical + new_categorical
        self.config["training"]["features"]["numerical_features"] += ['total_amount', 'avg_monthly_charges']
        
        logger.info(f"Feature engineering completed. New shape: {df_engineered.shape}")
        return df_engineered
        
    def prepare_data(self, data_path: str, validation_data_path: Optional[str] = None) -> Tuple[pd.DataFrame, ...]:
        """Complete data preparation pipeline."""
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Validate data quality
        validation_results = self.validate_data(df)
        logger.info(f"Data quality score: {validation_results.get('data_quality_score', 'N/A')}")
        
        # Check for data drift if validation data provided
        if validation_data_path and Path(validation_data_path).exists():
            validation_df = pd.read_csv(validation_data_path)
            drift_results = self.detect_data_drift(df, validation_df)
            logger.info(f"Data drift analysis completed for {len(drift_results)} features")
            
        # Clean data
        df_clean = self.clean_data(df)
        
        # Feature engineering
        df_engineered = self.feature_engineering(df_clean)
        
        # Split features and target
        target_column = self.config["training"]["data"]["target_column"]
        if target_column not in df_engineered.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
            
        X = df_engineered.drop(columns=[target_column])
        y = df_engineered[target_column]
        
        # Split data
        config = self.config["training"]["data"]
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(1 - config["train_ratio"]),
            random_state=config["random_state"],
            stratify=y
        )
        
        val_test_ratio = config["validation_ratio"] / (config["validation_ratio"] + config["test_ratio"])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_test_ratio),
            random_state=config["random_state"],
            stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test