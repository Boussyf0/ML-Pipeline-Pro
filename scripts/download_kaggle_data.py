#!/usr/bin/env python3
"""Download and prepare Telco Customer Churn dataset from Kaggle."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import subprocess
import sys
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_kaggle_api():
    """Setup Kaggle API and verify credentials."""
    try:
        import kaggle
        logger.info("Kaggle API found")
        return True
    except ImportError:
        logger.error("Kaggle API not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        import kaggle
        return True
    except Exception as e:
        logger.error(f"Error setting up Kaggle API: {e}")
        return False


def download_telco_churn_data(output_dir: str = "data/raw"):
    """Download Telco Customer Churn dataset from Kaggle."""
    try:
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Dataset identifier for IBM Telco Customer Churn
        dataset = "blastchar/telco-customer-churn"
        
        logger.info(f"Downloading dataset: {dataset}")
        
        # Use kaggle CLI to download
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", dataset,
            "-p", output_dir,
            "--unzip"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("Dataset downloaded successfully")
            return True
        else:
            logger.error(f"Failed to download dataset: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return False


def prepare_telco_data(data_path: str) -> pd.DataFrame:
    """Load and prepare the Telco Customer Churn data."""
    logger.info(f"Loading data from {data_path}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    logger.info(f"Original data shape: {df.shape}")
    
    # Data cleaning and preparation
    df_clean = df.copy()
    
    # Handle TotalCharges column (it's sometimes stored as object with spaces)
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        # Fill NaN values with 0 (new customers)
        df_clean['TotalCharges'].fillna(0, inplace=True)
    
    # Standardize column names to match our pipeline
    column_mapping = {
        'customerID': 'customer_id',
        'gender': 'gender',
        'SeniorCitizen': 'senior_citizen',
        'Partner': 'partner',
        'Dependents': 'dependents',
        'tenure': 'tenure',
        'PhoneService': 'phone_service',
        'MultipleLines': 'multiple_lines',
        'InternetService': 'internet_service',
        'OnlineSecurity': 'online_security',
        'OnlineBackup': 'online_backup',
        'DeviceProtection': 'device_protection',
        'TechSupport': 'tech_support',
        'StreamingTV': 'streaming_tv',
        'StreamingMovies': 'streaming_movies',
        'Contract': 'contract',
        'PaperlessBilling': 'paperless_billing',
        'PaymentMethod': 'payment_method',
        'MonthlyCharges': 'monthly_charges',
        'TotalCharges': 'total_charges',
        'Churn': 'churn'
    }
    
    # Rename columns
    df_clean = df_clean.rename(columns=column_mapping)
    
    # Ensure consistent data types
    categorical_cols = [
        'gender', 'partner', 'dependents', 'phone_service', 'multiple_lines',
        'internet_service', 'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies', 'contract',
        'paperless_billing', 'payment_method', 'churn'
    ]
    
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str)
    
    # Ensure numerical columns are numeric
    numerical_cols = ['senior_citizen', 'tenure', 'monthly_charges', 'total_charges']
    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove any remaining missing values
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    removed_rows = initial_rows - len(df_clean)
    
    if removed_rows > 0:
        logger.info(f"Removed {removed_rows} rows with missing values")
    
    logger.info(f"Cleaned data shape: {df_clean.shape}")
    logger.info(f"Churn rate: {(df_clean['churn'] == 'Yes').mean():.2%}")
    
    return df_clean


def create_train_val_split(df: pd.DataFrame, output_dir: str, 
                          train_ratio: float = 0.8, random_state: int = 42):
    """Create train/validation split and save to separate files."""
    from sklearn.model_selection import train_test_split
    
    # Stratify by churn to maintain class balance
    df_train, df_val = train_test_split(
        df, 
        test_size=(1 - train_ratio),
        random_state=random_state,
        stratify=df['churn']
    )
    
    # Save training data
    train_path = Path(output_dir) / "customer_data.csv"
    df_train.to_csv(train_path, index=False)
    logger.info(f"Training data saved: {train_path} ({len(df_train)} samples)")
    
    # Save validation data
    val_path = Path(output_dir) / "validation_data.csv" 
    df_val.to_csv(val_path, index=False)
    logger.info(f"Validation data saved: {val_path} ({len(df_val)} samples)")
    
    return df_train, df_val


def create_feature_config(df: pd.DataFrame, output_dir: str):
    """Create feature configuration file."""
    # Identify feature types
    categorical_features = []
    numerical_features = []
    
    for col in df.columns:
        if col in ['customer_id', 'churn']:  # Skip ID and target
            continue
            
        if df[col].dtype == 'object' or col in ['senior_citizen']:
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    feature_config = {
        'categorical_features': categorical_features,
        'numerical_features': numerical_features,
        'target_column': 'churn',
        'id_column': 'customer_id'
    }
    
    # Save configuration
    config_dir = Path(output_dir).parent / "processed"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "feature_config.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(feature_config, f, default_flow_style=False)
    
    logger.info(f"Feature configuration saved: {config_path}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    
    return feature_config


def update_main_config(feature_config: dict):
    """Update main config file with feature information."""
    config_path = Path("config/config.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Update training configuration
    if 'training' not in config:
        config['training'] = {}
    if 'features' not in config['training']:
        config['training']['features'] = {}
    
    config['training']['features'].update(feature_config)
    
    # Update data configuration
    if 'data' not in config['training']:
        config['training']['data'] = {}
    
    config['training']['data'].update({
        'target_column': feature_config['target_column'],
        'train_ratio': 0.7,
        'validation_ratio': 0.15,
        'test_ratio': 0.15,
        'random_state': 42
    })
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Main configuration updated: {config_path}")


def main():
    """Main function to download and prepare Kaggle data."""
    parser = argparse.ArgumentParser(description="Download and prepare Telco Customer Churn data from Kaggle")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Output directory for downloaded data")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio for training data split")
    parser.add_argument("--download-only", action="store_true",
                       help="Only download data, don't process")
    
    args = parser.parse_args()
    
    # Check Kaggle API setup
    logger.info("=== Setting up Kaggle API ===")
    if not setup_kaggle_api():
        logger.error("Failed to setup Kaggle API")
        logger.error("Please ensure you have:")
        logger.error("1. Kaggle account and API credentials")
        logger.error("2. kaggle.json file in ~/.kaggle/ directory")
        logger.error("3. Accepted the competition/dataset terms")
        sys.exit(1)
    
    # Download dataset
    logger.info("=== Downloading Dataset ===")
    if not download_telco_churn_data(args.output_dir):
        logger.error("Failed to download dataset")
        sys.exit(1)
    
    if args.download_only:
        logger.info("Download completed. Use --process to prepare data.")
        return
    
    # Find downloaded CSV file
    data_dir = Path(args.output_dir)
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error("No CSV files found in output directory")
        sys.exit(1)
    
    # Use the first CSV file found (should be WA_Fn-UseC_-Telco-Customer-Churn.csv)
    data_path = csv_files[0]
    logger.info(f"Processing data file: {data_path}")
    
    # Prepare data
    logger.info("=== Preparing Data ===")
    df = prepare_telco_data(data_path)
    
    # Create train/validation split
    logger.info("=== Creating Train/Validation Split ===")
    df_train, df_val = create_train_val_split(df, args.output_dir, args.train_ratio)
    
    # Create feature configuration
    logger.info("=== Creating Feature Configuration ===")
    feature_config = create_feature_config(df, args.output_dir)
    
    # Update main configuration file
    logger.info("=== Updating Main Configuration ===")
    update_main_config(feature_config)
    
    # Summary
    logger.info("\n=== Dataset Summary ===")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Training samples: {len(df_train)}")
    logger.info(f"Validation samples: {len(df_val)}")
    logger.info(f"Features: {len(feature_config['categorical_features']) + len(feature_config['numerical_features'])}")
    logger.info(f"Churn rate (train): {(df_train['churn'] == 'Yes').mean():.2%}")
    logger.info(f"Churn rate (validation): {(df_val['churn'] == 'Yes').mean():.2%}")
    
    logger.info("\n✅ Kaggle data setup completed!")
    logger.info("Next steps:")
    logger.info("1. Train model: python src/models/train.py --data-path data/raw/customer_data.csv")
    logger.info("2. Start API: python scripts/start_api.py")
    logger.info("3. Run validation: python scripts/validate_model.py")


if __name__ == "__main__":
    main()