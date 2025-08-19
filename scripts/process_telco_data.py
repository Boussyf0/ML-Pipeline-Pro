#!/usr/bin/env python3
"""Process the existing Telco customer dataset for the MLOps pipeline."""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_clean_telco_data(data_path: str) -> pd.DataFrame:
    """Load and clean the Telco dataset."""
    logger.info(f"Loading data from {data_path}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    logger.info(f"Original data shape: {df.shape}")
    
    # Create a cleaned version with standardized column names
    df_clean = df.copy()
    
    # Standardize column names (lowercase, replace spaces with underscores)
    df_clean.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df_clean.columns]
    
    # Key column mappings to match our pipeline expectations
    column_mapping = {
        'customer_id': 'customer_id',
        'gender': 'gender',
        'age': 'age',
        'senior_citizen': 'senior_citizen',
        'married': 'partner',  # Map married to partner
        'dependents': 'dependents',
        'tenure_in_months': 'tenure',
        'phone_service': 'phone_service',
        'multiple_lines': 'multiple_lines',
        'internet_service': 'internet_service',
        'internet_type': 'internet_type',
        'online_security': 'online_security',
        'online_backup': 'online_backup',
        'device_protection_plan': 'device_protection',
        'premium_tech_support': 'tech_support',
        'streaming_tv': 'streaming_tv',
        'streaming_movies': 'streaming_movies',
        'streaming_music': 'streaming_music',
        'contract': 'contract',
        'paperless_billing': 'paperless_billing',
        'payment_method': 'payment_method',
        'monthly_charge': 'monthly_charges',
        'total_charges': 'total_charges',
        'churn_label': 'churn',
        'satisfaction_score': 'satisfaction_score',
        'cltv': 'customer_lifetime_value'
    }
    
    # Rename columns that exist
    existing_mappings = {old: new for old, new in column_mapping.items() if old in df_clean.columns}
    df_clean = df_clean.rename(columns=existing_mappings)
    
    # Select relevant columns for our model
    model_columns = [
        'customer_id', 'gender', 'age', 'senior_citizen', 'partner', 'dependents',
        'tenure', 'phone_service', 'multiple_lines', 'internet_service', 'internet_type',
        'online_security', 'online_backup', 'device_protection', 'tech_support',
        'streaming_tv', 'streaming_movies', 'contract', 'paperless_billing',
        'payment_method', 'monthly_charges', 'total_charges', 'churn'
    ]
    
    # Keep only columns that exist
    available_columns = [col for col in model_columns if col in df_clean.columns]
    df_model = df_clean[available_columns].copy()
    
    logger.info(f"Selected {len(available_columns)} columns for modeling")
    
    # Data cleaning
    logger.info("Cleaning data...")
    
    # Handle data types
    if 'churn' in df_model.columns:
        # Standardize churn labels
        df_model['churn'] = df_model['churn'].map({'Yes': 'Yes', 'No': 'No', 
                                                   1: 'Yes', 0: 'No'}).fillna('No')
    
    # Handle Yes/No columns
    yes_no_columns = ['senior_citizen', 'partner', 'dependents', 'phone_service', 
                     'multiple_lines', 'online_security', 'online_backup', 
                     'device_protection', 'tech_support', 'streaming_tv', 
                     'streaming_movies', 'paperless_billing']
    
    for col in yes_no_columns:
        if col in df_model.columns:
            # Convert various formats to Yes/No
            df_model[col] = df_model[col].astype(str)
            df_model[col] = df_model[col].map({
                'True': 'Yes', 'False': 'No', '1': 'Yes', '0': 'No',
                'Yes': 'Yes', 'No': 'No', 'yes': 'Yes', 'no': 'No'
            }).fillna('No')
    
    # Handle numerical columns
    numerical_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
    for col in numerical_cols:
        if col in df_model.columns:
            df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
    
    # Remove rows with missing target variable
    if 'churn' in df_model.columns:
        initial_rows = len(df_model)
        df_model = df_model.dropna(subset=['churn'])
        logger.info(f"Removed {initial_rows - len(df_model)} rows with missing churn labels")
    
    # Fill remaining missing values
    for col in df_model.columns:
        if col == 'customer_id':
            continue
            
        if df_model[col].dtype == 'object':
            # Categorical columns - fill with mode or 'Unknown'
            mode_val = df_model[col].mode().iloc[0] if not df_model[col].mode().empty else 'Unknown'
            df_model[col].fillna(mode_val, inplace=True)
        else:
            # Numerical columns - fill with median
            median_val = df_model[col].median()
            df_model[col].fillna(median_val, inplace=True)
    
    logger.info(f"Final cleaned data shape: {df_model.shape}")
    
    if 'churn' in df_model.columns:
        churn_rate = (df_model['churn'] == 'Yes').mean()
        logger.info(f"Churn rate: {churn_rate:.2%}")
    
    return df_model


def create_feature_config(df: pd.DataFrame, output_dir: str):
    """Create feature configuration file."""
    # Identify feature types
    categorical_features = []
    numerical_features = []
    
    for col in df.columns:
        if col in ['customer_id', 'churn']:  # Skip ID and target
            continue
            
        if df[col].dtype == 'object':
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
        logger.info("Creating new config file")
        config = {
            'project': {
                'name': 'customer-churn-prediction',
                'version': '1.0.0'
            },
            'database': {
                'host': 'localhost',
                'port': 5432,
                'name': 'mlops_db',
                'user': 'mlops_user'
            },
            'monitoring': {
                'drift_detection': {
                    'enabled': True,
                    'thresholds': {
                        'ks': 0.05,
                        'psi': 0.1,
                        'wasserstein': 0.2
                    }
                }
            }
        }
    
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
    
    # Add model configuration
    if 'models' not in config['training']:
        config['training']['models'] = {
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        }
    
    # Add performance thresholds
    if 'thresholds' not in config['training']:
        config['training']['thresholds'] = {
            'min_accuracy': 0.75,
            'min_precision': 0.70,
            'min_recall': 0.65,
            'min_f1_score': 0.70,
            'min_auc_roc': 0.75
        }
    
    # Save updated config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Main configuration updated: {config_path}")


def main():
    """Main function to process the Telco data."""
    parser = argparse.ArgumentParser(description="Process Telco Customer dataset")
    parser.add_argument("--input-path", type=str, default="data/raw/telco.csv",
                       help="Input CSV file path")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                       help="Output directory for processed data")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                       help="Ratio for training data split")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_path).exists():
        logger.error(f"Input file not found: {args.input_path}")
        return
    
    # Process data
    logger.info("=== Processing Telco Dataset ===")
    df = load_and_clean_telco_data(args.input_path)
    
    # Create train/validation split
    logger.info("=== Creating Train/Validation Split ===")
    df_train, df_val = train_test_split(
        df, 
        test_size=(1 - args.train_ratio),
        random_state=42,
        stratify=df['churn'] if 'churn' in df.columns else None
    )
    
    # Save processed data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "customer_data.csv"
    val_path = output_dir / "validation_data.csv"
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    
    logger.info(f"Training data saved: {train_path} ({len(df_train)} samples)")
    logger.info(f"Validation data saved: {val_path} ({len(df_val)} samples)")
    
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
    
    if 'churn' in df.columns:
        logger.info(f"Churn rate (train): {(df_train['churn'] == 'Yes').mean():.2%}")
        logger.info(f"Churn rate (validation): {(df_val['churn'] == 'Yes').mean():.2%}")
    
    logger.info("\n✅ Data processing completed!")
    logger.info("Next steps:")
    logger.info("1. Train model: python src/models/train.py --data-path data/raw/customer_data.csv")
    logger.info("2. Start API: python scripts/start_api.py")
    logger.info("3. Run validation: python scripts/validate_model.py")


if __name__ == "__main__":
    main()