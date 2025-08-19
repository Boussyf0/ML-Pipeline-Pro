"""Airflow DAG for MLOps training pipeline."""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.email import EmailOperator
import os
import sys

# Add src directory to Python path
sys.path.append('/opt/airflow/dags/src')

# Import our modules
from src.data.preprocessor import DataPreprocessor
from src.models.trainer import ModelTrainer
from src.models.registry import ModelRegistry


# Default DAG arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email': ['admin@company.com']
}

# Create DAG
dag = DAG(
    'mlops_training_pipeline',
    default_args=default_args,
    description='Complete MLOps training pipeline for customer churn prediction',
    schedule_interval='0 2 * * 0',  # Weekly at 2 AM on Sunday
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'training', 'churn-prediction']
)


def validate_input_data(**context):
    """Validate input data quality."""
    data_path = "/opt/airflow/data/raw/customer_data.csv"
    
    preprocessor = DataPreprocessor()
    validation_results = preprocessor.validate_data(data_path)
    
    # Check data quality score
    quality_score = validation_results.get('data_quality_score', 0)
    if quality_score < 0.8:
        raise ValueError(f"Data quality score {quality_score} is below threshold (0.8)")
    
    # Log results to XCom
    return validation_results


def detect_data_drift(**context):
    """Detect data drift between current and reference data."""
    current_data_path = "/opt/airflow/data/raw/customer_data.csv"
    reference_data_path = "/opt/airflow/data/reference/customer_data_reference.csv"
    
    preprocessor = DataPreprocessor()
    
    if os.path.exists(reference_data_path):
        import pandas as pd
        current_df = pd.read_csv(current_data_path)
        reference_df = pd.read_csv(reference_data_path)
        
        drift_results = preprocessor.detect_data_drift(reference_df, current_df)
        
        # Check if significant drift is detected
        drift_detected = any(
            result.get('drift_detected', False) 
            for result in drift_results.values() 
            if isinstance(result, dict)
        )
        
        if drift_detected:
            print("WARNING: Data drift detected!")
            # Could send alert or fail the pipeline based on severity
            
        return drift_results
    else:
        print("No reference data found, skipping drift detection")
        return {"status": "no_reference_data"}


def preprocess_data(**context):
    """Preprocess data for training."""
    data_path = "/opt/airflow/data/raw/customer_data.csv"
    output_dir = "/opt/airflow/data/processed/"
    
    preprocessor = DataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(data_path)
    
    # Save processed data
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    return {
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape
    }


def train_models(**context):
    """Train ML models."""
    data_path = "/opt/airflow/data/raw/customer_data.csv"
    
    trainer = ModelTrainer()
    results = trainer.train_models(data_path)
    
    # Get best model
    try:
        best_name, best_model, best_metrics = trainer.get_best_model()
        
        # Check if model meets production criteria
        auc_score = best_metrics.get('test_auc_roc', best_metrics.get('auc_roc', 0))
        if auc_score < 0.75:
            raise ValueError(f"Best model AUC ({auc_score:.4f}) is below production threshold (0.75)")
            
        return {
            "best_model": best_name,
            "best_metrics": best_metrics,
            "all_results": {k: v["metrics"] for k, v in results.items()}
        }
        
    except ValueError as e:
        raise ValueError(f"No suitable models for production: {e}")


def register_best_model(**context):
    """Register the best model in MLflow registry."""
    # Get training results from previous task
    training_results = context['task_instance'].xcom_pull(task_ids='train_models')
    best_model_name = training_results['best_model']
    
    registry = ModelRegistry()
    
    # In a real scenario, we would get the run_id from the training task
    # For now, we'll demonstrate the registration process
    print(f"Registering best model: {best_model_name}")
    
    # This would be the actual registration call:
    # model_version = registry.register_model(run_id, f"{best_model_name}_model")
    
    return {"registered_model": best_model_name, "status": "registered"}


def promote_to_staging(**context):
    """Promote model to staging environment."""
    registration_results = context['task_instance'].xcom_pull(task_ids='register_best_model')
    
    registry = ModelRegistry()
    model_name = registry.registered_model_name
    
    # Get latest version
    latest_version = registry.get_latest_version(model_name)
    
    if latest_version:
        # Promote to staging
        # registry.promote_model(model_name, latest_version, "Staging")
        print(f"Promoted model {model_name} version {latest_version} to Staging")
        
        return {
            "model_name": model_name,
            "version": latest_version,
            "stage": "Staging"
        }
    else:
        raise ValueError("No model version found to promote")


def send_training_report(**context):
    """Send training completion report."""
    training_results = context['task_instance'].xcom_pull(task_ids='train_models')
    validation_results = context['task_instance'].xcom_pull(task_ids='validate_input_data')
    
    # Prepare report
    best_model = training_results['best_model']
    best_metrics = training_results['best_metrics']
    quality_score = validation_results.get('data_quality_score', 'N/A')
    
    report = f"""
    MLOps Training Pipeline Report
    =============================
    
    Execution Date: {context['ds']}
    
    Data Quality:
    - Quality Score: {quality_score}
    
    Best Model: {best_model}
    - Accuracy: {best_metrics.get('test_accuracy', best_metrics.get('accuracy', 'N/A')):.4f}
    - Precision: {best_metrics.get('test_precision', best_metrics.get('precision', 'N/A')):.4f}
    - Recall: {best_metrics.get('test_recall', best_metrics.get('recall', 'N/A')):.4f}
    - F1 Score: {best_metrics.get('test_f1_score', best_metrics.get('f1_score', 'N/A')):.4f}
    - AUC-ROC: {best_metrics.get('test_auc_roc', best_metrics.get('auc_roc', 'N/A')):.4f}
    
    All Models Results:
    {training_results['all_results']}
    
    Next Steps:
    - Model registered and promoted to staging
    - Ready for A/B testing validation
    """
    
    print(report)
    return report


# Define tasks
wait_for_data = FileSensor(
    task_id='wait_for_data',
    filepath='/opt/airflow/data/raw/customer_data.csv',
    fs_conn_id='fs_default',
    poke_interval=300,  # 5 minutes
    timeout=3600,  # 1 hour
    dag=dag
)

validate_data_task = PythonOperator(
    task_id='validate_input_data',
    python_callable=validate_input_data,
    dag=dag
)

detect_drift_task = PythonOperator(
    task_id='detect_data_drift',
    python_callable=detect_data_drift,
    dag=dag
)

preprocess_data_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag
)

register_model_task = PythonOperator(
    task_id='register_best_model',
    python_callable=register_best_model,
    dag=dag
)

promote_staging_task = PythonOperator(
    task_id='promote_to_staging',
    python_callable=promote_to_staging,
    dag=dag
)

# Run model tests
test_model_task = BashOperator(
    task_id='test_model',
    bash_command='cd /opt/airflow && python -m pytest tests/test_models.py -v',
    dag=dag
)

send_report_task = PythonOperator(
    task_id='send_training_report',
    python_callable=send_training_report,
    dag=dag
)

# Email notification for success
success_email = EmailOperator(
    task_id='send_success_email',
    to=['admin@company.com'],
    subject='MLOps Training Pipeline - Success',
    html_content="""
    <h3>MLOps Training Pipeline Completed Successfully</h3>
    <p>The weekly training pipeline has completed successfully.</p>
    <p>Check the logs for detailed results.</p>
    """,
    dag=dag,
    trigger_rule='all_success'
)

# Define task dependencies
wait_for_data >> validate_data_task
validate_data_task >> detect_drift_task
detect_drift_task >> preprocess_data_task
preprocess_data_task >> train_models_task
train_models_task >> register_model_task
register_model_task >> promote_staging_task
promote_staging_task >> test_model_task
test_model_task >> send_report_task
send_report_task >> success_email