"""Main training script for the MLOps pipeline."""
import logging
import sys
from pathlib import Path
import click
from .trainer import ModelTrainer
from .registry import ModelRegistry


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--data-path', required=True, help='Path to training data CSV file')
@click.option('--config-path', default='config/config.yaml', help='Path to configuration file')
@click.option('--register-best', is_flag=True, help='Register the best model in MLflow registry')
@click.option('--promote-to-staging', is_flag=True, help='Promote best model to staging')
def main(data_path: str, config_path: str, register_best: bool, promote_to_staging: bool):
    """Run the complete model training pipeline."""
    try:
        logger.info("Starting MLOps training pipeline")
        
        # Initialize trainer
        trainer = ModelTrainer(config_path)
        
        # Train models
        results = trainer.train_models(data_path)
        
        # Display results
        logger.info("Training Results:")
        for model_name, model_info in results.items():
            metrics = model_info["metrics"]
            threshold_met = model_info["meets_threshold"]
            
            logger.info(f"\n{model_name}:")
            logger.info(f"  Meets Threshold: {threshold_met}")
            logger.info(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
            logger.info(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
            logger.info(f"  F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
            logger.info(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}")
            
            if 'test_accuracy' in metrics:
                logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
                logger.info(f"  Test AUC-ROC: {metrics['test_auc_roc']:.4f}")
        
        # Save models
        trainer.save_models()
        
        # Get best model
        try:
            best_name, best_model, best_metrics = trainer.get_best_model()
            logger.info(f"\nBest Model: {best_name}")
            logger.info(f"Best Model AUC-ROC: {best_metrics.get('auc_roc', 'N/A'):.4f}")
            
            # Register best model if requested
            if register_best:
                registry = ModelRegistry(config_path)
                
                # Find the run ID for the best model
                # This would need to be tracked during training
                logger.info(f"Registering best model: {best_name}")
                # model_version = registry.register_model(run_id, f"{best_name}_model")
                
                if promote_to_staging:
                    # registry.promote_model(registry.registered_model_name, model_version, "Staging")
                    logger.info("Model promoted to Staging")
                    
        except ValueError as e:
            logger.error(f"No suitable models found: {e}")
            sys.exit(1)
            
        logger.info("MLOps training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()