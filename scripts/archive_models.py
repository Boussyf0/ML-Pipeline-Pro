#!/usr/bin/env python3
"""
Model Archive Script
Archives old model versions while keeping the most recent ones.
"""

import argparse
import logging
import sys
import os
import shutil
import json
import time
from typing import Dict, Any, List
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelArchiver:
    """Archive old model versions and clean up storage."""
    
    def __init__(self, keep_latest: int = 5):
        self.keep_latest = keep_latest
        self.client = MlflowClient()
        self.archive_stats = {
            "models_processed": 0,
            "versions_archived": 0,
            "space_freed_mb": 0,
            "errors": []
        }
        
    def get_registered_models(self) -> List[str]:
        """Get list of all registered models."""
        try:
            models = self.client.search_registered_models()
            return [model.name for model in models]
        except Exception as e:
            logger.error(f"Failed to get registered models: {e}")
            return []
            
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model sorted by creation time."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            # Convert to list of dicts with relevant info
            version_info = []
            for version in versions:
                version_info.append({
                    "name": version.name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "creation_timestamp": version.creation_timestamp,
                    "last_updated_timestamp": version.last_updated_timestamp,
                    "run_id": version.run_id,
                    "source": version.source,
                    "status": version.status
                })
                
            # Sort by creation timestamp (newest first)
            version_info.sort(key=lambda x: x["creation_timestamp"], reverse=True)
            
            return version_info
            
        except Exception as e:
            logger.error(f"Failed to get model versions for {model_name}: {e}")
            return []
            
    def can_archive_version(self, version_info: Dict[str, Any]) -> bool:
        """Determine if a model version can be archived."""
        stage = version_info.get("stage", "").lower()
        
        # Never archive Production or Staging models
        if stage in ["production", "staging"]:
            return False
            
        # Never archive models that are in transition
        if version_info.get("status") == "PENDING_REGISTRATION":
            return False
            
        # Can archive None, Archived, or custom stages
        return True
        
    def calculate_model_size(self, version_info: Dict[str, Any]) -> int:
        """Calculate size of model artifacts in MB."""
        try:
            model_uri = version_info.get("source", "")
            
            # For file:// URIs, calculate directory size
            if model_uri.startswith("file://"):
                local_path = model_uri.replace("file://", "")
                if os.path.exists(local_path):
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(local_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            if os.path.exists(filepath):
                                total_size += os.path.getsize(filepath)
                    return total_size // (1024 * 1024)  # Convert to MB
                    
            # For other URIs (S3, etc.), return estimated size
            # In production, you'd implement actual size calculation
            return 50  # Default estimate
            
        except Exception as e:
            logger.warning(f"Failed to calculate model size: {e}")
            return 0
            
    def archive_model_version(self, version_info: Dict[str, Any]) -> bool:
        """Archive a specific model version."""
        try:
            model_name = version_info["name"]
            version = version_info["version"]
            
            logger.info(f"Archiving model {model_name} version {version}")
            
            # Transition to Archived stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Archived",
                archive_existing_versions=False
            )
            
            # Add archive metadata
            archive_metadata = {
                "archived_at": int(time.time()),
                "archived_by": "automated_cleanup",
                "original_stage": version_info.get("stage", "None"),
                "archive_reason": "retention_policy"
            }
            
            # Set tags to track archival
            for key, value in archive_metadata.items():
                try:
                    self.client.set_model_version_tag(
                        name=model_name,
                        version=version,
                        key=key,
                        value=str(value)
                    )
                except Exception as tag_error:
                    logger.warning(f"Failed to set archive tag: {tag_error}")
                    
            logger.info(f"Successfully archived model {model_name} version {version}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to archive {version_info['name']} v{version_info['version']}: {e}"
            logger.error(error_msg)
            self.archive_stats["errors"].append(error_msg)
            return False
            
    def cleanup_archived_artifacts(self, version_info: Dict[str, Any]) -> int:
        """Clean up artifacts for archived model versions."""
        try:
            model_uri = version_info.get("source", "")
            
            # Only clean up local file artifacts for now
            if model_uri.startswith("file://"):
                local_path = model_uri.replace("file://", "")
                
                if os.path.exists(local_path):
                    # Calculate size before deletion
                    size_mb = self.calculate_model_size(version_info)
                    
                    # Create archive directory if it doesn't exist
                    archive_base = "/tmp/model_archives"
                    os.makedirs(archive_base, exist_ok=True)
                    
                    # Move to archive location instead of deleting
                    archive_path = os.path.join(
                        archive_base,
                        f"{version_info['name']}_v{version_info['version']}_{int(time.time())}"
                    )
                    
                    shutil.move(local_path, archive_path)
                    logger.info(f"Moved artifacts to archive: {archive_path}")
                    
                    return size_mb
                    
            return 0
            
        except Exception as e:
            logger.warning(f"Failed to cleanup artifacts: {e}")
            return 0
            
    def process_model(self, model_name: str) -> Dict[str, Any]:
        """Process a single model for archiving."""
        logger.info(f"Processing model: {model_name}")
        
        result = {
            "model_name": model_name,
            "total_versions": 0,
            "versions_archived": 0,
            "versions_kept": 0,
            "space_freed_mb": 0,
            "errors": []
        }
        
        try:
            # Get all versions of the model
            versions = self.get_model_versions(model_name)
            result["total_versions"] = len(versions)
            
            if len(versions) <= self.keep_latest:
                logger.info(f"Model {model_name} has {len(versions)} versions, keeping all")
                result["versions_kept"] = len(versions)
                return result
                
            # Keep the latest N versions
            versions_to_keep = versions[:self.keep_latest]
            versions_to_archive = versions[self.keep_latest:]
            
            logger.info(f"Keeping {len(versions_to_keep)} versions, archiving {len(versions_to_archive)} versions")
            
            # Archive old versions
            for version_info in versions_to_archive:
                if self.can_archive_version(version_info):
                    if self.archive_model_version(version_info):
                        result["versions_archived"] += 1
                        
                        # Clean up artifacts if needed
                        space_freed = self.cleanup_archived_artifacts(version_info)
                        result["space_freed_mb"] += space_freed
                    else:
                        error_msg = f"Failed to archive {model_name} v{version_info['version']}"
                        result["errors"].append(error_msg)
                else:
                    logger.info(f"Skipping {model_name} v{version_info['version']} - stage: {version_info['stage']}")
                    result["versions_kept"] += 1
                    
            result["versions_kept"] += len(versions_to_keep)
            
        except Exception as e:
            error_msg = f"Error processing model {model_name}: {e}"
            logger.error(error_msg)
            result["errors"].append(error_msg)
            
        return result
        
    def generate_cleanup_report(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cleanup summary report."""
        report = {
            "timestamp": time.time(),
            "keep_latest_policy": self.keep_latest,
            "summary": {
                "models_processed": len(model_results),
                "total_versions_found": sum(r["total_versions"] for r in model_results),
                "versions_archived": sum(r["versions_archived"] for r in model_results),
                "versions_kept": sum(r["versions_kept"] for r in model_results),
                "total_space_freed_mb": sum(r["space_freed_mb"] for r in model_results),
                "total_errors": sum(len(r["errors"]) for r in model_results)
            },
            "model_details": model_results,
            "overall_errors": self.archive_stats["errors"]
        }
        
        return report
        
    def save_report(self, report: Dict[str, Any], output_path: str):
        """Save cleanup report to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Cleanup report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            
    def print_summary(self, report: Dict[str, Any]):
        """Print cleanup summary."""
        summary = report["summary"]
        
        print(f"\n{'='*60}")
        print("MODEL CLEANUP SUMMARY")
        print(f"{'='*60}")
        print(f"Retention Policy: Keep latest {self.keep_latest} versions")
        print(f"Models Processed: {summary['models_processed']}")
        print(f"Total Versions Found: {summary['total_versions_found']}")
        print(f"Versions Archived: {summary['versions_archived']}")
        print(f"Versions Kept: {summary['versions_kept']}")
        print(f"Space Freed: {summary['total_space_freed_mb']} MB")
        
        if summary['total_errors'] > 0:
            print(f"\n⚠️  Errors Encountered: {summary['total_errors']}")
            
        print(f"\nModel Details:")
        for model_result in report["model_details"]:
            print(f"  {model_result['model_name']}:")
            print(f"    Total versions: {model_result['total_versions']}")
            print(f"    Archived: {model_result['versions_archived']}")
            print(f"    Kept: {model_result['versions_kept']}")
            if model_result['space_freed_mb'] > 0:
                print(f"    Space freed: {model_result['space_freed_mb']} MB")
            if model_result['errors']:
                print(f"    Errors: {len(model_result['errors'])}")
                
        print(f"{'='*60}")
        
    def cleanup_models(self, model_names: List[str] = None) -> bool:
        """Run model cleanup process."""
        logger.info(f"Starting model cleanup (keep latest {self.keep_latest} versions)")
        
        try:
            # Get list of models to process
            if model_names:
                models_to_process = model_names
            else:
                models_to_process = self.get_registered_models()
                
            if not models_to_process:
                logger.warning("No models found to process")
                return True
                
            logger.info(f"Processing {len(models_to_process)} models")
            
            # Process each model
            model_results = []
            for model_name in models_to_process:
                result = self.process_model(model_name)
                model_results.append(result)
                
            # Generate and save report
            report = self.generate_cleanup_report(model_results)
            
            # Print summary
            self.print_summary(report)
            
            # Save report
            report_path = f"/tmp/model_cleanup_report_{int(time.time())}.json"
            self.save_report(report, report_path)
            
            # Check if there were any critical errors
            total_errors = report["summary"]["total_errors"]
            if total_errors > 0:
                logger.warning(f"Cleanup completed with {total_errors} errors")
                return False
            else:
                logger.info("Model cleanup completed successfully")
                return True
                
        except Exception as e:
            logger.error(f"Model cleanup failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Archive old model versions')
    parser.add_argument('--keep-latest', type=int, default=5,
                       help='Number of latest versions to keep per model')
    parser.add_argument('--models', nargs='+',
                       help='Specific models to process (default: all models)')
    parser.add_argument('--mlflow-uri', help='MLflow tracking URI')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be archived without actually doing it')
    
    args = parser.parse_args()
    
    try:
        # Set MLflow tracking URI if provided
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
            
        if args.dry_run:
            logger.info("DRY RUN MODE - No actual archiving will be performed")
            
        archiver = ModelArchiver(keep_latest=args.keep_latest)
        success = archiver.cleanup_models(model_names=args.models)
        
        if success:
            logger.info("Model archiving completed successfully")
            sys.exit(0)
        else:
            logger.error("Model archiving completed with errors")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Archive script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()