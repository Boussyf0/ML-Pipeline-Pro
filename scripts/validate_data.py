#!/usr/bin/env python3
"""Data validation script for CI/CD pipeline."""
import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_data_file(data_path: str, config_path: str = "config/config.yaml") -> dict:
    """Validate data file and return results."""
    try:
        logger.info(f"Validating data file: {data_path}")
        
        # Check if file exists
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with shape: {df.shape}")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(config_path)
        
        # Run validation
        validation_results = preprocessor.validate_data(df)
        
        # Check data quality score
        quality_score = validation_results.get('data_quality_score', 0)
        logger.info(f"Data quality score: {quality_score:.3f}")
        
        # Log detailed results
        logger.info("Validation Results:")
        for check, result in validation_results.items():
            if isinstance(result, bool):
                status = "✓ PASS" if result else "✗ FAIL"
                logger.info(f"  {check}: {status}")
            elif isinstance(result, dict) and check == "target_distribution":
                logger.info(f"  Target distribution: {result}")
                
        return {
            "file_path": data_path,
            "data_shape": df.shape,
            "quality_score": quality_score,
            "validation_results": validation_results,
            "status": "passed" if quality_score >= 0.8 else "failed"
        }
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return {
            "file_path": data_path,
            "error": str(e),
            "status": "error"
        }


def check_data_schema(df: pd.DataFrame, expected_schema: dict) -> dict:
    """Check if data matches expected schema."""
    schema_check = {
        "columns_match": True,
        "types_match": True,
        "missing_columns": [],
        "extra_columns": [],
        "type_mismatches": []
    }
    
    try:
        # Check columns
        expected_columns = set(expected_schema.keys())
        actual_columns = set(df.columns)
        
        schema_check["missing_columns"] = list(expected_columns - actual_columns)
        schema_check["extra_columns"] = list(actual_columns - expected_columns)
        schema_check["columns_match"] = len(schema_check["missing_columns"]) == 0
        
        # Check data types for common columns
        common_columns = expected_columns & actual_columns
        for col in common_columns:
            expected_type = expected_schema[col]
            actual_type = str(df[col].dtype)
            
            # Simple type matching (could be more sophisticated)
            type_matches = (
                (expected_type in ["int", "integer"] and "int" in actual_type) or
                (expected_type in ["float", "numeric"] and "float" in actual_type) or
                (expected_type in ["str", "string", "object"] and "object" in actual_type) or
                (expected_type in ["bool", "boolean"] and "bool" in actual_type)
            )
            
            if not type_matches:
                schema_check["type_mismatches"].append({
                    "column": col,
                    "expected": expected_type,
                    "actual": actual_type
                })
                
        schema_check["types_match"] = len(schema_check["type_mismatches"]) == 0
        
    except Exception as e:
        logger.error(f"Schema check failed: {e}")
        schema_check["error"] = str(e)
        
    return schema_check


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate data for ML pipeline")
    parser.add_argument("--data-path", required=True, help="Path to data file")
    parser.add_argument("--config-path", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--schema-file", help="Path to expected schema JSON file")
    parser.add_argument("--output-file", help="Path to save validation results")
    parser.add_argument("--min-quality-score", type=float, default=0.8, 
                       help="Minimum data quality score")
    parser.add_argument("--fail-on-warnings", action="store_true", 
                       help="Fail pipeline on validation warnings")
    
    args = parser.parse_args()
    
    try:
        # Run validation
        results = validate_data_file(args.data_path, args.config_path)
        
        # Schema validation if provided
        if args.schema_file:
            logger.info(f"Checking schema against: {args.schema_file}")
            
            with open(args.schema_file, 'r') as f:
                expected_schema = json.load(f)
                
            df = pd.read_csv(args.data_path)
            schema_results = check_data_schema(df, expected_schema)
            results["schema_check"] = schema_results
            
            # Log schema results
            if schema_results.get("columns_match") and schema_results.get("types_match"):
                logger.info("✓ Schema validation passed")
            else:
                logger.warning("⚠ Schema validation issues found")
                if schema_results.get("missing_columns"):
                    logger.warning(f"  Missing columns: {schema_results['missing_columns']}")
                if schema_results.get("extra_columns"):
                    logger.warning(f"  Extra columns: {schema_results['extra_columns']}")
                if schema_results.get("type_mismatches"):
                    logger.warning(f"  Type mismatches: {schema_results['type_mismatches']}")
        
        # Save results if output file specified
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Validation results saved to: {args.output_file}")
            
        # Determine if validation passed
        quality_score = results.get("quality_score", 0)
        schema_passed = True
        
        if "schema_check" in results:
            schema_check = results["schema_check"]
            schema_passed = schema_check.get("columns_match", True) and schema_check.get("types_match", True)
            
        # Check pass/fail conditions
        validation_passed = (
            results.get("status") != "error" and
            quality_score >= args.min_quality_score and
            (schema_passed or not args.fail_on_warnings)
        )
        
        # Final result
        if validation_passed:
            logger.info("✅ Data validation PASSED")
            logger.info(f"Quality Score: {quality_score:.3f} (threshold: {args.min_quality_score})")
            sys.exit(0)
        else:
            logger.error("❌ Data validation FAILED")
            logger.error(f"Quality Score: {quality_score:.3f} (threshold: {args.min_quality_score})")
            if not schema_passed:
                logger.error("Schema validation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Validation script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()