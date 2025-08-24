#!/usr/bin/env python3
"""
Data Drift Detection Script for MLOps Pipeline
Compares reference data with current data to detect statistical drift.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_suite import MetricSuite
from evidently.metrics import DataDriftTable, DatasetDriftMetric
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detect statistical drift between reference and current datasets."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.drift_results = {}
        
    def load_data(self, reference_path: str, current_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load reference and current datasets."""
        try:
            if os.path.isdir(reference_path):
                # Load multiple reference files
                ref_files = list(Path(reference_path).glob('*.csv'))
                if not ref_files:
                    raise FileNotFoundError(f"No CSV files found in {reference_path}")
                reference_df = pd.concat([pd.read_csv(f) for f in ref_files], ignore_index=True)
            else:
                reference_df = pd.read_csv(reference_path)
                
            if os.path.isdir(current_path):
                # Load multiple current files
                curr_files = list(Path(current_path).glob('*.csv'))
                if not curr_files:
                    raise FileNotFoundError(f"No CSV files found in {current_path}")
                current_df = pd.concat([pd.read_csv(f) for f in curr_files], ignore_index=True)
            else:
                current_df = pd.read_csv(current_path)
                
            logger.info(f"Loaded reference data: {reference_df.shape}")
            logger.info(f"Loaded current data: {current_df.shape}")
            
            return reference_df, current_df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
            
    def detect_numerical_drift(self, ref_data: pd.Series, curr_data: pd.Series, 
                             column_name: str) -> Dict[str, Any]:
        """Detect drift in numerical columns using KS test."""
        try:
            # Remove NaN values
            ref_clean = ref_data.dropna()
            curr_clean = curr_data.dropna()
            
            if len(ref_clean) == 0 or len(curr_clean) == 0:
                return {
                    'column': column_name,
                    'drift_detected': False,
                    'p_value': None,
                    'statistic': None,
                    'method': 'ks_test',
                    'error': 'Empty data after removing NaN values'
                }
            
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(ref_clean, curr_clean)
            
            drift_detected = p_value < self.significance_level
            
            return {
                'column': column_name,
                'drift_detected': drift_detected,
                'p_value': p_value,
                'statistic': statistic,
                'method': 'ks_test',
                'ref_mean': ref_clean.mean(),
                'curr_mean': curr_clean.mean(),
                'ref_std': ref_clean.std(),
                'curr_std': curr_clean.std()
            }
            
        except Exception as e:
            logger.warning(f"Error in numerical drift detection for {column_name}: {e}")
            return {
                'column': column_name,
                'drift_detected': False,
                'p_value': None,
                'statistic': None,
                'method': 'ks_test',
                'error': str(e)
            }
            
    def detect_categorical_drift(self, ref_data: pd.Series, curr_data: pd.Series,
                               column_name: str) -> Dict[str, Any]:
        """Detect drift in categorical columns using Chi-square test."""
        try:
            # Get value counts
            ref_counts = ref_data.value_counts()
            curr_counts = curr_data.value_counts()
            
            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]
            
            # Chi-square test
            if sum(ref_aligned) == 0 or sum(curr_aligned) == 0:
                return {
                    'column': column_name,
                    'drift_detected': False,
                    'p_value': None,
                    'statistic': None,
                    'method': 'chi2_test',
                    'error': 'Empty data'
                }
            
            statistic, p_value = stats.chisquare(curr_aligned, ref_aligned)
            
            drift_detected = p_value < self.significance_level
            
            return {
                'column': column_name,
                'drift_detected': drift_detected,
                'p_value': p_value,
                'statistic': statistic,
                'method': 'chi2_test',
                'ref_categories': len(ref_counts),
                'curr_categories': len(curr_counts)
            }
            
        except Exception as e:
            logger.warning(f"Error in categorical drift detection for {column_name}: {e}")
            return {
                'column': column_name,
                'drift_detected': False,
                'p_value': None,
                'statistic': None,
                'method': 'chi2_test',
                'error': str(e)
            }
            
    def detect_drift_evidently(self, reference_df: pd.DataFrame, 
                             current_df: pd.DataFrame) -> Dict[str, Any]:
        """Use Evidently AI for comprehensive drift detection."""
        try:
            # Align columns
            common_columns = list(set(reference_df.columns) & set(current_df.columns))
            ref_aligned = reference_df[common_columns].copy()
            curr_aligned = current_df[common_columns].copy()
            
            # Create column mapping
            numerical_features = ref_aligned.select_dtypes(include=[np.number]).columns.tolist()
            categorical_features = ref_aligned.select_dtypes(include=['object', 'category']).columns.tolist()
            
            column_mapping = ColumnMapping(
                numerical_features=numerical_features,
                categorical_features=categorical_features
            )
            
            # Create and run report
            data_drift_report = Report(metrics=[
                DatasetDriftMetric(),
                DataDriftTable()
            ])
            
            data_drift_report.run(
                reference_data=ref_aligned,
                current_data=curr_aligned,
                column_mapping=column_mapping
            )
            
            # Extract results
            report_dict = data_drift_report.as_dict()
            
            return {
                'overall_drift': report_dict['metrics'][0]['result']['dataset_drift'],
                'drift_by_columns': report_dict['metrics'][1]['result']['drift_by_columns'],
                'number_of_columns': report_dict['metrics'][0]['result']['number_of_columns'],
                'number_of_drifted_columns': report_dict['metrics'][0]['result']['number_of_drifted_columns'],
                'share_of_drifted_columns': report_dict['metrics'][0]['result']['share_of_drifted_columns']
            }
            
        except Exception as e:
            logger.error(f"Error in Evidently drift detection: {e}")
            return {'error': str(e)}
            
    def run_drift_detection(self, reference_path: str, current_path: str) -> Dict[str, Any]:
        """Run complete drift detection analysis."""
        logger.info("Starting data drift detection...")
        
        # Load data
        reference_df, current_df = self.load_data(reference_path, current_path)
        
        # Get common columns
        common_columns = list(set(reference_df.columns) & set(current_df.columns))
        logger.info(f"Analyzing {len(common_columns)} common columns")
        
        results = {
            'overall_summary': {
                'total_columns': len(common_columns),
                'drifted_columns': 0,
                'drift_percentage': 0.0
            },
            'column_results': {},
            'evidently_results': {}
        }
        
        # Analyze each column
        for column in common_columns:
            ref_series = reference_df[column]
            curr_series = current_df[column]
            
            # Determine if numerical or categorical
            if pd.api.types.is_numeric_dtype(ref_series):
                drift_result = self.detect_numerical_drift(ref_series, curr_series, column)
            else:
                drift_result = self.detect_categorical_drift(ref_series, curr_series, column)
                
            results['column_results'][column] = drift_result
            
            # Count drifted columns
            if drift_result.get('drift_detected', False):
                results['overall_summary']['drifted_columns'] += 1
                
        # Calculate drift percentage
        if len(common_columns) > 0:
            results['overall_summary']['drift_percentage'] = (
                results['overall_summary']['drifted_columns'] / len(common_columns) * 100
            )
            
        # Run Evidently analysis
        results['evidently_results'] = self.detect_drift_evidently(reference_df, current_df)
        
        return results
        
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save drift detection results to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            
    def print_summary(self, results: Dict[str, Any]):
        """Print drift detection summary."""
        summary = results['overall_summary']
        
        print("\n" + "="*60)
        print("DATA DRIFT DETECTION SUMMARY")
        print("="*60)
        print(f"Total columns analyzed: {summary['total_columns']}")
        print(f"Columns with drift detected: {summary['drifted_columns']}")
        print(f"Drift percentage: {summary['drift_percentage']:.2f}%")
        
        if summary['drifted_columns'] > 0:
            print(f"\nColumns with detected drift:")
            for column, result in results['column_results'].items():
                if result.get('drift_detected', False):
                    p_val = result.get('p_value', 'N/A')
                    method = result.get('method', 'Unknown')
                    print(f"  - {column} (p-value: {p_val:.6f}, method: {method})")
                    
        # Evidently summary
        if 'evidently_results' in results and 'overall_drift' in results['evidently_results']:
            ev_results = results['evidently_results']
            print(f"\nEvidently AI Analysis:")
            print(f"  Overall drift detected: {ev_results.get('overall_drift', 'N/A')}")
            print(f"  Drifted columns: {ev_results.get('number_of_drifted_columns', 'N/A')}")
            print(f"  Drift share: {ev_results.get('share_of_drifted_columns', 'N/A'):.2%}")


def main():
    parser = argparse.ArgumentParser(description='Detect data drift between datasets')
    parser.add_argument('--reference', required=True, 
                       help='Path to reference dataset or directory')
    parser.add_argument('--current', required=True,
                       help='Path to current dataset or directory')
    parser.add_argument('--output', default='data_drift_report.json',
                       help='Output path for drift report')
    parser.add_argument('--significance-level', type=float, default=0.05,
                       help='Significance level for drift detection')
    parser.add_argument('--fail-on-drift', action='store_true',
                       help='Exit with error code if drift is detected')
    
    args = parser.parse_args()
    
    try:
        detector = DataDriftDetector(significance_level=args.significance_level)
        results = detector.run_drift_detection(args.reference, args.current)
        
        # Save results
        detector.save_results(results, args.output)
        
        # Print summary
        detector.print_summary(results)
        
        # Check if we should fail on drift
        if args.fail_on_drift and results['overall_summary']['drifted_columns'] > 0:
            logger.error("Data drift detected! Exiting with error code.")
            sys.exit(1)
            
        logger.info("Data drift detection completed successfully.")
        
    except Exception as e:
        logger.error(f"Data drift detection failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()