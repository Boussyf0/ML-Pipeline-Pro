#!/usr/bin/env python3
"""
Smoke Tests for MLOps Pipeline
Basic health checks and API endpoint validation.
"""

import argparse
import logging
import sys
import time
import requests
import json
from typing import Dict, Any, Optional
from urllib.parse import urljoin

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmokeTestSuite:
    """Smoke test suite for API health checks."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.test_results = []
        
    def run_test(self, test_name: str, test_func):
        """Run a single test and record results."""
        logger.info(f"Running test: {test_name}")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            
            self.test_results.append({
                'test_name': test_name,
                'status': 'PASS',
                'duration': duration,
                'message': result.get('message', 'Test passed'),
                'details': result.get('details', {})
            })
            logger.info(f"✅ {test_name} - PASSED ({duration:.2f}s)")
            return True
            
        except Exception as e:
            duration = time.time() - start_time
            self.test_results.append({
                'test_name': test_name,
                'status': 'FAIL',
                'duration': duration,
                'message': str(e),
                'details': {}
            })
            logger.error(f"❌ {test_name} - FAILED ({duration:.2f}s): {e}")
            return False
            
    def test_health_endpoint(self) -> Dict[str, Any]:
        """Test API health endpoint."""
        url = urljoin(self.base_url, '/health')
        response = self.session.get(url, timeout=self.timeout)
        
        if response.status_code != 200:
            raise Exception(f"Health check failed with status {response.status_code}")
            
        data = response.json()
        
        return {
            'message': 'Health endpoint is responsive',
            'details': {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'health_data': data
            }
        }
        
    def test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test Prometheus metrics endpoint."""
        url = urljoin(self.base_url, '/metrics')
        response = self.session.get(url, timeout=self.timeout)
        
        if response.status_code != 200:
            raise Exception(f"Metrics endpoint failed with status {response.status_code}")
            
        metrics_text = response.text
        
        # Check for key metrics
        required_metrics = [
            'http_requests_total',
            'prediction_requests_total',
            'model_prediction_duration_seconds'
        ]
        
        missing_metrics = []
        for metric in required_metrics:
            if metric not in metrics_text:
                missing_metrics.append(metric)
                
        if missing_metrics:
            raise Exception(f"Missing required metrics: {missing_metrics}")
            
        return {
            'message': 'Metrics endpoint is working',
            'details': {
                'status_code': response.status_code,
                'metrics_count': len(metrics_text.split('\n')),
                'has_required_metrics': True
            }
        }
        
    def test_docs_endpoint(self) -> Dict[str, Any]:
        """Test API documentation endpoint."""
        url = urljoin(self.base_url, '/docs')
        response = self.session.get(url, timeout=self.timeout)
        
        if response.status_code != 200:
            raise Exception(f"Docs endpoint failed with status {response.status_code}")
            
        return {
            'message': 'API documentation is accessible',
            'details': {
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': len(response.content)
            }
        }
        
    def test_prediction_endpoint(self) -> Dict[str, Any]:
        """Test model prediction endpoint with sample data."""
        url = urljoin(self.base_url, '/predict')
        
        # Sample customer data for testing
        sample_data = {
            "tenure": 24,
            "monthly_charges": 75.50,
            "total_charges": 1800.0,
            "contract": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "online_backup": "Yes",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes"
        }
        
        response = self.session.post(
            url, 
            json=sample_data,
            headers={'Content-Type': 'application/json'},
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Prediction endpoint failed with status {response.status_code}")
            
        data = response.json()
        
        # Validate response structure
        required_fields = ['prediction', 'probability', 'model_version']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise Exception(f"Missing required fields in response: {missing_fields}")
            
        # Validate prediction values
        prediction = data['prediction']
        probability = data['probability']
        
        if prediction not in [0, 1]:
            raise Exception(f"Invalid prediction value: {prediction}")
            
        if not (0 <= probability <= 1):
            raise Exception(f"Invalid probability value: {probability}")
            
        return {
            'message': 'Prediction endpoint is working correctly',
            'details': {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'prediction': prediction,
                'probability': probability,
                'model_version': data['model_version']
            }
        }
        
    def test_batch_prediction_endpoint(self) -> Dict[str, Any]:
        """Test batch prediction endpoint."""
        url = urljoin(self.base_url, '/predict/batch')
        
        # Sample batch data
        batch_data = {
            "instances": [
                {
                    "tenure": 24,
                    "monthly_charges": 75.50,
                    "total_charges": 1800.0,
                    "contract": "Month-to-month",
                    "payment_method": "Electronic check",
                    "internet_service": "Fiber optic",
                    "online_security": "No",
                    "online_backup": "Yes",
                    "device_protection": "No",
                    "tech_support": "No",
                    "streaming_tv": "Yes",
                    "streaming_movies": "Yes"
                },
                {
                    "tenure": 60,
                    "monthly_charges": 45.00,
                    "total_charges": 2700.0,
                    "contract": "Two year",
                    "payment_method": "Bank transfer",
                    "internet_service": "DSL",
                    "online_security": "Yes",
                    "online_backup": "No",
                    "device_protection": "Yes",
                    "tech_support": "Yes",
                    "streaming_tv": "No",
                    "streaming_movies": "No"
                }
            ]
        }
        
        response = self.session.post(
            url,
            json=batch_data,
            headers={'Content-Type': 'application/json'},
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise Exception(f"Batch prediction endpoint failed with status {response.status_code}")
            
        data = response.json()
        
        # Validate response structure
        if 'predictions' not in data:
            raise Exception("Missing 'predictions' field in batch response")
            
        predictions = data['predictions']
        if len(predictions) != 2:
            raise Exception(f"Expected 2 predictions, got {len(predictions)}")
            
        # Validate each prediction
        for i, pred in enumerate(predictions):
            if 'prediction' not in pred or 'probability' not in pred:
                raise Exception(f"Invalid prediction structure at index {i}")
                
        return {
            'message': 'Batch prediction endpoint is working correctly',
            'details': {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'predictions_count': len(predictions),
                'predictions': predictions
            }
        }
        
    def test_model_info_endpoint(self) -> Dict[str, Any]:
        """Test model information endpoint."""
        url = urljoin(self.base_url, '/model/info')
        response = self.session.get(url, timeout=self.timeout)
        
        if response.status_code != 200:
            raise Exception(f"Model info endpoint failed with status {response.status_code}")
            
        data = response.json()
        
        # Check for required model info fields
        required_fields = ['model_name', 'model_version', 'training_date']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise Exception(f"Missing required model info fields: {missing_fields}")
            
        return {
            'message': 'Model info endpoint is working',
            'details': {
                'status_code': response.status_code,
                'model_info': data
            }
        }
        
    def test_response_times(self) -> Dict[str, Any]:
        """Test API response times."""
        endpoints_to_test = [
            '/health',
            '/predict',
            '/model/info'
        ]
        
        response_times = {}
        slow_endpoints = []
        
        for endpoint in endpoints_to_test:
            url = urljoin(self.base_url, endpoint)
            
            if endpoint == '/predict':
                # Use POST with sample data
                sample_data = {
                    "tenure": 24,
                    "monthly_charges": 75.50,
                    "total_charges": 1800.0,
                    "contract": "Month-to-month",
                    "payment_method": "Electronic check",
                    "internet_service": "Fiber optic",
                    "online_security": "No",
                    "online_backup": "Yes",
                    "device_protection": "No",
                    "tech_support": "No",
                    "streaming_tv": "Yes",
                    "streaming_movies": "Yes"
                }
                response = self.session.post(url, json=sample_data, timeout=self.timeout)
            else:
                response = self.session.get(url, timeout=self.timeout)
                
            response_time = response.elapsed.total_seconds()
            response_times[endpoint] = response_time
            
            # Flag slow endpoints (>2 seconds)
            if response_time > 2.0:
                slow_endpoints.append(endpoint)
                
        if slow_endpoints:
            logger.warning(f"Slow endpoints detected: {slow_endpoints}")
            
        return {
            'message': f'Response time test completed. Average response time: {sum(response_times.values())/len(response_times):.2f}s',
            'details': {
                'response_times': response_times,
                'slow_endpoints': slow_endpoints,
                'average_response_time': sum(response_times.values())/len(response_times)
            }
        }
        
    def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        logger.info(f"Starting smoke tests for {self.base_url}")
        
        tests = [
            ('Health Check', self.test_health_endpoint),
            ('Metrics Endpoint', self.test_metrics_endpoint),
            ('API Documentation', self.test_docs_endpoint),
            ('Single Prediction', self.test_prediction_endpoint),
            ('Batch Prediction', self.test_batch_prediction_endpoint),
            ('Model Information', self.test_model_info_endpoint),
            ('Response Times', self.test_response_times)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed_tests += 1
                
        success_rate = passed_tests / total_tests * 100
        logger.info(f"Smoke tests completed: {passed_tests}/{total_tests} passed ({success_rate:.1f}%)")
        
        return passed_tests == total_tests
        
    def print_summary(self):
        """Print test results summary."""
        print("\n" + "="*60)
        print("SMOKE TESTS SUMMARY")
        print("="*60)
        
        passed = sum(1 for result in self.test_results if result['status'] == 'PASS')
        failed = sum(1 for result in self.test_results if result['status'] == 'FAIL')
        total = len(self.test_results)
        
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        if failed > 0:
            print(f"\nFailed tests:")
            for result in self.test_results:
                if result['status'] == 'FAIL':
                    print(f"  - {result['test_name']}: {result['message']}")
                    
    def save_results(self, output_path: str):
        """Save test results to JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    'base_url': self.base_url,
                    'timestamp': time.time(),
                    'results': self.test_results
                }, f, indent=2)
            logger.info(f"Test results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving test results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run smoke tests for MLOps API')
    parser.add_argument('--url', default='http://localhost:8000',
                       help='Base URL of the API to test')
    parser.add_argument('--environment', default='local',
                       help='Environment being tested (local, staging, production)')
    parser.add_argument('--output', default='smoke_test_results.json',
                       help='Output file for test results')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds')
    
    args = parser.parse_args()
    
    # Map environment to URL if not explicitly provided
    if args.url == 'http://localhost:8000' and args.environment != 'local':
        env_urls = {
            'staging': 'https://staging-api.mlops-pipeline.com',
            'production': 'https://api.mlops-pipeline.com'
        }
        args.url = env_urls.get(args.environment, args.url)
        
    try:
        test_suite = SmokeTestSuite(args.url, timeout=args.timeout)
        success = test_suite.run_all_tests()
        
        test_suite.print_summary()
        test_suite.save_results(args.output)
        
        if not success:
            logger.error("Some smoke tests failed!")
            sys.exit(1)
            
        logger.info("All smoke tests passed!")
        
    except Exception as e:
        logger.error(f"Smoke tests failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()