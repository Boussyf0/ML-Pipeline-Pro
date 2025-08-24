#!/usr/bin/env python3
"""
Integration Tests for MLOps Pipeline API
Tests end-to-end workflows and API integrations.
"""

import pytest
import requests
import json
import time
import os
from typing import Dict, List, Any
from urllib.parse import urljoin


class TestAPIIntegration:
    """Integration tests for the MLOps API."""
    
    @pytest.fixture(scope="class")
    def api_base_url(self):
        """Get API base URL from environment or use default."""
        environment = os.getenv('ENVIRONMENT', 'local')
        
        urls = {
            'local': 'http://localhost:8000',
            'staging': os.getenv('STAGING_API_URL', 'https://staging-api.mlops-pipeline.com'),
            'production': os.getenv('PROD_API_URL', 'https://api.mlops-pipeline.com')
        }
        
        return urls.get(environment, urls['local'])
        
    @pytest.fixture(scope="class")
    def api_client(self, api_base_url):
        """Create API client session."""
        session = requests.Session()
        session.headers.update({'Content-Type': 'application/json'})
        return session
        
    @pytest.fixture
    def sample_customer_data(self):
        """Sample customer data for testing."""
        return {
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
        
    def test_api_health_check(self, api_client, api_base_url):
        """Test API health endpoint."""
        url = urljoin(api_base_url, '/health')
        response = api_client.get(url, timeout=30)
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
        
    def test_single_prediction_workflow(self, api_client, api_base_url, sample_customer_data):
        """Test complete single prediction workflow."""
        # Step 1: Make prediction
        predict_url = urljoin(api_base_url, '/predict')
        response = api_client.post(predict_url, json=sample_customer_data, timeout=30)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure
        assert 'prediction' in data
        assert 'probability' in data
        assert 'model_version' in data
        assert 'prediction_id' in data
        
        # Validate data types and ranges
        assert isinstance(data['prediction'], int)
        assert data['prediction'] in [0, 1]
        assert isinstance(data['probability'], float)
        assert 0 <= data['probability'] <= 1
        
        prediction_id = data['prediction_id']
        
        # Step 2: Check prediction in history (if endpoint exists)
        history_url = urljoin(api_base_url, f'/predictions/{prediction_id}')
        try:
            history_response = api_client.get(history_url, timeout=30)
            if history_response.status_code == 200:
                history_data = history_response.json()
                assert history_data['prediction_id'] == prediction_id
        except requests.exceptions.RequestException:
            # History endpoint may not be implemented
            pass
            
    def test_batch_prediction_workflow(self, api_client, api_base_url, sample_customer_data):
        """Test batch prediction workflow."""
        # Create batch data
        batch_data = {
            "instances": [
                sample_customer_data,
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
        
        batch_url = urljoin(api_base_url, '/predict/batch')
        response = api_client.post(batch_url, json=batch_data, timeout=60)
        
        assert response.status_code == 200
        data = response.json()
        
        assert 'predictions' in data
        predictions = data['predictions']
        assert len(predictions) == 2
        
        # Validate each prediction
        for pred in predictions:
            assert 'prediction' in pred
            assert 'probability' in pred
            assert isinstance(pred['prediction'], int)
            assert pred['prediction'] in [0, 1]
            assert isinstance(pred['probability'], float)
            assert 0 <= pred['probability'] <= 1
            
    def test_model_information_consistency(self, api_client, api_base_url):
        """Test model information endpoint and consistency."""
        model_info_url = urljoin(api_base_url, '/model/info')
        response = api_client.get(model_info_url, timeout=30)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate required fields
        required_fields = ['model_name', 'model_version', 'training_date', 'accuracy', 'features']
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
            
        # Validate data types
        assert isinstance(data['model_name'], str)
        assert isinstance(data['model_version'], str)
        assert isinstance(data['accuracy'], (int, float))
        assert isinstance(data['features'], list)
        
        # Store model version for consistency check
        model_version = data['model_version']
        
        # Make a prediction and verify model version consistency
        predict_url = urljoin(api_base_url, '/predict')
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
        
        pred_response = api_client.post(predict_url, json=sample_data, timeout=30)
        assert pred_response.status_code == 200
        pred_data = pred_response.json()
        
        # Verify model version consistency
        assert pred_data['model_version'] == model_version
        
    def test_error_handling(self, api_client, api_base_url):
        """Test API error handling."""
        predict_url = urljoin(api_base_url, '/predict')
        
        # Test with invalid data
        invalid_data_tests = [
            {},  # Empty data
            {"tenure": "invalid"},  # Invalid data type
            {"tenure": 24},  # Missing required fields
            {"tenure": -1, "monthly_charges": -50}  # Invalid values
        ]
        
        for invalid_data in invalid_data_tests:
            response = api_client.post(predict_url, json=invalid_data, timeout=30)
            assert response.status_code in [400, 422], f"Expected error for data: {invalid_data}"
            
            error_data = response.json()
            assert 'detail' in error_data or 'error' in error_data
            
    def test_prediction_consistency(self, api_client, api_base_url, sample_customer_data):
        """Test prediction consistency across multiple requests."""
        predict_url = urljoin(api_base_url, '/predict')
        
        # Make multiple predictions with same data
        predictions = []
        for _ in range(5):
            response = api_client.post(predict_url, json=sample_customer_data, timeout=30)
            assert response.status_code == 200
            data = response.json()
            predictions.append((data['prediction'], data['probability']))
            time.sleep(0.1)  # Small delay
            
        # All predictions should be identical for same input
        first_prediction = predictions[0]
        for pred in predictions[1:]:
            assert pred == first_prediction, "Predictions should be consistent for same input"
            
    def test_concurrent_requests(self, api_client, api_base_url, sample_customer_data):
        """Test API under concurrent load."""
        import concurrent.futures
        import threading
        
        predict_url = urljoin(api_base_url, '/predict')
        results = []
        errors = []
        
        def make_request():
            try:
                response = api_client.post(predict_url, json=sample_customer_data, timeout=30)
                if response.status_code == 200:
                    results.append(response.json())
                else:
                    errors.append(f"Status {response.status_code}")
            except Exception as e:
                errors.append(str(e))
                
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            concurrent.futures.wait(futures)
            
        # Most requests should succeed
        success_rate = len(results) / (len(results) + len(errors))
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2f} (errors: {errors[:3]})"
        
    def test_response_time_performance(self, api_client, api_base_url, sample_customer_data):
        """Test API response time performance."""
        predict_url = urljoin(api_base_url, '/predict')
        response_times = []
        
        # Make 10 requests and measure response times
        for _ in range(10):
            start_time = time.time()
            response = api_client.post(predict_url, json=sample_customer_data, timeout=30)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
            
        # Calculate statistics
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Performance assertions
        assert avg_response_time < 2.0, f"Average response time too high: {avg_response_time:.2f}s"
        assert max_response_time < 5.0, f"Max response time too high: {max_response_time:.2f}s"
        
    def test_metrics_collection(self, api_client, api_base_url, sample_customer_data):
        """Test that metrics are being collected."""
        metrics_url = urljoin(api_base_url, '/metrics')
        predict_url = urljoin(api_base_url, '/predict')
        
        # Get initial metrics
        initial_response = api_client.get(metrics_url, timeout=30)
        assert initial_response.status_code == 200
        initial_metrics = initial_response.text
        
        # Make a prediction
        pred_response = api_client.post(predict_url, json=sample_customer_data, timeout=30)
        assert pred_response.status_code == 200
        
        # Give metrics time to update
        time.sleep(1)
        
        # Get updated metrics
        updated_response = api_client.get(metrics_url, timeout=30)
        assert updated_response.status_code == 200
        updated_metrics = updated_response.text
        
        # Verify metrics have key indicators
        required_metric_patterns = [
            'http_requests_total',
            'prediction_requests_total',
            'model_prediction_duration_seconds'
        ]
        
        for pattern in required_metric_patterns:
            assert pattern in updated_metrics, f"Missing metric pattern: {pattern}"
            
    @pytest.mark.parametrize("endpoint", [
        "/health",
        "/docs",
        "/metrics",
        "/model/info"
    ])
    def test_endpoint_accessibility(self, api_client, api_base_url, endpoint):
        """Test that all documented endpoints are accessible."""
        url = urljoin(api_base_url, endpoint)
        response = api_client.get(url, timeout=30)
        
        # All these endpoints should return 200
        assert response.status_code == 200, f"Endpoint {endpoint} not accessible"
        
    def test_data_validation_edge_cases(self, api_client, api_base_url):
        """Test edge cases in data validation."""
        predict_url = urljoin(api_base_url, '/predict')
        
        # Test boundary values
        edge_cases = [
            {
                "tenure": 0,  # Minimum tenure
                "monthly_charges": 0.01,  # Very low charge
                "total_charges": 0.01,
                "contract": "Month-to-month",
                "payment_method": "Electronic check",
                "internet_service": "DSL",
                "online_security": "No",
                "online_backup": "No",
                "device_protection": "No",
                "tech_support": "No",
                "streaming_tv": "No",
                "streaming_movies": "No"
            },
            {
                "tenure": 72,  # High tenure
                "monthly_charges": 150.0,  # High charge
                "total_charges": 10000.0,
                "contract": "Two year",
                "payment_method": "Credit card",
                "internet_service": "Fiber optic",
                "online_security": "Yes",
                "online_backup": "Yes",
                "device_protection": "Yes",
                "tech_support": "Yes",
                "streaming_tv": "Yes",
                "streaming_movies": "Yes"
            }
        ]
        
        for edge_case in edge_cases:
            response = api_client.post(predict_url, json=edge_case, timeout=30)
            assert response.status_code == 200, f"Edge case failed: {edge_case}"
            
            data = response.json()
            assert 'prediction' in data
            assert 'probability' in data
            assert data['prediction'] in [0, 1]
            assert 0 <= data['probability'] <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])