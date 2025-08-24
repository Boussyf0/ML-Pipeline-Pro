#!/usr/bin/env python3
"""
Load Testing Script for MLOps Pipeline API
Simulates concurrent users and measures performance under load.
"""

import argparse
import asyncio
import aiohttp
import time
import json
import logging
import statistics
from typing import List, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import random

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    timestamp: float
    status_code: int
    response_time: float
    success: bool
    error_message: str = ""


class LoadTester:
    """Load testing class for API endpoints."""
    
    def __init__(self, base_url: str, concurrent_users: int = 10, duration: int = 300):
        self.base_url = base_url.rstrip('/')
        self.concurrent_users = concurrent_users
        self.duration = duration
        self.results: List[LoadTestResult] = []
        
        # Sample data variants for realistic testing
        self.sample_data_variants = [
            {
                "tenure": random.randint(1, 72),
                "monthly_charges": round(random.uniform(18.0, 118.0), 2),
                "total_charges": round(random.uniform(18.0, 8500.0), 2),
                "contract": random.choice(["Month-to-month", "One year", "Two year"]),
                "payment_method": random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"]),
                "internet_service": random.choice(["DSL", "Fiber optic", "No"]),
                "online_security": random.choice(["Yes", "No", "No internet service"]),
                "online_backup": random.choice(["Yes", "No", "No internet service"]),
                "device_protection": random.choice(["Yes", "No", "No internet service"]),
                "tech_support": random.choice(["Yes", "No", "No internet service"]),
                "streaming_tv": random.choice(["Yes", "No", "No internet service"]),
                "streaming_movies": random.choice(["Yes", "No", "No internet service"])
            } for _ in range(50)  # Create 50 variants
        ]
        
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, data: Dict = None) -> LoadTestResult:
        """Make a single HTTP request and record the result."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if data:
                async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    await response.read()  # Consume response body
                    end_time = time.time()
                    
                    return LoadTestResult(
                        timestamp=start_time,
                        status_code=response.status,
                        response_time=end_time - start_time,
                        success=response.status == 200
                    )
            else:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    await response.read()  # Consume response body
                    end_time = time.time()
                    
                    return LoadTestResult(
                        timestamp=start_time,
                        status_code=response.status,
                        response_time=end_time - start_time,
                        success=response.status == 200
                    )
                    
        except Exception as e:
            end_time = time.time()
            return LoadTestResult(
                timestamp=start_time,
                status_code=0,
                response_time=end_time - start_time,
                success=False,
                error_message=str(e)
            )
            
    async def simulate_user(self, user_id: int, session: aiohttp.ClientSession):
        """Simulate a single user making requests for the duration."""
        logger.info(f"Starting user {user_id}")
        start_time = time.time()
        requests_made = 0
        
        while time.time() - start_time < self.duration:
            # Randomly choose endpoint and data
            endpoint_choice = random.choices(
                ['/health', '/predict', '/model/info', '/metrics'],
                weights=[10, 70, 15, 5],  # Weighted distribution
                k=1
            )[0]
            
            if endpoint_choice == '/predict':
                # Use random sample data
                data = random.choice(self.sample_data_variants)
                result = await self.make_request(session, endpoint_choice, data)
            else:
                result = await self.make_request(session, endpoint_choice)
                
            self.results.append(result)
            requests_made += 1
            
            # Add small delay to simulate real user behavior
            await asyncio.sleep(random.uniform(0.1, 2.0))
            
        logger.info(f"User {user_id} completed {requests_made} requests")
        
    async def run_load_test(self) -> Dict[str, Any]:
        """Run the complete load test."""
        logger.info(f"Starting load test: {self.concurrent_users} users for {self.duration} seconds")
        logger.info(f"Target URL: {self.base_url}")
        
        connector = aiohttp.TCPConnector(limit=self.concurrent_users * 2, limit_per_host=self.concurrent_users * 2)
        
        async with aiohttp.ClientSession(
            connector=connector,
            headers={'Content-Type': 'application/json'}
        ) as session:
            # Start all users concurrently
            tasks = []
            for user_id in range(self.concurrent_users):
                task = asyncio.create_task(self.simulate_user(user_id, session))
                tasks.append(task)
                
            # Wait for all users to complete
            await asyncio.gather(*tasks)
            
        logger.info(f"Load test completed. Total requests: {len(self.results)}")
        return self.analyze_results()
        
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze load test results and generate statistics."""
        if not self.results:
            return {"error": "No results to analyze"}
            
        # Separate successful and failed requests
        successful_requests = [r for r in self.results if r.success]
        failed_requests = [r for r in self.results if not r.success]
        
        # Calculate response time statistics
        response_times = [r.response_time for r in successful_requests]
        
        if response_times:
            response_stats = {
                'min': min(response_times),
                'max': max(response_times),
                'mean': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'p95': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                'p99': statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times),
                'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
        else:
            response_stats = {
                'min': 0, 'max': 0, 'mean': 0, 'median': 0,
                'p95': 0, 'p99': 0, 'std_dev': 0
            }
            
        # Calculate throughput
        total_time = self.duration
        throughput = len(successful_requests) / total_time if total_time > 0 else 0
        
        # Status code distribution
        status_codes = {}
        for result in self.results:
            status_codes[result.status_code] = status_codes.get(result.status_code, 0) + 1
            
        # Error analysis
        error_types = {}
        for result in failed_requests:
            error_key = result.error_message or f"HTTP_{result.status_code}"
            error_types[error_key] = error_types.get(error_key, 0) + 1
            
        # Calculate success metrics
        total_requests = len(self.results)
        success_rate = len(successful_requests) / total_requests * 100 if total_requests > 0 else 0
        error_rate = len(failed_requests) / total_requests * 100 if total_requests > 0 else 0
        
        return {
            'test_configuration': {
                'concurrent_users': self.concurrent_users,
                'duration_seconds': self.duration,
                'target_url': self.base_url
            },
            'summary': {
                'total_requests': total_requests,
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate_percent': round(success_rate, 2),
                'error_rate_percent': round(error_rate, 2),
                'throughput_rps': round(throughput, 2)
            },
            'response_time_stats': {
                'min_seconds': round(response_stats['min'], 3),
                'max_seconds': round(response_stats['max'], 3),
                'mean_seconds': round(response_stats['mean'], 3),
                'median_seconds': round(response_stats['median'], 3),
                'p95_seconds': round(response_stats['p95'], 3),
                'p99_seconds': round(response_stats['p99'], 3),
                'std_dev_seconds': round(response_stats['std_dev'], 3)
            },
            'status_codes': status_codes,
            'error_types': error_types,
            'performance_assessment': self.assess_performance(response_stats, success_rate, throughput)
        }
        
    def assess_performance(self, response_stats: Dict, success_rate: float, throughput: float) -> Dict[str, Any]:
        """Assess overall performance and provide recommendations."""
        issues = []
        warnings = []
        
        # Check success rate
        if success_rate < 95:
            issues.append(f"Low success rate: {success_rate:.1f}% (expected >95%)")
        elif success_rate < 99:
            warnings.append(f"Moderate success rate: {success_rate:.1f}% (recommended >99%)")
            
        # Check response times
        if response_stats['p95'] > 2.0:
            issues.append(f"High P95 response time: {response_stats['p95']:.3f}s (expected <2.0s)")
        elif response_stats['p95'] > 1.0:
            warnings.append(f"Moderate P95 response time: {response_stats['p95']:.3f}s (recommended <1.0s)")
            
        if response_stats['mean'] > 1.0:
            issues.append(f"High mean response time: {response_stats['mean']:.3f}s (expected <1.0s)")
        elif response_stats['mean'] > 0.5:
            warnings.append(f"Moderate mean response time: {response_stats['mean']:.3f}s (recommended <0.5s)")
            
        # Check throughput
        expected_min_throughput = self.concurrent_users * 0.5  # Conservative estimate
        if throughput < expected_min_throughput:
            issues.append(f"Low throughput: {throughput:.2f} RPS (expected >{expected_min_throughput:.2f} RPS)")
            
        # Overall assessment
        if issues:
            overall_status = "POOR"
        elif warnings:
            overall_status = "ACCEPTABLE"
        else:
            overall_status = "EXCELLENT"
            
        return {
            'overall_status': overall_status,
            'issues': issues,
            'warnings': warnings,
            'recommendations': self.get_recommendations(issues, warnings)
        }
        
    def get_recommendations(self, issues: List[str], warnings: List[str]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if any("success rate" in issue.lower() for issue in issues + warnings):
            recommendations.append("Investigate error logs and fix application bugs")
            recommendations.append("Add health checks and circuit breakers")
            
        if any("response time" in issue.lower() for issue in issues + warnings):
            recommendations.append("Profile application for performance bottlenecks")
            recommendations.append("Consider adding caching layers")
            recommendations.append("Optimize database queries and model inference")
            recommendations.append("Scale up resources (CPU, memory)")
            
        if any("throughput" in issue.lower() for issue in issues + warnings):
            recommendations.append("Scale out with more replicas")
            recommendations.append("Implement load balancing")
            recommendations.append("Consider async processing for heavy operations")
            
        if not recommendations:
            recommendations.append("Performance is good - monitor for any degradation")
            
        return recommendations
        
    def print_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of the load test results."""
        print("\n" + "="*80)
        print("LOAD TEST RESULTS SUMMARY")
        print("="*80)
        
        # Test configuration
        config = results['test_configuration']
        print(f"\nTest Configuration:")
        print(f"  Concurrent Users: {config['concurrent_users']}")
        print(f"  Duration: {config['duration_seconds']} seconds")
        print(f"  Target URL: {config['target_url']}")
        
        # Summary statistics
        summary = results['summary']
        print(f"\nRequest Summary:")
        print(f"  Total Requests: {summary['total_requests']}")
        print(f"  Successful: {summary['successful_requests']} ({summary['success_rate_percent']}%)")
        print(f"  Failed: {summary['failed_requests']} ({summary['error_rate_percent']}%)")
        print(f"  Throughput: {summary['throughput_rps']} requests/second")
        
        # Response times
        rt_stats = results['response_time_stats']
        print(f"\nResponse Time Statistics:")
        print(f"  Mean: {rt_stats['mean_seconds']}s")
        print(f"  Median: {rt_stats['median_seconds']}s")
        print(f"  P95: {rt_stats['p95_seconds']}s")
        print(f"  P99: {rt_stats['p99_seconds']}s")
        print(f"  Min: {rt_stats['min_seconds']}s")
        print(f"  Max: {rt_stats['max_seconds']}s")
        
        # Performance assessment
        assessment = results['performance_assessment']
        print(f"\nPerformance Assessment: {assessment['overall_status']}")
        
        if assessment['issues']:
            print(f"\n❌ Issues Found:")
            for issue in assessment['issues']:
                print(f"  - {issue}")
                
        if assessment['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in assessment['warnings']:
                print(f"  - {warning}")
                
        if assessment['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in assessment['recommendations']:
                print(f"  - {rec}")
                
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save detailed results to JSON file."""
        try:
            # Add raw results for detailed analysis
            detailed_results = results.copy()
            detailed_results['raw_results'] = [
                {
                    'timestamp': r.timestamp,
                    'status_code': r.status_code,
                    'response_time': r.response_time,
                    'success': r.success,
                    'error_message': r.error_message
                } for r in self.results
            ]
            
            with open(output_path, 'w') as f:
                json.dump(detailed_results, f, indent=2)
                
            logger.info(f"Detailed results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


async def main():
    parser = argparse.ArgumentParser(description='Load test MLOps API')
    parser.add_argument('--url', required=True, help='Base URL of the API to test')
    parser.add_argument('--users', type=int, default=10, help='Number of concurrent users')
    parser.add_argument('--duration', type=int, default=300, help='Test duration in seconds')
    parser.add_argument('--output', default='load_test_results.json', help='Output file for results')
    
    args = parser.parse_args()
    
    try:
        tester = LoadTester(args.url, args.users, args.duration)
        results = await tester.run_load_test()
        
        tester.print_summary(results)
        tester.save_results(results, args.output)
        
        # Exit with error code if performance is poor
        if results['performance_assessment']['overall_status'] == 'POOR':
            logger.error("Load test failed: Poor performance detected")
            exit(1)
        elif results['performance_assessment']['overall_status'] == 'ACCEPTABLE':
            logger.warning("Load test completed with warnings")
            exit(0)
        else:
            logger.info("Load test completed successfully")
            exit(0)
            
    except Exception as e:
        logger.error(f"Load test failed: {e}")
        exit(1)


if __name__ == '__main__':
    asyncio.run(main())