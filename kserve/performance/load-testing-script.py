#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

@dataclass
class LoadTestConfig:
    service_name: str
    target_url: str
    concurrent_users: List[int]
    test_duration_seconds: int
    ramp_up_duration_seconds: int
    request_payload: Dict[str, Any]
    performance_targets: Dict[str, float]

@dataclass
class TestResult:
    timestamp: str
    service_name: str
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_rps: float
    error_rate_percent: float
    target_met: bool

class MLSchedulerLoadTester:
    """Production load tester for ML-Scheduler KServe services"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[TestResult] = []
        
    def _setup_logging(self) -> logging.Logger:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
        
    async def create_session(self):
        """Create aiohttp session with optimized settings"""
        connector = aiohttp.TCPConnector(
            limit=1000,
            limit_per_host=200,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=20
        )
        
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'ML-Scheduler-LoadTester/1.0'
        }
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            
    async def make_request(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make single HTTP request and measure response time"""
        start_time = time.perf_counter()
        
        try:
            async with self.session.post(url, json=payload) as response:
                response_time_ms = (time.perf_counter() - start_time) * 1000
                
                if response.status == 200:
                    await response.text()  # Consume response
                    return {
                        'success': True,
                        'response_time_ms': response_time_ms,
                        'status_code': response.status
                    }
                else:
                    return {
                        'success': False,
                        'response_time_ms': response_time_ms,
                        'status_code': response.status,
                        'error': f'HTTP {response.status}'
                    }
                    
        except asyncio.TimeoutError:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            return {
                'success': False,
                'response_time_ms': response_time_ms,
                'error': 'Timeout'
            }
        except Exception as e:
            response_time_ms = (time.perf_counter() - start_time) * 1000
            return {
                'success': False,
                'response_time_ms': response_time_ms,
                'error': str(e)
            }
            
    async def run_load_test(self, config: LoadTestConfig, concurrent_users: int) -> TestResult:
        """Run load test for specific concurrency level"""
        
        self.logger.info(f"Starting load test for {config.service_name} with {concurrent_users} concurrent users")
        
        # Calculate request rate and intervals
        requests_per_second = concurrent_users / config.ramp_up_duration_seconds if config.ramp_up_duration_seconds > 0 else concurrent_users
        request_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        
        start_time = time.time()
        end_time = start_time + config.test_duration_seconds
        
        tasks = []
        request_results = []
        
        # Generate requests with gradual ramp-up
        async def request_worker(user_id: int):
            # Stagger start times during ramp-up period
            ramp_delay = (user_id * config.ramp_up_duration_seconds) / concurrent_users
            await asyncio.sleep(ramp_delay)
            
            user_results = []
            
            while time.time() < end_time:
                result = await self.make_request(config.target_url, config.request_payload)
                result['user_id'] = user_id
                result['timestamp'] = time.time()
                user_results.append(result)
                
                # Wait before next request (simulate realistic load)
                await asyncio.sleep(max(0, request_interval))
                
            return user_results
        
        # Create and run concurrent tasks
        tasks = [request_worker(user_id) for user_id in range(concurrent_users)]
        user_results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for user_results in user_results_list:
            if isinstance(user_results, list):
                request_results.extend(user_results)
                
        # Analyze results
        return self._analyze_results(config, concurrent_users, request_results)
        
    def _analyze_results(self, config: LoadTestConfig, concurrent_users: int, results: List[Dict[str, Any]]) -> TestResult:
        """Analyze load test results and calculate metrics"""
        
        total_requests = len(results)
        successful_requests = sum(1 for r in results if r.get('success', False))
        failed_requests = total_requests - successful_requests
        
        if total_requests == 0:
            return TestResult(
                timestamp=datetime.now().isoformat(),
                service_name=config.service_name,
                concurrent_users=concurrent_users,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time_ms=0,
                p50_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                throughput_rps=0,
                error_rate_percent=100,
                target_met=False
            )
        
        # Calculate response time metrics
        response_times = [r['response_time_ms'] for r in results if 'response_time_ms' in r]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times)
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0
            
        # Calculate throughput
        if results:
            test_duration = max(r['timestamp'] for r in results) - min(r['timestamp'] for r in results)
            throughput_rps = successful_requests / test_duration if test_duration > 0 else 0
        else:
            throughput_rps = 0
            
        # Calculate error rate
        error_rate_percent = (failed_requests / total_requests) * 100 if total_requests > 0 else 100
        
        # Check if performance targets are met
        targets = config.performance_targets
        target_met = (
            p99_response_time <= targets.get('latency_p99_ms', float('inf')) and
            throughput_rps >= targets.get('throughput_rps', 0) and
            error_rate_percent <= targets.get('error_rate_percent', 100)
        )
        
        result = TestResult(
            timestamp=datetime.now().isoformat(),
            service_name=config.service_name,
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time_ms=round(avg_response_time, 2),
            p50_response_time_ms=round(p50_response_time, 2),
            p95_response_time_ms=round(p95_response_time, 2),
            p99_response_time_ms=round(p99_response_time, 2),
            throughput_rps=round(throughput_rps, 2),
            error_rate_percent=round(error_rate_percent, 2),
            target_met=target_met
        )
        
        self.logger.info(f"Test completed for {config.service_name} ({concurrent_users} users): "
                        f"RPS={result.throughput_rps}, P99={result.p99_response_time_ms}ms, "
                        f"Errors={result.error_rate_percent}%, Target Met={result.target_met}")
        
        return result
        
    async def run_comprehensive_test(self) -> List[TestResult]:
        """Run comprehensive load tests for all ML-Scheduler services"""
        
        # Test configurations for each service
        test_configs = [
            LoadTestConfig(
                service_name="xgboost-predictor",
                target_url="https://xgboost.ml-scheduler.hydatis.local/v1/models/xgboost-load-predictor:predict",
                concurrent_users=[10, 50, 100, 200, 500],
                test_duration_seconds=180,
                ramp_up_duration_seconds=30,
                request_payload={
                    "instances": [{
                        "cpu_usage_rate": 0.75,
                        "memory_usage_rate": 0.65,
                        "network_io": 1024000,
                        "disk_io": 512000,
                        "hour_of_day": 14,
                        "day_of_week": 2,
                        "business_hours": 1,
                        "cpu_pressure": 0.25,
                        "memory_pressure": 0.35,
                        "efficiency_ratio": 0.85,
                        "load_trend_1h": 0.15,
                        "resource_volatility": 0.08
                    }]
                },
                performance_targets={
                    "latency_p99_ms": 75,
                    "throughput_rps": 500,
                    "error_rate_percent": 0.1
                }
            ),
            LoadTestConfig(
                service_name="qlearning-optimizer", 
                target_url="https://qlearning.ml-scheduler.hydatis.local/v1/models/qlearning-placement-optimizer:predict",
                concurrent_users=[5, 25, 50, 100, 300],
                test_duration_seconds=180,
                ramp_up_duration_seconds=30,
                request_payload={
                    "instances": [{
                        "node_cpu_available": 0.40,
                        "node_memory_available": 0.55,
                        "node_load_score": 0.60,
                        "pod_cpu_request": 0.20,
                        "pod_memory_request": 0.15,
                        "pod_priority": 0,
                        "cluster_fragmentation": 0.30,
                        "node_affinity_score": 0.80
                    }]
                },
                performance_targets={
                    "latency_p99_ms": 100,
                    "throughput_rps": 300,
                    "error_rate_percent": 0.1
                }
            ),
            LoadTestConfig(
                service_name="isolation-detector",
                target_url="https://isolation.ml-scheduler.hydatis.local/v1/models/isolation-anomaly-detector:predict",
                concurrent_users=[10, 25, 50, 100, 200],
                test_duration_seconds=180,
                ramp_up_duration_seconds=30,
                request_payload={
                    "instances": [{
                        "cpu_usage_rate": 0.85,
                        "memory_usage_rate": 0.70,
                        "network_io": 2048000,
                        "disk_io": 1024000,
                        "cpu_pressure": 0.15,
                        "memory_pressure": 0.25,
                        "load_average": 2.5,
                        "context_switches": 15000,
                        "interrupt_rate": 8000,
                        "swap_usage": 0.05,
                        "disk_utilization": 0.75,
                        "network_errors": 10
                    }]
                },
                performance_targets={
                    "latency_p99_ms": 50,
                    "throughput_rps": 200,
                    "error_rate_percent": 0.1
                }
            )
        ]
        
        await self.create_session()
        
        try:
            all_results = []
            
            for config in test_configs:
                self.logger.info(f"Starting comprehensive test for {config.service_name}")
                
                for concurrent_users in config.concurrent_users:
                    result = await self.run_load_test(config, concurrent_users)
                    all_results.append(result)
                    self.results.append(result)
                    
                    # Brief pause between tests
                    await asyncio.sleep(10)
                    
            return all_results
            
        finally:
            await self.close_session()
            
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive load test report"""
        
        if not self.results:
            return {"error": "No test results available"}
            
        report = {
            "test_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.results),
                "services_tested": list(set(r.service_name for r in self.results))
            },
            "service_results": {},
            "performance_summary": {
                "targets_met": sum(1 for r in self.results if r.target_met),
                "total_tests": len(self.results),
                "success_rate": (sum(1 for r in self.results if r.target_met) / len(self.results)) * 100
            }
        }
        
        # Organize results by service
        for result in self.results:
            service_name = result.service_name
            if service_name not in report["service_results"]:
                report["service_results"][service_name] = []
                
            report["service_results"][service_name].append(asdict(result))
            
        # Calculate best performance for each service
        report["best_performance"] = {}
        for service_name in report["service_results"]:
            service_results = [r for r in self.results if r.service_name == service_name]
            
            best_throughput = max(r.throughput_rps for r in service_results)
            best_latency = min(r.p99_response_time_ms for r in service_results)
            lowest_error_rate = min(r.error_rate_percent for r in service_results)
            
            report["best_performance"][service_name] = {
                "max_throughput_rps": best_throughput,
                "min_p99_latency_ms": best_latency,
                "min_error_rate_percent": lowest_error_rate
            }
            
        return report
        
    def save_results(self, filename: str = "ml_scheduler_load_test_results.json"):
        """Save load test results to JSON file"""
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Load test results saved to {filename}")

async def main():
    """Main function to run comprehensive load tests"""
    
    tester = MLSchedulerLoadTester()
    
    print("=" * 80)
    print("ML-SCHEDULER KSERVE PRODUCTION LOAD TESTING")
    print("=" * 80)
    
    try:
        results = await tester.run_comprehensive_test()
        
        print("\n" + "=" * 80)
        print("LOAD TEST RESULTS SUMMARY")
        print("=" * 80)
        
        # Display summary
        for result in results:
            status = "✅ PASS" if result.target_met else "❌ FAIL"
            print(f"{status} {result.service_name} ({result.concurrent_users} users): "
                  f"RPS={result.throughput_rps}, P99={result.p99_response_time_ms}ms, "
                  f"Errors={result.error_rate_percent}%")
        
        # Generate and save report
        tester.save_results()
        
        # Summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.target_met)
        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print(f"\n📊 Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)")
        
        if success_rate >= 90:
            print("🎉 LOAD TESTING: PRODUCTION READY!")
        elif success_rate >= 75:
            print("⚠️  LOAD TESTING: NEEDS OPTIMIZATION")
        else:
            print("❌ LOAD TESTING: SIGNIFICANT ISSUES DETECTED")
            
    except Exception as e:
        logging.error(f"Load testing failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())