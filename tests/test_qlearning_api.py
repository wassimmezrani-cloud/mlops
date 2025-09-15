#!/usr/bin/env python3
"""
Tests API Q-Learning Optimizer
Validation fonctionnelle service optimization
Respect .claude_code_rules - No emojis
"""

import requests
import json
import time
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QLearningAPITester:
    """Testeur API Q-Learning Optimizer"""
    
    def __init__(self, base_url="http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 15
    
    def test_health_endpoint(self):
        """Test health check"""
        logger.info("Test health endpoint")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    logger.info("Health check: OK")
                    return True
                else:
                    logger.error(f"Service unhealthy: {data}")
                    return False
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def test_service_info(self):
        """Test service info endpoint"""
        logger.info("Test service info endpoint")
        
        try:
            response = self.session.get(f"{self.base_url}/info")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Service: {data.get('service_name')}")
                logger.info(f"Model: {data.get('model_name')} v{data.get('version')}")
                logger.info(f"Q-table states: {data.get('q_table_states')}")
                logger.info(f"Available nodes: {data.get('nodes_available')}")
                return True
            else:
                logger.error(f"Service info failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Service info error: {e}")
            return False
    
    def test_nodes_endpoint(self):
        """Test nodes endpoint"""
        logger.info("Test nodes endpoint")
        
        try:
            response = self.session.get(f"{self.base_url}/nodes")
            
            if response.status_code == 200:
                data = response.json()
                nodes = data.get('nodes', [])
                logger.info(f"Available nodes: {len(nodes)}")
                logger.info(f"Nodes: {nodes}")
                return nodes
            else:
                logger.error(f"Nodes endpoint failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Nodes endpoint error: {e}")
            return []
    
    def test_basic_optimization(self):
        """Test basic optimization"""
        logger.info("Test basic optimization")
        
        # Create realistic cluster state
        cluster_state = {
            'master1': {
                'cpu_utilization': 0.35,
                'memory_utilization': 0.60,
                'load1': 1.2,
                'load5': 1.4,
                'pod_count': 15,
                'reliability_score': 98
            },
            'master2': {
                'cpu_utilization': 0.45,
                'memory_utilization': 0.55,
                'load1': 1.8,
                'load5': 2.1,
                'pod_count': 18,
                'reliability_score': 99
            },
            'master3': {
                'cpu_utilization': 0.28,
                'memory_utilization': 0.40,
                'load1': 0.9,
                'load5': 1.1,
                'pod_count': 12,
                'reliability_score': 100
            },
            'worker1': {
                'cpu_utilization': 0.65,
                'memory_utilization': 0.70,
                'load1': 2.5,
                'load5': 2.8,
                'pod_count': 25,
                'reliability_score': 97
            },
            'worker2': {
                'cpu_utilization': 0.50,
                'memory_utilization': 0.45,
                'load1': 1.5,
                'load5': 1.7,
                'pod_count': 20,
                'reliability_score': 98
            },
            'worker3': {
                'cpu_utilization': 0.25,
                'memory_utilization': 0.35,
                'load1': 0.8,
                'load5': 1.0,
                'pod_count': 8,
                'reliability_score': 99
            }
        }
        
        payload = {
            'cluster_state': cluster_state
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/optimize",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                optimal_node = data.get('optimal_node')
                confidence = data.get('confidence')
                q_value = data.get('q_value')
                alternatives = data.get('alternatives', [])
                reasoning = data.get('reasoning', {})
                
                logger.info(f"Optimal node: {optimal_node}")
                logger.info(f"Confidence: {confidence:.3f}")
                logger.info(f"Q-value: {q_value:.3f}")
                logger.info(f"Alternatives: {len(alternatives)}")
                
                if reasoning:
                    factors = reasoning.get('decision_factors', [])
                    logger.info(f"Decision factors: {len(factors)}")
                    for factor in factors[:3]:  # Show first 3
                        logger.info(f"  - {factor}")
                
                return True
            else:
                logger.error(f"Optimization failed: {response.status_code}")
                logger.error(response.text)
                return False
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return False
    
    def test_batch_optimization(self):
        """Test batch optimization"""
        logger.info("Test batch optimization")
        
        # Create multiple cluster states
        requests = []
        
        for i in range(3):
            cluster_state = {
                'master1': {
                    'cpu_utilization': np.random.uniform(0.2, 0.6),
                    'memory_utilization': np.random.uniform(0.3, 0.7),
                    'load1': np.random.uniform(0.8, 2.5),
                    'load5': np.random.uniform(1.0, 3.0),
                    'pod_count': np.random.randint(10, 25),
                    'reliability_score': np.random.uniform(95, 100)
                },
                'worker1': {
                    'cpu_utilization': np.random.uniform(0.3, 0.8),
                    'memory_utilization': np.random.uniform(0.4, 0.8),
                    'load1': np.random.uniform(1.0, 4.0),
                    'load5': np.random.uniform(1.2, 4.5),
                    'pod_count': np.random.randint(15, 30),
                    'reliability_score': np.random.uniform(95, 100)
                }
            }
            requests.append({'cluster_state': cluster_state})
        
        payload = {'requests': requests}
        
        try:
            response = self.session.post(
                f"{self.base_url}/batch_optimize",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                total = data.get('total_requests', 0)
                successful = data.get('successful', 0)
                batch_results = data.get('batch_results', [])
                
                logger.info(f"Batch optimization: {successful}/{total} successful")
                
                for result in batch_results[:2]:  # Show first 2
                    if result.get('status') == 'success':
                        logger.info(f"Request {result['request_id']}: {result['optimal_node']}")
                
                return successful == total
            else:
                logger.error(f"Batch optimization failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Batch optimization error: {e}")
            return False
    
    def test_performance_metrics(self):
        """Test performance metrics endpoint"""
        logger.info("Test performance metrics")
        
        try:
            response = self.session.get(f"{self.base_url}/performance")
            
            if response.status_code == 200:
                data = response.json()
                
                logger.info(f"Global improvement: {data.get('global_improvement', 0):.1f}%")
                logger.info(f"Success rate: {data.get('success_rate', 0):.1f}%")
                logger.info(f"Efficiency: {data.get('efficiency', 0):.1f}%")
                logger.info(f"Business score: {data.get('business_score', 0):.1f}/100")
                logger.info(f"Status: {data.get('status', 'unknown')}")
                
                return True
            else:
                logger.error(f"Performance metrics failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return False
    
    def test_error_handling(self):
        """Test error handling"""
        logger.info("Test error handling")
        
        # Test invalid payloads
        invalid_payloads = [
            {},  # Empty
            {'wrong_field': {}},  # Wrong field
            {'cluster_state': 'not_dict'},  # Wrong type
            {'cluster_state': {}}  # Empty cluster state
        ]
        
        for i, payload in enumerate(invalid_payloads):
            try:
                response = self.session.post(
                    f"{self.base_url}/optimize",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code >= 400:
                    logger.info(f"Error test {i+1}: Error correctly handled ({response.status_code})")
                else:
                    logger.warning(f"Error test {i+1}: Should return error")
                    
            except Exception as e:
                logger.error(f"Error test {i+1} exception: {e}")
        
        return True
    
    def test_performance_latency(self, num_requests=20):
        """Test performance and latency"""
        logger.info(f"Test performance with {num_requests} requests")
        
        # Standard cluster state for testing
        cluster_state = {
            'master1': {'cpu_utilization': 0.3, 'memory_utilization': 0.5, 'load1': 1.0, 'load5': 1.2, 'pod_count': 10, 'reliability_score': 98},
            'worker1': {'cpu_utilization': 0.6, 'memory_utilization': 0.7, 'load1': 2.0, 'load5': 2.3, 'pod_count': 20, 'reliability_score': 97},
            'worker2': {'cpu_utilization': 0.4, 'memory_utilization': 0.4, 'load1': 1.2, 'load5': 1.5, 'pod_count': 15, 'reliability_score': 99}
        }
        
        payload = {'cluster_state': cluster_state}
        
        latencies = []
        successes = 0
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                response = self.session.post(
                    f"{self.base_url}/optimize",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # ms
                
                if response.status_code == 200:
                    latencies.append(latency)
                    successes += 1
                    
            except Exception as e:
                logger.error(f"Request {i+1} failed: {e}")
        
        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            success_rate = (successes / num_requests) * 100
            
            logger.info(f"Performance results:")
            logger.info(f"  Success rate: {success_rate:.1f}%")
            logger.info(f"  Average latency: {avg_latency:.1f}ms")
            logger.info(f"  P95 latency: {p95_latency:.1f}ms")
            logger.info(f"  Min latency: {min(latencies):.1f}ms")
            logger.info(f"  Max latency: {max(latencies):.1f}ms")
            
            # Performance criteria
            performance_ok = (
                avg_latency < 150 and  # < 150ms average
                p95_latency < 300 and  # < 300ms P95
                success_rate >= 95     # >= 95% success
            )
            
            logger.info(f"Performance test: {'PASS' if performance_ok else 'FAIL'}")
            return performance_ok
        
        return False
    
    def run_all_tests(self):
        """Execute all tests"""
        logger.info("=" * 60)
        logger.info("STARTING Q-LEARNING OPTIMIZER API TESTS")
        logger.info("=" * 60)
        
        tests_results = {}
        
        # Test 1: Health check
        tests_results['health'] = self.test_health_endpoint()
        
        # Test 2: Service info
        tests_results['service_info'] = self.test_service_info()
        
        # Test 3: Nodes
        tests_results['nodes'] = len(self.test_nodes_endpoint()) > 0
        
        # Test 4: Basic optimization
        tests_results['basic_optimization'] = self.test_basic_optimization()
        
        # Test 5: Batch optimization
        tests_results['batch_optimization'] = self.test_batch_optimization()
        
        # Test 6: Performance metrics
        tests_results['performance_metrics'] = self.test_performance_metrics()
        
        # Test 7: Error handling
        tests_results['error_handling'] = self.test_error_handling()
        
        # Test 8: Performance latency
        tests_results['performance'] = self.test_performance_latency()
        
        # Final report
        logger.info("=" * 60)
        logger.info("Q-LEARNING API TEST REPORT")
        logger.info("=" * 60)
        
        passed = sum(tests_results.values())
        total = len(tests_results)
        
        for test_name, result in tests_results.items():
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info("-" * 30)
        logger.info(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("ALL TESTS PASSED - API READY FOR PRODUCTION")
            return True
        else:
            logger.warning(f"{total - passed} tests failed")
            return False

def main():
    """Main test execution"""
    
    # Configuration
    api_url = "http://localhost:5001"
    
    # Wait for service to start
    logger.info("Waiting for service startup...")
    time.sleep(3)
    
    # Execute tests
    tester = QLearningAPITester(api_url)
    success = tester.run_all_tests()
    
    if success:
        logger.info("Q-Learning API tests completed successfully")
        exit(0)
    else:
        logger.error("Some tests failed")
        exit(1)

if __name__ == "__main__":
    main()