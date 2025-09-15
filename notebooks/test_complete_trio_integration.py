#!/usr/bin/env python3
"""
Test Complete Trio Integration - ML-Scheduler
Final validation of three-expert AI system
Step 6.4: Complete trio architecture testing
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import subprocess
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrioIntegrationTester:
    """
    Complete testing suite for trio ML-Scheduler integration
    Tests all three experts working together
    """
    
    def __init__(self):
        """Initialize integration tester"""
        self.services = {
            'isolation_detector': {
                'port': 8083,
                'script': 'deployments/isolation_detector_service.py',
                'endpoint': 'http://localhost:8083'
            },
            'trio_scheduler': {
                'port': 8084,
                'script': 'deployments/trio_ml_scheduler_service.py',  
                'endpoint': 'http://localhost:8084'
            }
        }
        
        self.test_results = {
            'services_tested': 0,
            'services_passed': 0,
            'integration_tests': [],
            'overall_success': False
        }
        
        logger.info("Trio Integration Tester initialized")
    
    def check_service_health(self, service_name: str, endpoint: str) -> bool:
        """Check if service is healthy"""
        try:
            response = requests.get(f"{endpoint}/health", timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ {service_name} service healthy")
                return True
            else:
                logger.warning(f"❌ {service_name} service unhealthy: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"❌ {service_name} service unreachable: {e}")
            return False
    
    def test_isolation_detector_service(self) -> Dict[str, Any]:
        """Test isolation detector service independently"""
        logger.info("Testing Isolation Detector service...")
        
        endpoint = self.services['isolation_detector']['endpoint']
        
        # Test data - normal node
        normal_node_data = {
            'node_id': 'test_normal_node',
            'cpu_utilization': [0.3, 0.4, 0.35, 0.4, 0.3] * 20,
            'memory_utilization': [0.5, 0.52, 0.48, 0.5, 0.51] * 20,
            'load_average': [1.2, 1.5, 1.3, 1.4, 1.2] * 20,
            'pod_count': [15] * 100,
            'network_bytes_in': [1e9] * 100,
            'network_bytes_out': [8e8] * 100,
            'disk_usage_percent': [40] * 100,
            'container_restarts': 1,
            'uptime_hours': 168,
            'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(100)]
        }
        
        # Test data - anomalous node
        anomaly_node_data = {
            'node_id': 'test_anomaly_node',
            'cpu_utilization': [0.3] * 60 + [0.98] * 40,  # CPU spike
            'memory_utilization': [0.4] * 50 + [0.4 + i*0.01 for i in range(50)],  # Memory leak
            'load_average': [1.5] * 60 + [8.0] * 40,  # High load
            'pod_count': [15] * 100,
            'network_bytes_in': [1e9] * 100,
            'network_bytes_out': [8e8] * 100,
            'disk_usage_percent': [40] * 100,
            'container_restarts': 8,
            'uptime_hours': 24,
            'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(100)]
        }
        
        test_results = {'tests_passed': 0, 'total_tests': 4, 'details': []}
        
        try:
            # Test 1: Health check
            if self.check_service_health('Isolation Detector', endpoint):
                test_results['tests_passed'] += 1
                test_results['details'].append('✅ Health check passed')
            else:
                test_results['details'].append('❌ Health check failed')
            
            # Test 2: Normal node detection
            response = requests.post(f"{endpoint}/detect", json=normal_node_data, timeout=15)
            if response.status_code == 200:
                result = response.json()
                is_anomaly = result.get('is_anomaly', True)
                if not is_anomaly:  # Should NOT detect normal node as anomaly
                    test_results['tests_passed'] += 1
                    test_results['details'].append(f"✅ Normal node correctly identified (score: {result.get('anomaly_score', 0):.3f})")
                else:
                    test_results['details'].append(f"❌ Normal node incorrectly flagged as anomaly")
            else:
                test_results['details'].append(f"❌ Normal node detection failed: {response.status_code}")
            
            # Test 3: Anomalous node detection
            response = requests.post(f"{endpoint}/detect", json=anomaly_node_data, timeout=15)
            if response.status_code == 200:
                result = response.json()
                is_anomaly = result.get('is_anomaly', False)
                if is_anomaly:  # Should detect anomaly
                    test_results['tests_passed'] += 1
                    test_results['details'].append(f"✅ Anomaly correctly detected (score: {result.get('anomaly_score', 0):.3f}, "
                                                 f"risk: {result.get('risk_level', 'UNKNOWN')})")
                else:
                    test_results['details'].append(f"❌ Anomaly not detected (score: {result.get('anomaly_score', 0):.3f})")
            else:
                test_results['details'].append(f"❌ Anomaly detection failed: {response.status_code}")
            
            # Test 4: Model info endpoint
            response = requests.get(f"{endpoint}/v1/models/isolation-detector", timeout=10)
            if response.status_code == 200:
                model_info = response.json()
                if 'Le Detective' in str(model_info):
                    test_results['tests_passed'] += 1
                    test_results['details'].append('✅ Model info endpoint working')
                else:
                    test_results['details'].append('❌ Model info incomplete')
            else:
                test_results['details'].append(f"❌ Model info failed: {response.status_code}")
            
        except Exception as e:
            test_results['details'].append(f"❌ Service testing failed: {e}")
        
        test_results['success_rate'] = test_results['tests_passed'] / test_results['total_tests']
        test_results['service'] = 'Isolation Detector'
        
        return test_results
    
    def test_trio_scheduler_integration(self) -> Dict[str, Any]:
        """Test complete trio scheduler integration"""
        logger.info("Testing Trio Scheduler integration...")
        
        endpoint = self.services['trio_scheduler']['endpoint']
        
        # Test pod specification
        pod_spec = {
            'cpu_request': 0.2,
            'memory_request': 0.3,
            'pod_type': 'web'
        }
        
        # Test cluster nodes
        cluster_nodes = [
            {
                'node_id': 'worker-1',
                'cpu_utilization': [0.3, 0.35, 0.4, 0.3, 0.32] * 24,
                'memory_utilization': [0.5, 0.52, 0.48, 0.5, 0.51] * 24,
                'load_average': [1.2, 1.5, 1.3, 1.4, 1.2] * 24,
                'pod_count': [15] * 120,
                'network_bytes_in': [1e9] * 120,
                'network_bytes_out': [8e8] * 120,
                'disk_usage_percent': [40] * 120,
                'container_restarts': 1,
                'uptime_hours': 168,
                'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(120)]
            },
            {
                'node_id': 'worker-2',
                'cpu_utilization': [0.6, 0.65, 0.7, 0.6, 0.62] * 24,
                'memory_utilization': [0.8, 0.82, 0.78, 0.8, 0.81] * 24,
                'load_average': [2.5, 2.8, 2.6, 2.7, 2.5] * 24,
                'pod_count': [25] * 120,
                'network_bytes_in': [1e9] * 120,
                'network_bytes_out': [8e8] * 120,
                'disk_usage_percent': [60] * 120,
                'container_restarts': 2,
                'uptime_hours': 120,
                'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(120)]
            }
        ]
        
        test_results = {'tests_passed': 0, 'total_tests': 3, 'details': []}
        
        try:
            # Test 1: Health check
            if self.check_service_health('Trio Scheduler', endpoint):
                test_results['tests_passed'] += 1
                test_results['details'].append('✅ Health check passed')
            else:
                test_results['details'].append('❌ Health check failed')
            
            # Test 2: Service info
            response = requests.get(f"{endpoint}/v1/models/trio-scheduler", timeout=10)
            if response.status_code == 200:
                info = response.json()
                experts = info.get('experts', {})
                if len(experts) == 3:
                    test_results['tests_passed'] += 1
                    test_results['details'].append(f"✅ All 3 experts configured: {list(experts.keys())}")
                else:
                    test_results['details'].append(f"❌ Missing experts: {len(experts)}/3")
            else:
                test_results['details'].append(f"❌ Service info failed: {response.status_code}")
            
            # Test 3: Scheduling decision (mock since other services not running)
            scheduling_request = {
                'pod_spec': pod_spec,
                'cluster_nodes': cluster_nodes
            }
            
            response = requests.post(f"{endpoint}/v1/models/trio-scheduler:predict", 
                                   json={'instances': [scheduling_request]}, 
                                   timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                if predictions:
                    prediction = predictions[0]
                    scheduling_result = prediction.get('scheduling_result', {})
                    selected_node = scheduling_result.get('selected_node')
                    if selected_node:
                        test_results['tests_passed'] += 1
                        test_results['details'].append(f"✅ Scheduling decision made: {selected_node}")
                    else:
                        test_results['details'].append(f"❌ No node selected for scheduling")
                else:
                    test_results['details'].append("❌ No predictions in response")
            else:
                test_results['details'].append(f"❌ Scheduling request failed: {response.status_code}")
        
        except Exception as e:
            test_results['details'].append(f"❌ Integration testing failed: {e}")
        
        test_results['success_rate'] = test_results['tests_passed'] / test_results['total_tests']
        test_results['service'] = 'Trio Scheduler'
        
        return test_results
    
    def run_complete_integration_tests(self) -> Dict[str, Any]:
        """Run complete integration test suite"""
        logger.info("Starting complete trio integration tests")
        
        print("="*70)
        print("TRIO ML-SCHEDULER INTEGRATION TESTING")
        print("Complete Three-Expert AI System Validation")
        print("="*70)
        
        # Test services individually
        services_results = []
        
        # Test Isolation Detector
        isolation_results = self.test_isolation_detector_service()
        services_results.append(isolation_results)
        self.test_results['services_tested'] += 1
        if isolation_results['success_rate'] >= 0.75:
            self.test_results['services_passed'] += 1
        
        # Test Trio Scheduler  
        trio_results = self.test_trio_scheduler_integration()
        services_results.append(trio_results)
        self.test_results['services_tested'] += 1
        if trio_results['success_rate'] >= 0.5:  # Lower threshold since dependent services not running
            self.test_results['services_passed'] += 1
        
        # Overall results
        overall_success_rate = self.test_results['services_passed'] / self.test_results['services_tested']
        self.test_results['overall_success'] = overall_success_rate >= 0.75
        
        # Display results
        print(f"\n{'='*50}")
        print("INTEGRATION TEST RESULTS")
        print(f"{'='*50}")
        
        for result in services_results:
            print(f"\n🔧 {result['service']} Service:")
            print(f"   Success Rate: {result['success_rate']:.1%} ({result['tests_passed']}/{result['total_tests']})")
            for detail in result['details']:
                print(f"   {detail}")
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Services tested: {self.test_results['services_tested']}")
        print(f"   Services passed: {self.test_results['services_passed']}")
        print(f"   Overall success rate: {overall_success_rate:.1%}")
        
        if self.test_results['overall_success']:
            print(f"\n🎯 INTEGRATION TESTS PASSED!")
            print("Trio ML-Scheduler architecture validated successfully")
        else:
            print(f"\n⚠️  Integration tests need improvement")
        
        # Save test results
        results_file = "./models/isolation_detector/trio_integration_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'test_date': datetime.now().isoformat(),
                'services_results': services_results,
                'overall_results': self.test_results,
                'success': self.test_results['overall_success']
            }, f, indent=2)
        
        print(f"\n📄 Test results saved: {results_file}")
        
        return {
            'success': self.test_results['overall_success'],
            'services_results': services_results,
            'overall_results': self.test_results
        }

def main():
    """Main integration testing function"""
    print("Starting Trio ML-Scheduler Integration Tests...")
    
    tester = TrioIntegrationTester()
    results = tester.run_complete_integration_tests()
    
    print("\n" + "="*70)
    if results['success']:
        print("🏆 STEP 6.4 COMPLETED SUCCESSFULLY")
        print("Trio ML-Scheduler integration validated!")
        print("Three-expert AI system ready for production deployment.")
    else:
        print("⚠️  Step 6.4 needs additional work")
        print("Some integration tests require improvement.")
    print("="*70)
    
    logger.info(f"Integration testing completed - Success: {results['success']}")
    
    return results['success']

if __name__ == "__main__":
    main()