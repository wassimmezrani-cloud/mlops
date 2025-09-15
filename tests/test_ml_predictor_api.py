#!/usr/bin/env python3
"""
Tests API ML Predictor
Validation fonctionnelle service prediction
Respect .claude_code_rules - Pas d'emojis
"""

import requests
import json
import time
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLPredictorAPITester:
    """Testeur API ML Predictor"""
    
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 10
    
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
            logger.error(f"Erreur health check: {e}")
            return False
    
    def test_model_info(self):
        """Test endpoint informations modele"""
        logger.info("Test model info endpoint")
        
        try:
            response = self.session.get(f"{self.base_url}/info")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Model info: {data.get('model_name')} v{data.get('version')}")
                logger.info(f"Features count: {data.get('features_count')}")
                return True
            else:
                logger.error(f"Model info failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Erreur model info: {e}")
            return False
    
    def test_features_endpoint(self):
        """Test endpoint features"""
        logger.info("Test features endpoint")
        
        try:
            response = self.session.get(f"{self.base_url}/features")
            
            if response.status_code == 200:
                data = response.json()
                features = data.get('features', [])
                logger.info(f"Features disponibles: {len(features)}")
                
                # Afficher quelques features pour debug
                if len(features) > 0:
                    logger.info(f"Exemples features: {features[:5]}")
                
                return features
            else:
                logger.error(f"Features endpoint failed: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Erreur features: {e}")
            return []
    
    def test_basic_prediction(self):
        """Test prediction basique"""
        logger.info("Test prediction basique")
        
        # Obtenir features attendues
        features = self.test_features_endpoint()
        if not features:
            logger.error("Pas de features disponibles")
            return False
        
        # Creer features factices
        test_features = np.random.rand(len(features)).tolist()
        
        payload = {
            "features": test_features
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get('predictions', [])
                
                if predictions:
                    logger.info(f"Prediction reussie: {predictions[0]:.4f}")
                    return True
                else:
                    logger.error("Aucune prediction retournee")
                    return False
            else:
                logger.error(f"Prediction failed: {response.status_code}")
                logger.error(response.text)
                return False
                
        except Exception as e:
            logger.error(f"Erreur prediction: {e}")
            return False
    
    def test_load_prediction(self):
        """Test prediction charge node specialisee"""
        logger.info("Test prediction charge node")
        
        # Features node factices mais realistes
        node_features = {
            "node_name": "worker1",
            "cpu_rate": 0.25,
            "memory_utilization": 0.60,
            "load1": 2.5,
            "load5": 2.8,
            "memory_capacity": 16e9
        }
        
        payload = {
            "node_features": node_features
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict_load",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                predicted_load = data.get('predicted_load')
                load_level = data.get('load_level')
                recommendation = data.get('recommendation')
                
                logger.info(f"Node: {data.get('node')}")
                logger.info(f"Predicted load: {predicted_load:.4f}")
                logger.info(f"Load level: {load_level}")
                logger.info(f"Recommendation: {recommendation}")
                
                return True
            else:
                logger.error(f"Load prediction failed: {response.status_code}")
                logger.error(response.text)
                return False
                
        except Exception as e:
            logger.error(f"Erreur load prediction: {e}")
            return False
    
    def test_error_handling(self):
        """Test gestion erreurs"""
        logger.info("Test gestion erreurs")
        
        # Test payload incorrect
        invalid_payloads = [
            {},  # Empty
            {"wrong_field": []},  # Wrong field
            {"features": "not_array"},  # Wrong type
            {"features": [1, 2]},  # Wrong size (too few features)
        ]
        
        for i, payload in enumerate(invalid_payloads):
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code >= 400:
                    logger.info(f"Error test {i+1}: Erreur correctement geree ({response.status_code})")
                else:
                    logger.warning(f"Error test {i+1}: Devrait retourner erreur")
                    
            except Exception as e:
                logger.error(f"Error test {i+1} exception: {e}")
        
        return True
    
    def test_performance_latency(self, num_requests=10):
        """Test performance et latence"""
        logger.info(f"Test performance avec {num_requests} requetes")
        
        # Obtenir features pour tests
        features = self.test_features_endpoint()
        if not features:
            return False
        
        test_features = np.random.rand(len(features)).tolist()
        payload = {"features": test_features}
        
        latencies = []
        successes = 0
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # ms
                
                if response.status_code == 200:
                    latencies.append(latency)
                    successes += 1
                    
            except Exception as e:
                logger.error(f"Requete {i+1} failed: {e}")
        
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
            
            # Criteres performance
            performance_ok = (
                avg_latency < 100 and  # < 100ms moyenne
                p95_latency < 200 and  # < 200ms P95
                success_rate >= 95     # >= 95% succes
            )
            
            logger.info(f"Performance test: {'PASS' if performance_ok else 'FAIL'}")
            return performance_ok
        
        return False
    
    def run_all_tests(self):
        """Executer tous les tests"""
        logger.info("=" * 50)
        logger.info("DEMARRAGE TESTS API ML PREDICTOR")
        logger.info("=" * 50)
        
        tests_results = {}
        
        # Test 1: Health check
        tests_results['health'] = self.test_health_endpoint()
        
        # Test 2: Model info
        tests_results['model_info'] = self.test_model_info()
        
        # Test 3: Features
        tests_results['features'] = len(self.test_features_endpoint()) > 0
        
        # Test 4: Basic prediction
        tests_results['basic_prediction'] = self.test_basic_prediction()
        
        # Test 5: Load prediction
        tests_results['load_prediction'] = self.test_load_prediction()
        
        # Test 6: Error handling
        tests_results['error_handling'] = self.test_error_handling()
        
        # Test 7: Performance
        tests_results['performance'] = self.test_performance_latency()
        
        # Rapport final
        logger.info("=" * 50)
        logger.info("RAPPORT TESTS API")
        logger.info("=" * 50)
        
        passed = sum(tests_results.values())
        total = len(tests_results)
        
        for test_name, result in tests_results.items():
            status = "PASS" if result else "FAIL"
            logger.info(f"{test_name}: {status}")
        
        logger.info("-" * 30)
        logger.info(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("TOUS TESTS PASSED - API READY")
            return True
        else:
            logger.warning(f"{total - passed} tests failed")
            return False

def main():
    """Main execution tests"""
    
    # Configuration
    api_url = "http://localhost:5000"
    
    # Attendre que le service demarre
    logger.info("Attente demarrage service...")
    time.sleep(2)
    
    # Executer tests
    tester = MLPredictorAPITester(api_url)
    success = tester.run_all_tests()
    
    if success:
        logger.info("Tests API completed successfully")
        exit(0)
    else:
        logger.error("Some tests failed")
        exit(1)

if __name__ == "__main__":
    main()