#!/usr/bin/env python3
"""
ML-Scheduler Data Collection Module
Collecte donnees historiques Prometheus pour algorithmes ML
Structure MLOps Lifecycle conforme
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import json
import logging
warnings.filterwarnings('ignore')

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrometheusCollector:
    """Collecteur donnees Prometheus pour ML-Scheduler"""
    
    def __init__(self, prometheus_url="http://10.110.190.83:9090"):
        self.prometheus_url = prometheus_url
        self.api_url = f"{prometheus_url}/api/v1"
        self.session = requests.Session()
        self.session.timeout = 30
        
    def test_connection(self):
        """Test connexion basique Prometheus"""
        logger.info("Test connexion Prometheus")
        
        try:
            response = self.session.get(f"{self.api_url}/query?query=up")
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success':
                    services_count = len(data['data']['result'])
                    logger.info(f"Connexion Prometheus reussie: {services_count} services")
                    return True
                else:
                    logger.error(f"Erreur API Prometheus: {data}")
                    return False
            else:
                logger.error(f"HTTP Error: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Erreur connexion: {str(e)}")
            return False
    
    def check_data_retention(self, days_back=35):
        """Verifier periode donnees historiques disponibles"""
        logger.info("Verification retention donnees")
        
        query_metric = 'node_cpu_seconds_total'
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        params = {
            'query': query_metric,
            'start': start_time.timestamp(),
            'end': end_time.timestamp(),
            'step': '1h'
        }
        
        try:
            response = self.session.get(f"{self.api_url}/query_range", params=params)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success' and data['data']['result']:
                    timestamps = []
                    for result in data['data']['result']:
                        if result['values']:
                            for timestamp, value in result['values']:
                                timestamps.append(float(timestamp))
                    
                    if timestamps:
                        first_date = datetime.fromtimestamp(min(timestamps))
                        last_date = datetime.fromtimestamp(max(timestamps))
                        retention_days = (last_date - first_date).days
                        
                        logger.info(f"Periode donnees: {first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}")
                        logger.info(f"Retention: {retention_days} jours")
                        
                        return retention_days
                    else:
                        logger.error("Aucun timestamp trouve")
                        return 0
                else:
                    logger.error("Pas de donnees trouvees")
                    return 0
            else:
                logger.error(f"HTTP Error: {response.status_code}")
                return 0
        except Exception as e:
            logger.error(f"Erreur requete: {str(e)}")
            return 0
    
    def check_critical_metrics(self):
        """Verifier disponibilite metriques critiques"""
        logger.info("Verification metriques critiques")
        
        critical_metrics = [
            'node_cpu_seconds_total',
            'node_memory_MemAvailable_bytes',
            'node_memory_MemTotal_bytes', 
            'node_load1',
            'container_cpu_usage_seconds_total',
            'container_memory_usage_bytes',
            'kube_pod_info'
        ]
        
        available_metrics = []
        missing_metrics = []
        
        for metric_name in critical_metrics:
            try:
                response = self.session.get(f"{self.api_url}/query?query={metric_name}")
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 'success' and data['data']['result']:
                        series_count = len(data['data']['result'])
                        available_metrics.append(metric_name)
                        logger.info(f"Metrique disponible: {metric_name} ({series_count} series)")
                    else:
                        missing_metrics.append(metric_name)
                        logger.warning(f"Metrique manquante: {metric_name}")
                else:
                    missing_metrics.append(metric_name)
                    logger.error(f"Erreur HTTP pour {metric_name}: {response.status_code}")
            except Exception as e:
                missing_metrics.append(metric_name)
                logger.error(f"Exception pour {metric_name}: {str(e)}")
        
        coverage = len(available_metrics) / len(critical_metrics) * 100
        logger.info(f"Couverture metriques: {coverage:.1f}%")
        
        return coverage >= 80, available_metrics, missing_metrics
    
    def validate_data_sources(self):
        """Validation complete des sources de donnees"""
        logger.info("Debut validation sources donnees ML-Scheduler")
        
        # Test connexion
        connection_ok = self.test_connection()
        
        # Test retention
        retention_days = self.check_data_retention()
        
        # Test metriques
        metrics_ok, available_metrics, missing_metrics = self.check_critical_metrics()
        
        # Compilation resultats
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'prometheus_url': self.prometheus_url,
            'connection_status': connection_ok,
            'retention_days': retention_days,
            'retention_sufficient': retention_days >= 7,
            'metrics_coverage_ok': metrics_ok,
            'available_metrics': available_metrics,
            'missing_metrics': missing_metrics,
            'validation_passed': connection_ok and retention_days >= 7 and metrics_ok
        }
        
        # Log resultats
        if validation_results['validation_passed']:
            logger.info("VALIDATION REUSSIE - Sources donnees pretes pour ML")
        else:
            logger.error("VALIDATION ECHOUEE - Problemes identifies")
            
        return validation_results

if __name__ == "__main__":
    collector = PrometheusCollector()
    validation_results = collector.validate_data_sources()
    
    # Sauvegarde resultats
    os.makedirs('../../configs/data_collection', exist_ok=True)
    with open('../../configs/data_collection/prometheus_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
        
    logger.info("Validation terminee - Resultats sauvegardes")