#!/usr/bin/env python3
"""
ML-Scheduler Data Collection - Etape 3
Collecte donnees historiques Prometheus pour algorithmes ML
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import json
warnings.filterwarnings('ignore')

# Configuration Prometheus  
PROMETHEUS_URL = "http://10.110.190.83:9090"
API_URL = f"{PROMETHEUS_URL}/api/v1"

print("="*60)
print("ETAPE 3 - COLLECTE DONNEES HISTORIQUES ML-SCHEDULER")
print("="*60)

def test_prometheus_connection():
    """Test connexion basique Prometheus"""
    print("\n1. TEST CONNEXION PROMETHEUS")
    print("-" * 30)
    
    try:
        response = requests.get(f"{API_URL}/query?query=up", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success':
                services_up_count = len(data['data']['result'])
                print(f"SUCCESS: Connexion Prometheus etablie")
                print(f"SUCCESS: Services UP detectes: {services_up_count}")
                return True
            else:
                print(f"ERROR: Erreur API Prometheus: {data}")
                return False
        else:
            print(f"ERROR: HTTP Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Erreur connexion: {str(e)}")
        return False

def check_data_retention():
    """Verifier periode donnees historiques disponibles"""
    print("\n2. VERIFICATION RETENTION DONNEES")
    print("-" * 35)
    
    query_metric = 'node_cpu_seconds_total'
    end_time = datetime.now()
    start_time = end_time - timedelta(days=35)  # Test 35 jours
    
    params = {
        'query': query_metric,
        'start': start_time.timestamp(),
        'end': end_time.timestamp(), 
        'step': '1h'
    }
    
    try:
        response = requests.get(f"{API_URL}/query_range", params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'success' and data['data']['result']:
                # Calculer periode disponible
                all_timestamps = []
                for result in data['data']['result']:
                    if result['values']:
                        for timestamp, value in result['values']:
                            all_timestamps.append(float(timestamp))
                
                if all_timestamps:
                    first_timestamp = min(all_timestamps)
                    last_timestamp = max(all_timestamps)
                    
                    first_date = datetime.fromtimestamp(first_timestamp)
                    last_date = datetime.fromtimestamp(last_timestamp)
                    retention_days = (last_date - first_date).days
                    
                    print(f"SUCCESS: Premiere donnee: {first_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"SUCCESS: Derniere donnee: {last_date.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"SUCCESS: Periode disponible: {retention_days} jours")
                    
                    if retention_days >= 7:
                        print(f"SUCCESS: Retention suffisante pour ML (>= 7 jours)")
                        return retention_days
                    else:
                        print(f"WARNING: Retention limitee ({retention_days} jours)")
                        return retention_days
                else:
                    print("ERROR: Aucun timestamp trouve")
                    return 0
            else:
                print("ERROR: Pas de donnees trouvees")
                return 0
        else:
            print(f"ERROR: HTTP Error: {response.status_code}")
            return 0
    except Exception as e:
        print(f"ERROR: Erreur requete: {str(e)}")
        return 0

def check_critical_metrics():
    """Verifier disponibilite metriques critiques"""
    print("\n3. VERIFICATION METRIQUES CRITIQUES")
    print("-" * 38)
    
    critical_metrics_list = [
        'node_cpu_seconds_total',
        'node_memory_MemAvailable_bytes', 
        'node_memory_MemTotal_bytes',
        'node_load1',
        'container_cpu_usage_seconds_total',
        'container_memory_usage_bytes',
        'kube_pod_info'
    ]
    
    available_metrics_list = []
    missing_metrics_list = []
    
    for metric_name in critical_metrics_list:
        try:
            response = requests.get(f"{API_URL}/query?query={metric_name}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data['status'] == 'success' and data['data']['result']:
                    series_count = len(data['data']['result'])
                    available_metrics_list.append(metric_name)
                    print(f"SUCCESS: {metric_name}: {series_count} series")
                else:
                    missing_metrics_list.append(metric_name)
                    print(f"MISSING: {metric_name}: Pas de donnees")
            else:
                missing_metrics_list.append(metric_name)
                print(f"ERROR: {metric_name}: Erreur HTTP {response.status_code}")
        except Exception as e:
            missing_metrics_list.append(metric_name)
            print(f"ERROR: {metric_name}: Exception {str(e)[:50]}")
    
    coverage_percentage = len(available_metrics_list) / len(critical_metrics_list) * 100
    print(f"\nCOUVERTURE METRIQUES: {coverage_percentage:.1f}% ({len(available_metrics_list)}/{len(critical_metrics_list)})")
    
    return coverage_percentage >= 80, available_metrics_list, missing_metrics_list

if __name__ == "__main__":
    # Test connexion
    prometheus_connection_ok = test_prometheus_connection()
    
    # Test retention 
    retention_days_available = check_data_retention()
    
    # Test metriques critiques
    metrics_coverage_ok, available_metrics_list, missing_metrics_list = check_critical_metrics()
    
    # Resultat global
    print("\n" + "="*60)
    print("RESULTAT VALIDATION SOURCES DONNEES")
    print("="*60)
    
    validation_results = {
        "Connexion Prometheus": prometheus_connection_ok,
        "Donnees >= 7 jours": retention_days_available >= 7,
        "Metriques critiques >= 80%": metrics_coverage_ok
    }
    
    all_validations_passed = all(validation_results.values())
    
    for test_name, test_status in validation_results.items():
        status_symbol = "SUCCESS" if test_status else "FAILED"
        print(f"{status_symbol}: {test_name}")
    
    if all_validations_passed:
        print(f"\nVALIDATION REUSSIE - Donnees disponibles pour ML")
        collection_period_days = min(retention_days_available, 30)
        print(f"Periode exploitable: {collection_period_days} jours")
        estimated_volume = len(available_metrics_list) * collection_period_days * 24 * 60
        print(f"Volume estime: {len(available_metrics_list)} metriques x {estimated_volume} points")
        
        # Sauvegarder config validation
        validation_config = {
            'validation_date': datetime.now().isoformat(),
            'prometheus_url': PROMETHEUS_URL,
            'retention_days': retention_days_available,
            'collection_days': collection_period_days,
            'available_metrics': available_metrics_list,
            'missing_metrics': missing_metrics_list,
            'validation_passed': all_validations_passed
        }
        
        os.makedirs('/tmp', exist_ok=True)
        with open('/tmp/prometheus_validation.json', 'w') as f:
            json.dump(validation_config, f, indent=2)
        
        print(f"Configuration sauvee: /tmp/prometheus_validation.json")
        
    else:
        print(f"\nVALIDATION ECHOUEE - Problemes a resoudre")
        if not prometheus_connection_ok:
            print("   -> Verifier connectivite Prometheus")
        if retention_days_available < 7:
            print("   -> Periode de donnees insuffisante")
        if not metrics_coverage_ok:
            print("   -> Metriques critiques manquantes")