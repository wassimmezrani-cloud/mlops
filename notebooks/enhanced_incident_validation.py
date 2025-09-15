#!/usr/bin/env python3
"""
Enhanced Incident Validation - Le Detective
Re-validation with enhanced sensitivity model
"""

import os
import json
import numpy as np
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

from isolation_forest_enhanced_model import EnhancedIsolationForestDetector

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_enhanced_model(model_path: str) -> EnhancedIsolationForestDetector:
    """Load enhanced model"""
    logger.info(f"Loading enhanced model from {model_path}")
    
    artifacts = joblib.load(model_path)
    
    detector = EnhancedIsolationForestDetector()
    detector.model = artifacts['model']
    detector.scaler = artifacts['scaler']
    detector.feature_names = artifacts['feature_names']
    detector.contamination = artifacts['contamination']
    detector.anomaly_threshold = artifacts['anomaly_threshold']
    detector.is_trained = True
    
    return detector

def generate_realistic_incidents() -> List[Dict]:
    """Generate realistic incident scenarios"""
    incidents = []
    
    # Memory leak gradual
    incidents.append({
        'incident_id': 'memory_leak_001',
        'scenario': 'memory_leak_gradual',
        'node_id': 'worker-troubled-001',
        'cpu_utilization': ([0.3] * 60 + [0.4 + i*0.01 for i in range(60)]),
        'memory_utilization': ([0.4] * 30 + [0.4 + i*0.008 for i in range(90)]),
        'load_average': ([1.5] * 60 + [1.5 + i*0.05 for i in range(60)]),
        'pod_count': [15] * 120,
        'network_bytes_in': [1e9] * 120,
        'network_bytes_out': [8e8] * 120,
        'disk_usage_percent': [40] * 120,
        'container_restarts': 2,
        'uptime_hours': 72,
        'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(120)],
        'ground_truth': True
    })
    
    # CPU spike sudden
    incidents.append({
        'incident_id': 'cpu_spike_001',
        'scenario': 'cpu_spike_sudden',
        'node_id': 'worker-troubled-002',
        'cpu_utilization': ([0.3] * 60 + [0.98] * 15 + [0.3] * 45),
        'memory_utilization': [0.5] * 120,
        'load_average': ([1.5] * 60 + [8.0] * 15 + [1.5] * 45),
        'pod_count': [15] * 120,
        'network_bytes_in': [1e9] * 120,
        'network_bytes_out': [8e8] * 120,
        'disk_usage_percent': [40] * 120,
        'container_restarts': 1,
        'uptime_hours': 120,
        'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(120)],
        'ground_truth': True
    })
    
    # Disk I/O storm
    incidents.append({
        'incident_id': 'disk_io_001',
        'scenario': 'disk_io_storm',
        'node_id': 'worker-troubled-003',
        'cpu_utilization': [0.4] * 120,
        'memory_utilization': [0.7] * 120,
        'load_average': ([2.0] * 30 + [8.5] * 45 + [2.0] * 45),
        'pod_count': [15] * 120,
        'network_bytes_in': [1e9] * 120,
        'network_bytes_out': [8e8] * 120,
        'disk_usage_percent': ([40] * 30 + [85] * 45 + [40] * 45),
        'container_restarts': 1,
        'uptime_hours': 96,
        'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(120)],
        'ground_truth': True
    })
    
    # Network saturation
    incidents.append({
        'incident_id': 'network_sat_001',
        'scenario': 'network_saturation',
        'node_id': 'worker-troubled-004',
        'cpu_utilization': ([0.3] * 30 + [0.7] * 30 + [0.3] * 60),
        'memory_utilization': [0.5] * 120,
        'load_average': ([1.5] * 30 + [4.0] * 30 + [1.5] * 60),
        'pod_count': [15] * 120,
        'network_bytes_in': ([1e9] * 30 + [2e10] * 30 + [1e9] * 60),
        'network_bytes_out': ([8e8] * 30 + [1.5e10] * 30 + [8e8] * 60),
        'disk_usage_percent': [40] * 120,
        'container_restarts': 1,
        'uptime_hours': 144,
        'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(120)],
        'ground_truth': True
    })
    
    # Pod crash cascade
    incidents.append({
        'incident_id': 'pod_crash_001',
        'scenario': 'pod_crash_cascade',
        'node_id': 'worker-troubled-005',
        'cpu_utilization': ([0.4] * 30 + [0.8, 0.2, 0.9, 0.1] * 15 + [0.4] * 30),
        'memory_utilization': ([0.5] * 30 + [0.8, 0.3, 0.9, 0.2] * 15 + [0.5] * 30),
        'load_average': ([2.0] * 30 + [6.0] * 60 + [2.0] * 30),
        'pod_count': ([15] * 30 + [8, 12, 6, 14] * 15 + [15] * 30),
        'network_bytes_in': [1e9] * 120,
        'network_bytes_out': [8e8] * 120,
        'disk_usage_percent': [40] * 120,
        'container_restarts': 25,
        'uptime_hours': 48,
        'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(120)],
        'ground_truth': True
    })
    
    # Resource exhaustion combo
    incidents.append({
        'incident_id': 'resource_combo_001',
        'scenario': 'resource_exhaustion_combo',
        'node_id': 'worker-troubled-006',
        'cpu_utilization': ([0.4] * 30 + [0.95] * 90),
        'memory_utilization': ([0.5] * 30 + [0.95] * 90),
        'load_average': ([2.0] * 30 + [12.0] * 90),
        'pod_count': [15] * 120,
        'network_bytes_in': [1e9] * 120,
        'network_bytes_out': [8e8] * 120,
        'disk_usage_percent': ([40] * 30 + [90 + i*0.1 for i in range(90)]),
        'container_restarts': 5,
        'uptime_hours': 24,
        'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(120)],
        'ground_truth': True
    })
    
    return incidents

def run_enhanced_validation():
    """Run validation with enhanced model"""
    print("="*70)
    print("ENHANCED INCIDENT VALIDATION - LE DETECTIVE")
    print("Re-validation with Enhanced Sensitivity Model")
    print("="*70)
    
    # Load enhanced model
    model_path = "./models/isolation_detector/isolation_forest_enhanced.pkl"
    detector = load_enhanced_model(model_path)
    
    # Generate incidents
    incidents = generate_realistic_incidents()
    
    print(f"\nTesting {len(incidents)} realistic incident scenarios...")
    
    results = []
    detected_count = 0
    
    for incident in incidents:
        print(f"\nTesting: {incident['scenario']}")
        
        # Detect anomalies
        detection_result = detector.detect_anomalies(incident)
        
        detected = detection_result['is_anomaly']
        if detected:
            detected_count += 1
        
        status = "✅ DETECTED" if detected else "❌ MISSED"
        print(f"  {status}")
        print(f"  Anomaly Score: {detection_result['anomaly_score']:.4f}")
        print(f"  Confidence: {detection_result['confidence']:.3f}")
        print(f"  Risk Level: {detection_result['risk_level']}")
        print(f"  Method: {detection_result['detection_method']}")
        
        if detection_result['contributing_factors']:
            print("  Contributing factors:")
            for factor in detection_result['contributing_factors'][:3]:
                print(f"    • {factor}")
        
        results.append({
            'scenario': incident['scenario'],
            'detected': detected,
            'score': detection_result['anomaly_score'],
            'confidence': detection_result['confidence'],
            'risk_level': detection_result['risk_level'],
            'method': detection_result['detection_method']
        })
    
    # Calculate metrics
    detection_rate = detected_count / len(incidents)
    
    print("\n" + "="*50)
    print("ENHANCED VALIDATION RESULTS")
    print("="*50)
    print(f"Total scenarios: {len(incidents)}")
    print(f"Detected scenarios: {detected_count}")
    print(f"Detection rate: {detection_rate:.1%}")
    
    # Business score calculation
    business_score = min(100, (detection_rate / 0.85) * 100)
    
    print(f"Business score: {business_score:.1f}/100")
    
    performance_level = "EXCELLENT" if business_score >= 90 else \
                       "GOOD" if business_score >= 80 else \
                       "ACCEPTABLE" if business_score >= 70 else "NEEDS_IMPROVEMENT"
    
    print(f"Performance level: {performance_level}")
    
    # Save enhanced results
    enhanced_results = {
        'validation_type': 'Enhanced Incident Validation',
        'model_path': model_path,
        'validation_date': datetime.now().isoformat(),
        'total_scenarios': len(incidents),
        'detected_scenarios': detected_count,
        'detection_rate': detection_rate,
        'business_score': business_score,
        'performance_level': performance_level,
        'scenario_results': results
    }
    
    results_path = "./models/isolation_detector/enhanced_validation_results.json"
    with open(results_path, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    print(f"\nResults saved: {results_path}")
    
    if business_score >= 80:
        print("\n🎯 SUCCESS - Enhanced Le Detective ready for production!")
        return True
    else:
        print("\n⚠️  Still needs improvement")
        return False

if __name__ == "__main__":
    success = run_enhanced_validation()
    logger.info(f"Enhanced validation completed - Success: {success}")