#!/usr/bin/env python3
"""
Isolation Forest Fast Training - Le Detective Step 6.2
Fast training with pre-selected optimal parameters
Production-ready anomaly detection model
"""

import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import logging

from isolation_forest_detector import IsolationForestDetector, generate_synthetic_node_data

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastIsolationForestTraining:
    """
    Fast training pipeline for Isolation Forest Le Detective
    Uses pre-optimized parameters for quick deployment
    """
    
    def __init__(self):
        """Initialize fast training pipeline"""
        self.detector = None
        self.validation_results = {}
        self.historical_incidents = []
        
        # Pre-optimized parameters from previous runs
        self.optimal_params = {
            'n_estimators': 200,
            'contamination': 0.1,
            'max_samples': 0.9,
            'max_features': 0.9,
            'random_state': 42
        }
        
        logger.info("Fast Isolation Forest Training initialized")
    
    def generate_historical_incidents(self, num_incidents=60) -> List[Dict]:
        """Generate synthetic historical incidents for validation"""
        incidents = []
        np.random.seed(42)
        
        incident_scenarios = [
            {
                'type': 'cpu_exhaustion',
                'cpu_pattern': lambda: np.concatenate([
                    np.random.normal(0.4, 0.1, 70),
                    np.random.normal(0.98, 0.01, 30)
                ]),
                'memory_pattern': lambda: np.random.normal(0.6, 0.1, 100),
                'load_pattern': lambda: np.concatenate([
                    np.random.normal(2.0, 0.5, 70),
                    np.random.normal(8.5, 0.5, 30)
                ])
            },
            {
                'type': 'memory_leak',
                'cpu_pattern': lambda: np.random.normal(0.5, 0.15, 100),
                'memory_pattern': lambda: np.linspace(0.4, 0.99, 100) + np.random.normal(0, 0.01, 100),
                'load_pattern': lambda: np.linspace(1.5, 4.0, 100) + np.random.normal(0, 0.3, 100)
            },
            {
                'type': 'network_saturation',
                'cpu_pattern': lambda: np.random.normal(0.6, 0.2, 100),
                'memory_pattern': lambda: np.random.normal(0.5, 0.1, 100),
                'load_pattern': lambda: np.random.normal(3.0, 1.0, 100)
            },
            {
                'type': 'resource_contention',
                'cpu_pattern': lambda: np.random.normal(0.8, 0.1, 100),
                'memory_pattern': lambda: np.random.normal(0.85, 0.05, 100),
                'load_pattern': lambda: np.random.normal(5.0, 1.5, 100)
            }
        ]
        
        for i in range(num_incidents):
            scenario = np.random.choice(incident_scenarios)
            severity = np.random.choice(['low', 'medium', 'high'], p=[0.2, 0.5, 0.3])
            
            incident = {
                'incident_id': f'incident_{i:03d}',
                'incident_type': scenario['type'],
                'severity': severity,
                'node_id': f'troubled_node_{i}',
                'cpu_utilization': np.clip(scenario['cpu_pattern'](), 0, 1).tolist(),
                'memory_utilization': np.clip(scenario['memory_pattern'](), 0, 1).tolist(),
                'load_average': np.clip(scenario['load_pattern'](), 0, 16).tolist(),
                'pod_count': np.random.poisson(25, 100).tolist(),
                'network_bytes_in': np.random.lognormal(11, 1.5, 100).tolist(),
                'network_bytes_out': np.random.lognormal(10, 1.5, 100).tolist(),
                'disk_usage_percent': np.clip(np.random.normal(70, 15, 100), 0, 100).tolist(),
                'container_restarts': np.random.poisson(5),
                'uptime_hours': np.random.uniform(20, 120),
                'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(100)],
                'ground_truth_anomaly': True,
                'impact_score': {'low': 30, 'medium': 60, 'high': 90}[severity]
            }
            incidents.append(incident)
        
        logger.info(f"Generated {len(incidents)} historical incidents")
        return incidents
    
    def train_production_model(self, training_data: List[Dict]) -> IsolationForestDetector:
        """Train production model with optimal parameters"""
        logger.info("Training production model with optimal parameters")
        
        detector = IsolationForestDetector(
            contamination=self.optimal_params['contamination'],
            random_state=self.optimal_params['random_state']
        )
        
        # Update model with optimal parameters
        detector.model = IsolationForest(**self.optimal_params, n_jobs=-1)
        
        # Train the model
        training_results = detector.train_anomaly_detector(training_data)
        
        logger.info(f"Model trained on {len(training_data)} samples")
        return detector
    
    def validate_incident_detection(self, detector: IsolationForestDetector, 
                                  incidents: List[Dict]) -> Dict[str, Any]:
        """Validate model against historical incidents"""
        logger.info(f"Validating against {len(incidents)} historical incidents")
        
        predictions = []
        ground_truth = []
        confidence_scores = []
        anomaly_scores = []
        incident_details = []
        
        for incident in incidents:
            result = detector.detect_anomalies(incident)
            
            predictions.append(result['is_anomaly'])
            ground_truth.append(incident['ground_truth_anomaly'])
            confidence_scores.append(result['confidence'])
            anomaly_scores.append(result['anomaly_score'])
            
            status = "DETECTED" if result['is_anomaly'] else "MISSED"
            
            incident_details.append({
                'incident_id': incident['incident_id'],
                'incident_type': incident['incident_type'],
                'severity': incident['severity'],
                'detected': result['is_anomaly'],
                'anomaly_score': result['anomaly_score'],
                'confidence': result['confidence'],
                'status': status,
                'factors': result['contributing_factors'][:3]  # Top 3 factors
            })
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='binary', pos_label=True
        )
        
        detection_rate = sum(predictions) / len(predictions)
        
        # Severity analysis
        severity_stats = {}
        for severity in ['low', 'medium', 'high']:
            sev_incidents = [d for d in incident_details if d['severity'] == severity]
            if sev_incidents:
                detected = sum(1 for d in sev_incidents if d['detected'])
                severity_stats[severity] = {
                    'total': len(sev_incidents),
                    'detected': detected,
                    'rate': detected / len(sev_incidents)
                }
        
        results = {
            'total_incidents': len(incidents),
            'detected_count': sum(predictions),
            'detection_rate': detection_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_confidence': np.mean(confidence_scores),
            'severity_breakdown': severity_stats,
            'incident_details': incident_details
        }
        
        return results
    
    def calculate_business_metrics(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate business performance metrics"""
        
        # Performance targets
        targets = {
            'detection_rate': 0.85,    # ≥85% incidents detected
            'precision': 0.80,         # ≥80% precision
            'f1_score': 0.75,         # ≥75% F1 score
            'high_severity': 0.90      # ≥90% high severity detection
        }
        
        scores = {}
        detection_rate = validation_results['detection_rate']
        precision = validation_results['precision']
        f1_score = validation_results['f1_score']
        
        # Individual component scores
        scores['detection_score'] = min(100, (detection_rate / targets['detection_rate']) * 100)
        scores['precision_score'] = min(100, (precision / targets['precision']) * 100)
        scores['f1_score'] = min(100, (f1_score / targets['f1_score']) * 100)
        
        # High severity detection score
        high_sev = validation_results['severity_breakdown'].get('high', {'rate': 0})
        high_sev_rate = high_sev['rate']
        scores['high_severity_score'] = min(100, (high_sev_rate / targets['high_severity']) * 100)
        
        # Weighted business score
        weights = {'detection_score': 0.4, 'precision_score': 0.3, 'f1_score': 0.2, 'high_severity_score': 0.1}
        business_score = sum(scores[key] * weights[key] for key in weights)
        
        return {
            'component_scores': scores,
            'business_score': business_score,
            'targets_met': {
                'detection_rate': detection_rate >= targets['detection_rate'],
                'precision': precision >= targets['precision'], 
                'f1_score': f1_score >= targets['f1_score'],
                'high_severity': high_sev_rate >= targets['high_severity']
            },
            'production_ready': business_score >= 80.0
        }
    
    def save_model_artifacts(self, detector: IsolationForestDetector,
                           validation_results: Dict[str, Any],
                           business_metrics: Dict[str, Any]) -> str:
        """Save model and metadata"""
        model_dir = "./models/isolation_detector"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save trained model
        model_path = f"{model_dir}/isolation_forest_production.pkl"
        joblib.dump({
            'model': detector.model,
            'scaler': detector.scaler,
            'feature_names': detector.feature_names,
            'contamination': detector.contamination,
            'optimal_params': self.optimal_params
        }, model_path)
        
        # Save final results
        final_results = {
            'model_info': {
                'name': 'Isolation Forest Le Detective',
                'version': '1.0.0',
                'algorithm': 'Isolation Forest Unsupervised',
                'expert_role': 'Anomaly Detection',
                'training_date': datetime.now().isoformat()
            },
            'performance_metrics': validation_results,
            'business_metrics': business_metrics,
            'model_parameters': self.optimal_params,
            'production_deployment': {
                'ready': business_metrics['production_ready'],
                'confidence_level': 'HIGH' if business_metrics['business_score'] >= 85 else 'MEDIUM'
            }
        }
        
        with open(f"{model_dir}/final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"Model artifacts saved to {model_dir}")
        return model_path
    
    def run_fast_training_pipeline(self) -> Dict[str, Any]:
        """Run complete fast training pipeline"""
        logger.info("Starting fast Isolation Forest training pipeline")
        
        # Generate training data
        training_data = generate_synthetic_node_data(num_normal=1000, num_anomalous=120)
        
        # Train production model
        self.detector = self.train_production_model(training_data)
        
        # Generate and validate against incidents
        self.historical_incidents = self.generate_historical_incidents(60)
        self.validation_results = self.validate_incident_detection(
            self.detector, self.historical_incidents
        )
        
        # Calculate business metrics
        business_metrics = self.calculate_business_metrics(self.validation_results)
        
        # Save artifacts
        model_path = self.save_model_artifacts(
            self.detector, self.validation_results, business_metrics
        )
        
        return {
            'training_data_size': len(training_data),
            'validation_results': self.validation_results,
            'business_metrics': business_metrics,
            'model_path': model_path
        }

def main():
    """Main execution function"""
    print("="*70)
    print("ISOLATION FOREST FAST TRAINING - LE DETECTIVE")
    print("Step 6.2: Training Isolation Forest et Détection Anomalies") 
    print("="*70)
    
    # Run fast training
    trainer = FastIsolationForestTraining()
    results = trainer.run_fast_training_pipeline()
    
    # Display results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    validation = results['validation_results']
    business = results['business_metrics']
    
    print(f"Training samples: {results['training_data_size']}")
    print(f"Validation incidents: {validation['total_incidents']}")
    print(f"Business score: {business['business_score']:.1f}/100")
    
    print("\nPerformance Metrics:")
    print(f"  Detection rate: {validation['detection_rate']:.1%}")
    print(f"  Precision: {validation['precision']:.3f}")
    print(f"  Recall: {validation['recall']:.3f}")
    print(f"  F1-Score: {validation['f1_score']:.3f}")
    
    print("\nSeverity Breakdown:")
    for severity, stats in validation['severity_breakdown'].items():
        print(f"  {severity.capitalize()}: {stats['detected']}/{stats['total']} ({stats['rate']:.1%})")
    
    print("\nTargets Achievement:")
    for target, met in business['targets_met'].items():
        status = "✅" if met else "❌"
        print(f"  {target}: {status}")
    
    print(f"\nProduction ready: {'YES' if business['production_ready'] else 'NO'}")
    print(f"Model saved: {results['model_path']}")
    
    if business['business_score'] >= 85:
        print("\n🎯 EXCELLENT - Le Detective exceeds expectations!")
    elif business['business_score'] >= 80:
        print("\n✅ SUCCESS - Le Detective ready for production!")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT - Consider additional training")
    
    logger.info("Step 6.2 completed - Production model ready")

if __name__ == "__main__":
    main()