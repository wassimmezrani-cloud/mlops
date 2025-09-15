#!/usr/bin/env python3
"""
Isolation Forest Advanced Training - Le Detective Step 6.2
Enhanced training with historical incident validation
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
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import logging

from isolation_forest_detector import IsolationForestDetector, generate_synthetic_node_data

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedIsolationForestTraining:
    """
    Advanced training pipeline for Isolation Forest Le Detective
    Production-ready model with historical incident validation
    """
    
    def __init__(self):
        """Initialize advanced training pipeline"""
        self.detector = None
        self.hyperparameter_results = {}
        self.validation_results = {}
        self.historical_incidents = []
        
        logger.info("Advanced Isolation Forest Training initialized")
    
    def generate_historical_incidents(self, num_incidents=50) -> List[Dict]:
        """
        Generate synthetic historical incident data for validation
        
        Args:
            num_incidents: Number of historical incidents to generate
            
        Returns:
            List of historical incident scenarios
        """
        incidents = []
        np.random.seed(42)
        
        incident_types = [
            'cpu_exhaustion', 'memory_leak', 'disk_full', 'network_saturation',
            'pod_crash_loop', 'kernel_panic', 'io_bottleneck', 'resource_contention'
        ]
        
        for i in range(num_incidents):
            incident_type = np.random.choice(incident_types)
            severity = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
            
            if incident_type == 'cpu_exhaustion':
                # CPU exhaustion incident
                cpu_pattern = np.concatenate([
                    np.random.normal(0.4, 0.1, 60),  # Normal period
                    np.random.normal(0.98, 0.01, 40)  # CPU exhaustion
                ])
                memory_pattern = np.random.normal(0.6, 0.1, 100)
                load_pattern = np.concatenate([
                    np.random.normal(2.0, 0.5, 60),
                    np.random.normal(8.5, 0.5, 40)  # Very high load
                ])
                
            elif incident_type == 'memory_leak':
                # Memory leak incident
                cpu_pattern = np.random.normal(0.5, 0.15, 100)
                memory_pattern = np.linspace(0.4, 0.99, 100) + np.random.normal(0, 0.01, 100)
                load_pattern = np.linspace(1.5, 4.0, 100) + np.random.normal(0, 0.3, 100)
                
            elif incident_type == 'disk_full':
                # Disk space exhaustion
                cpu_pattern = np.random.normal(0.3, 0.1, 100)
                memory_pattern = np.random.normal(0.7, 0.1, 100)
                load_pattern = np.concatenate([
                    np.random.normal(1.5, 0.5, 80),
                    np.random.normal(6.0, 1.0, 20)  # IO wait spike
                ])
                disk_usage = np.linspace(70, 99.5, 100)
                
            elif incident_type == 'network_saturation':
                # Network bandwidth saturation
                cpu_pattern = np.random.normal(0.6, 0.2, 100)
                memory_pattern = np.random.normal(0.5, 0.1, 100)
                load_pattern = np.random.normal(3.0, 1.0, 100)
                network_multiplier = 10  # 10x normal network traffic
                
            else:
                # Generic resource contention
                cpu_pattern = np.random.normal(0.8, 0.1, 100)
                memory_pattern = np.random.normal(0.85, 0.05, 100)
                load_pattern = np.random.normal(5.0, 1.5, 100)
            
            # Common incident characteristics
            if 'disk_usage' not in locals():
                disk_usage = np.random.normal(60, 20, 100)
            if 'network_multiplier' not in locals():
                network_multiplier = 1
            
            incident = {
                'incident_id': f'incident_{i:03d}',
                'incident_type': incident_type,
                'severity': severity,
                'node_id': f'troubled_node_{i}',
                'cpu_utilization': np.clip(cpu_pattern, 0, 1).tolist(),
                'memory_utilization': np.clip(memory_pattern, 0, 1).tolist(),
                'load_average': np.clip(load_pattern, 0, 16).tolist(),
                'pod_count': np.random.poisson(25, 100).tolist(),
                'network_bytes_in': (np.random.lognormal(10, 1, 100) * network_multiplier).tolist(),
                'network_bytes_out': (np.random.lognormal(9, 1, 100) * network_multiplier).tolist(),
                'disk_usage_percent': np.clip(disk_usage, 0, 100).tolist(),
                'container_restarts': np.random.poisson(8) if incident_type == 'pod_crash_loop' else np.random.poisson(2),
                'uptime_hours': np.random.uniform(10, 100),  # Usually shorter uptime for incidents
                'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(100)],
                'ground_truth_anomaly': True,  # All incidents should be detected as anomalies
                'impact_score': {'low': 30, 'medium': 60, 'high': 90}[severity]
            }
            incidents.append(incident)
            
            # Reset variables for next iteration
            if 'disk_usage' in locals():
                del disk_usage
            if 'network_multiplier' in locals():
                del network_multiplier
        
        logger.info(f"Generated {len(incidents)} historical incidents for validation")
        return incidents
    
    def hyperparameter_optimization(self, training_data: List[Dict]) -> Dict[str, Any]:
        """
        Optimize Isolation Forest hyperparameters using grid search
        
        Args:
            training_data: Training dataset
            
        Returns:
            Best hyperparameters and optimization results
        """
        logger.info("Starting hyperparameter optimization")
        
        # Prepare feature matrix
        temp_detector = IsolationForestDetector()
        X, feature_names = temp_detector.prepare_training_data(training_data)
        X_scaled = temp_detector.scaler.fit_transform(X)
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'contamination': [0.08, 0.1, 0.12],
            'max_samples': [0.8, 0.9, 1.0],
            'max_features': [0.8, 0.9, 1.0]
        }
        
        # Custom scoring function for unsupervised learning
        def anomaly_score_variance(estimator, X):
            """Score based on variance of anomaly scores - higher variance is better"""
            scores = estimator.decision_function(X)
            return np.std(scores)
        
        # Grid search
        isolation_forest = IsolationForest(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            isolation_forest,
            param_grid,
            scoring=anomaly_score_variance,
            cv=3,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'all_results': grid_search.cv_results_
        }
    
    def train_production_model(self, training_data: List[Dict], 
                             best_params: Dict[str, Any]) -> IsolationForestDetector:
        """
        Train production model with optimized parameters
        
        Args:
            training_data: Training dataset
            best_params: Optimized hyperparameters
            
        Returns:
            Trained production detector
        """
        logger.info("Training production model with optimized parameters")
        
        # Initialize detector with best parameters
        detector = IsolationForestDetector(
            contamination=best_params.get('contamination', 0.1),
            random_state=42
        )
        
        # Update model with best parameters
        detector.model = IsolationForest(
            n_estimators=best_params.get('n_estimators', 200),
            contamination=best_params.get('contamination', 0.1),
            max_samples=best_params.get('max_samples', 0.9),
            max_features=best_params.get('max_features', 0.9),
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        training_results = detector.train_anomaly_detector(training_data)
        
        logger.info("Production model training completed")
        return detector
    
    def validate_with_historical_incidents(self, detector: IsolationForestDetector, 
                                         incidents: List[Dict]) -> Dict[str, Any]:
        """
        Validate model performance against historical incidents
        
        Args:
            detector: Trained detector model
            incidents: Historical incident data
            
        Returns:
            Validation results and metrics
        """
        logger.info(f"Validating against {len(incidents)} historical incidents")
        
        predictions = []
        ground_truth = []
        confidence_scores = []
        anomaly_scores = []
        
        detected_incidents = 0
        missed_incidents = 0
        
        incident_detection_details = []
        
        for incident in incidents:
            result = detector.detect_anomalies(incident)
            
            predictions.append(result['is_anomaly'])
            ground_truth.append(incident['ground_truth_anomaly'])
            confidence_scores.append(result['confidence'])
            anomaly_scores.append(result['anomaly_score'])
            
            if result['is_anomaly'] and incident['ground_truth_anomaly']:
                detected_incidents += 1
                status = "DETECTED"
            elif not result['is_anomaly'] and incident['ground_truth_anomaly']:
                missed_incidents += 1
                status = "MISSED"
            else:
                status = "CORRECT_NORMAL"
            
            detection_detail = {
                'incident_id': incident['incident_id'],
                'incident_type': incident['incident_type'],
                'severity': incident['severity'],
                'predicted_anomaly': result['is_anomaly'],
                'actual_anomaly': incident['ground_truth_anomaly'],
                'anomaly_score': result['anomaly_score'],
                'confidence': result['confidence'],
                'status': status,
                'contributing_factors': result['contributing_factors']
            }
            incident_detection_details.append(detection_detail)
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, predictions, average='binary', pos_label=True
        )
        
        detection_rate = detected_incidents / len(incidents)
        false_negative_rate = missed_incidents / len(incidents)
        
        # Severity-based analysis
        severity_analysis = {}
        for severity in ['low', 'medium', 'high']:
            severity_incidents = [d for d in incident_detection_details if d['severity'] == severity]
            if severity_incidents:
                detected = sum(1 for d in severity_incidents if d['status'] == 'DETECTED')
                total = len(severity_incidents)
                severity_analysis[severity] = {
                    'total': total,
                    'detected': detected,
                    'detection_rate': detected / total
                }
        
        validation_results = {
            'total_incidents': len(incidents),
            'detected_incidents': detected_incidents,
            'missed_incidents': missed_incidents,
            'detection_rate': detection_rate,
            'false_negative_rate': false_negative_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_confidence': np.mean(confidence_scores),
            'mean_anomaly_score': np.mean(anomaly_scores),
            'severity_analysis': severity_analysis,
            'incident_details': incident_detection_details
        }
        
        logger.info(f"Incident detection rate: {detection_rate:.1%}")
        logger.info(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        
        return validation_results
    
    def calculate_business_score(self, validation_results: Dict[str, Any]) -> float:
        """
        Calculate business score based on performance metrics
        
        Args:
            validation_results: Model validation results
            
        Returns:
            Business score (0-100)
        """
        # Business scoring weights
        weights = {
            'detection_rate': 0.35,      # 35% - Most important: catch incidents
            'precision': 0.25,           # 25% - Important: avoid false positives  
            'f1_score': 0.20,           # 20% - Balance precision/recall
            'high_severity_detection': 0.20  # 20% - Critical incidents detection
        }
        
        scores = {}
        
        # Detection rate score (target: ≥90%)
        detection_rate = validation_results['detection_rate']
        scores['detection_rate'] = min(100, (detection_rate / 0.9) * 100)
        
        # Precision score (target: ≥85%)
        precision = validation_results['precision']
        scores['precision'] = min(100, (precision / 0.85) * 100)
        
        # F1 score (target: ≥80%)
        f1_score = validation_results['f1_score']
        scores['f1_score'] = min(100, (f1_score / 0.8) * 100)
        
        # High severity detection (target: ≥95%)
        high_severity_analysis = validation_results['severity_analysis'].get('high', {'detection_rate': 0})
        high_severity_detection = high_severity_analysis['detection_rate']
        scores['high_severity_detection'] = min(100, (high_severity_detection / 0.95) * 100)
        
        # Calculate weighted business score
        business_score = sum(scores[metric] * weights[metric] for metric in weights.keys())
        
        logger.info(f"Business Score Components:")
        for metric, score in scores.items():
            logger.info(f"  {metric}: {score:.1f}/100 (weight: {weights[metric]:.1%})")
        
        return business_score
    
    def save_production_model(self, detector: IsolationForestDetector, 
                            validation_results: Dict[str, Any],
                            business_score: float) -> str:
        """
        Save production model and metadata
        
        Args:
            detector: Trained detector
            validation_results: Validation results
            business_score: Business performance score
            
        Returns:
            Model save path
        """
        model_dir = "./models/isolation_detector"
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = f"{model_dir}/isolation_forest_model.pkl"
        joblib.dump({
            'model': detector.model,
            'scaler': detector.scaler,
            'feature_names': detector.feature_names,
            'contamination': detector.contamination,
            'anomaly_threshold': detector.anomaly_threshold
        }, model_path)
        
        # Save metadata
        metadata = {
            'model_name': 'isolation_forest_detector',
            'model_version': '1.0.0',
            'expert_name': 'Le Detective',
            'algorithm': 'Isolation Forest',
            'training_timestamp': datetime.now().isoformat(),
            'business_score': business_score,
            'validation_metrics': {
                'detection_rate': validation_results['detection_rate'],
                'precision': validation_results['precision'],
                'recall': validation_results['recall'],
                'f1_score': validation_results['f1_score']
            },
            'production_ready': business_score >= 75.0,
            'performance_targets': {
                'detection_rate': '≥90%',
                'precision': '≥85%', 
                'false_positive_rate': '≤10%'
            }
        }
        
        with open(f"{model_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save detailed results
        with open(f"{model_dir}/validation_results.json", 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_results = json.loads(json.dumps(validation_results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Production model saved to {model_path}")
        return model_path
    
    def run_complete_training_pipeline(self) -> Dict[str, Any]:
        """
        Run complete advanced training pipeline
        
        Returns:
            Complete training and validation results
        """
        logger.info("Starting complete Isolation Forest training pipeline")
        
        # Step 1: Generate training data
        logger.info("Step 1: Generating enhanced training data")
        training_data = generate_synthetic_node_data(num_normal=1200, num_anomalous=150)
        
        # Step 2: Hyperparameter optimization
        logger.info("Step 2: Hyperparameter optimization")
        optimization_results = self.hyperparameter_optimization(training_data)
        best_params = optimization_results['best_parameters']
        
        # Step 3: Train production model
        logger.info("Step 3: Training production model")
        self.detector = self.train_production_model(training_data, best_params)
        
        # Step 4: Generate historical incidents for validation
        logger.info("Step 4: Generating historical incidents")
        self.historical_incidents = self.generate_historical_incidents(num_incidents=80)
        
        # Step 5: Validate against historical incidents
        logger.info("Step 5: Validating against historical incidents")
        self.validation_results = self.validate_with_historical_incidents(
            self.detector, self.historical_incidents
        )
        
        # Step 6: Calculate business score
        logger.info("Step 6: Calculating business performance score")
        business_score = self.calculate_business_score(self.validation_results)
        
        # Step 7: Save production model
        logger.info("Step 7: Saving production model")
        model_path = self.save_production_model(
            self.detector, self.validation_results, business_score
        )
        
        # Complete results
        complete_results = {
            'training_samples': len(training_data),
            'optimization_results': optimization_results,
            'validation_results': self.validation_results,
            'business_score': business_score,
            'model_path': model_path,
            'production_ready': business_score >= 75.0
        }
        
        return complete_results

def main():
    """
    Main execution function for advanced training
    """
    print("="*70)
    print("ISOLATION FOREST ADVANCED TRAINING - LE DETECTIVE")
    print("Step 6.2: Training Isolation Forest et Détection Anomalies")
    print("="*70)
    
    # Initialize training pipeline
    training_pipeline = AdvancedIsolationForestTraining()
    
    # Run complete training
    results = training_pipeline.run_complete_training_pipeline()
    
    # Display final results
    print("\n" + "="*70)
    print("FINAL TRAINING RESULTS")
    print("="*70)
    print(f"Training samples: {results['training_samples']}")
    print(f"Business score: {results['business_score']:.1f}/100")
    print(f"Production ready: {'YES' if results['production_ready'] else 'NO'}")
    
    print("\nDetection Performance:")
    validation = results['validation_results']
    print(f"  Incident detection rate: {validation['detection_rate']:.1%}")
    print(f"  Precision: {validation['precision']:.3f}")
    print(f"  Recall: {validation['recall']:.3f}")
    print(f"  F1-Score: {validation['f1_score']:.3f}")
    
    print("\nSeverity Analysis:")
    for severity, analysis in validation['severity_analysis'].items():
        print(f"  {severity.capitalize()}: {analysis['detected']}/{analysis['total']} "
              f"({analysis['detection_rate']:.1%})")
    
    print(f"\nModel saved to: {results['model_path']}")
    
    if results['business_score'] >= 85.0:
        print("\n🎯 EXCELLENT - Le Detective ready for production deployment!")
    elif results['business_score'] >= 75.0:
        print("\n✅ GOOD - Le Detective meets production requirements!")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT - Additional tuning recommended")
    
    logger.info("Step 6.2 completed successfully - Advanced training finished")

if __name__ == "__main__":
    main()