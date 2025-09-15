#!/usr/bin/env python3
"""
Isolation Forest Detector - Le Detective
Step 6: Anomaly Detection for Node Behavioral Analysis
ML-Scheduler Third Expert AI - Unsupervised Learning
"""

import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IsolationForestDetector:
    """
    Le Detective - Isolation Forest pour detection anomalies nodes
    Third expert AI for ML-Scheduler trio architecture
    """
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize Isolation Forest Detector
        
        Args:
            contamination: Expected proportion of anomalies (10%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = RobustScaler()
        self.feature_names = []
        self.anomaly_threshold = -0.1
        self.is_trained = False
        
        logger.info("Isolation Forest Detector initialized")
        logger.info(f"Contamination rate: {contamination}")
    
    def generate_node_behavioral_features(self, node_data: Dict) -> Dict[str, float]:
        """
        Generate comprehensive behavioral features for anomaly detection
        
        Args:
            node_data: Historical node metrics data
            
        Returns:
            Dictionary of engineered features
        """
        features = {}
        
        # Basic resource utilization features
        cpu_util = node_data.get('cpu_utilization', [])
        memory_util = node_data.get('memory_utilization', [])
        load_avg = node_data.get('load_average', [])
        
        if cpu_util:
            # CPU behavior patterns
            features['cpu_mean'] = np.mean(cpu_util)
            features['cpu_std'] = np.std(cpu_util)
            features['cpu_max'] = np.max(cpu_util)
            features['cpu_min'] = np.min(cpu_util)
            features['cpu_volatility'] = np.std(cpu_util) / (np.mean(cpu_util) + 0.001)
            features['cpu_spike_count'] = sum(1 for x in cpu_util if x > 0.9)
            features['cpu_idle_ratio'] = sum(1 for x in cpu_util if x < 0.1) / len(cpu_util)
            
            # CPU trend analysis
            cpu_array = np.array(cpu_util)
            if len(cpu_array) > 1:
                features['cpu_trend'] = np.polyfit(range(len(cpu_array)), cpu_array, 1)[0]
        
        if memory_util:
            # Memory behavior patterns
            features['memory_mean'] = np.mean(memory_util)
            features['memory_std'] = np.std(memory_util)
            features['memory_max'] = np.max(memory_util)
            features['memory_pressure'] = sum(1 for x in memory_util if x > 0.85) / len(memory_util)
            features['memory_volatility'] = np.std(memory_util) / (np.mean(memory_util) + 0.001)
            
            # Memory leak detection
            memory_array = np.array(memory_util)
            if len(memory_array) > 1:
                features['memory_trend'] = np.polyfit(range(len(memory_array)), memory_array, 1)[0]
                features['memory_leak_indicator'] = max(0, features['memory_trend'])
        
        if load_avg:
            # System load patterns
            features['load_mean'] = np.mean(load_avg)
            features['load_std'] = np.std(load_avg)
            features['load_max'] = np.max(load_avg)
            features['load_spike_ratio'] = sum(1 for x in load_avg if x > 4.0) / len(load_avg)
        
        # Pod scheduling behavior
        pod_count = node_data.get('pod_count', [])
        if pod_count:
            features['pod_mean'] = np.mean(pod_count)
            features['pod_std'] = np.std(pod_count)
            features['pod_max'] = np.max(pod_count)
            features['pod_churn_rate'] = np.std(np.diff(pod_count)) if len(pod_count) > 1 else 0
        
        # Network and I/O patterns
        network_in = node_data.get('network_bytes_in', [])
        network_out = node_data.get('network_bytes_out', [])
        
        if network_in:
            features['network_in_mean'] = np.mean(network_in)
            features['network_in_volatility'] = np.std(network_in) / (np.mean(network_in) + 1)
            features['network_anomaly_score'] = len([x for x in network_in if x > np.mean(network_in) + 3*np.std(network_in)]) / len(network_in)
        
        if network_out:
            features['network_out_mean'] = np.mean(network_out)
            features['network_out_volatility'] = np.std(network_out) / (np.mean(network_out) + 1)
        
        # Disk I/O behavior
        disk_usage = node_data.get('disk_usage_percent', [])
        if disk_usage:
            features['disk_mean'] = np.mean(disk_usage)
            features['disk_growth_rate'] = np.polyfit(range(len(disk_usage)), disk_usage, 1)[0] if len(disk_usage) > 1 else 0
        
        # Reliability and stability metrics
        restart_count = node_data.get('container_restarts', 0)
        uptime_hours = node_data.get('uptime_hours', 168)
        
        features['restart_rate'] = restart_count / max(uptime_hours, 1)
        features['stability_score'] = max(0, 1 - features['restart_rate'])
        
        # Temporal behavior patterns
        timestamp_data = node_data.get('timestamps', [])
        if timestamp_data and len(timestamp_data) > 10:
            # Extract hour-of-day patterns
            hours = [datetime.fromisoformat(ts.replace('Z', '+00:00')).hour for ts in timestamp_data]
            features['peak_hour_activity'] = max(set(hours), key=hours.count) / 24.0
            features['activity_spread'] = len(set(hours)) / 24.0
        
        # Composite anomaly indicators
        features['resource_imbalance'] = abs(features.get('cpu_mean', 0) - features.get('memory_mean', 0))
        features['performance_degradation'] = (features.get('load_mean', 0) / max(features.get('cpu_mean', 0.1), 0.1))
        
        # Handle missing values
        for key, value in features.items():
            if np.isnan(value) or np.isinf(value):
                features[key] = 0.0
        
        return features
    
    def prepare_training_data(self, historical_data: List[Dict]) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare feature matrix for training
        
        Args:
            historical_data: List of node historical data
            
        Returns:
            Feature matrix and feature names
        """
        logger.info(f"Preparing training data from {len(historical_data)} node samples")
        
        feature_dicts = []
        for node_data in historical_data:
            features = self.generate_node_behavioral_features(node_data)
            feature_dicts.append(features)
        
        # Convert to DataFrame for consistent feature ordering
        df = pd.DataFrame(feature_dicts)
        df = df.fillna(0)  # Fill any remaining NaN values
        
        self.feature_names = list(df.columns)
        feature_matrix = df.values
        
        logger.info(f"Generated {len(self.feature_names)} behavioral features")
        logger.info(f"Feature matrix shape: {feature_matrix.shape}")
        
        return feature_matrix, self.feature_names
    
    def train_anomaly_detector(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """
        Train Isolation Forest on historical node data
        
        Args:
            historical_data: Historical node behavioral data
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting Isolation Forest training")
        
        # Prepare features
        X, feature_names = self.prepare_training_data(historical_data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Generate anomaly scores
        anomaly_scores = self.model.decision_function(X_scaled)
        anomaly_predictions = self.model.predict(X_scaled)
        
        # Calculate metrics
        anomaly_count = sum(1 for pred in anomaly_predictions if pred == -1)
        anomaly_percentage = (anomaly_count / len(anomaly_predictions)) * 100
        
        # Feature importance approximation
        feature_importance = self._estimate_feature_importance(X_scaled)
        
        training_results = {
            'samples_trained': len(X),
            'features_count': len(feature_names),
            'anomalies_detected': anomaly_count,
            'anomaly_percentage': anomaly_percentage,
            'mean_anomaly_score': np.mean(anomaly_scores),
            'std_anomaly_score': np.std(anomaly_scores),
            'feature_importance': dict(zip(feature_names, feature_importance)),
            'contamination_rate': self.contamination
        }
        
        logger.info(f"Training completed: {anomaly_count}/{len(X)} anomalies detected ({anomaly_percentage:.1f}%)")
        
        return training_results
    
    def _estimate_feature_importance(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate feature importance for Isolation Forest
        Uses permutation-based importance approximation
        
        Args:
            X: Scaled feature matrix
            
        Returns:
            Feature importance scores
        """
        baseline_scores = self.model.decision_function(X)
        baseline_mean = np.mean(baseline_scores)
        
        importance_scores = []
        
        for feature_idx in range(X.shape[1]):
            # Permute feature values
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature_idx])
            
            # Calculate score change
            permuted_scores = self.model.decision_function(X_permuted)
            permuted_mean = np.mean(permuted_scores)
            
            # Importance is the change in mean anomaly score
            importance = abs(baseline_mean - permuted_mean)
            importance_scores.append(importance)
        
        # Normalize importance scores
        importance_array = np.array(importance_scores)
        if np.sum(importance_array) > 0:
            importance_array = importance_array / np.sum(importance_array)
        
        return importance_array
    
    def detect_anomalies(self, node_data: Dict) -> Dict[str, Any]:
        """
        Detect anomalies in node behavior
        
        Args:
            node_data: Current node metrics data
            
        Returns:
            Anomaly detection result
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detection")
        
        # Generate features
        features = self.generate_node_behavioral_features(node_data)
        
        # Convert to array with correct feature order
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict anomaly
        anomaly_score = self.model.decision_function(feature_vector_scaled)[0]
        is_anomaly = self.model.predict(feature_vector_scaled)[0] == -1
        
        # Calculate confidence
        confidence = min(0.95, abs(anomaly_score) / 2.0)
        
        # Identify contributing factors
        contributing_factors = self._identify_anomaly_factors(features, feature_vector_scaled[0])
        
        result = {
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'confidence': float(confidence),
            'risk_level': 'HIGH' if anomaly_score < -0.3 else 'MEDIUM' if anomaly_score < -0.1 else 'LOW',
            'contributing_factors': contributing_factors,
            'node_id': node_data.get('node_id', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _identify_anomaly_factors(self, features: Dict[str, float], feature_vector: np.ndarray) -> List[str]:
        """
        Identify which features contribute most to anomaly detection
        
        Args:
            features: Original feature dictionary
            feature_vector: Scaled feature vector
            
        Returns:
            List of contributing factor descriptions
        """
        contributing_factors = []
        
        # Check high CPU utilization
        if features.get('cpu_mean', 0) > 0.9:
            contributing_factors.append(f"High CPU utilization: {features['cpu_mean']:.1%}")
        
        # Check high memory pressure
        if features.get('memory_pressure', 0) > 0.5:
            contributing_factors.append(f"Memory pressure: {features['memory_pressure']:.1%}")
        
        # Check load spikes
        if features.get('load_spike_ratio', 0) > 0.1:
            contributing_factors.append(f"Load spikes: {features['load_spike_ratio']:.1%}")
        
        # Check restart rate
        if features.get('restart_rate', 0) > 0.1:
            contributing_factors.append(f"High restart rate: {features['restart_rate']:.2f}/hour")
        
        # Check memory leak indicator
        if features.get('memory_leak_indicator', 0) > 0.01:
            contributing_factors.append("Potential memory leak detected")
        
        # Check resource imbalance
        if features.get('resource_imbalance', 0) > 0.3:
            contributing_factors.append("CPU-Memory utilization imbalance")
        
        # Check performance degradation
        if features.get('performance_degradation', 0) > 3.0:
            contributing_factors.append("Performance degradation detected")
        
        return contributing_factors[:5]  # Return top 5 factors

def generate_synthetic_node_data(num_normal=800, num_anomalous=100, random_state=42) -> List[Dict]:
    """
    Generate synthetic node data for training and testing
    
    Args:
        num_normal: Number of normal node samples
        num_anomalous: Number of anomalous node samples
        random_state: Random seed
        
    Returns:
        List of synthetic node data
    """
    np.random.seed(random_state)
    synthetic_data = []
    
    # Generate normal node behavior
    for i in range(num_normal):
        # Normal healthy node patterns
        base_cpu = np.random.normal(0.4, 0.15)  # 40% average CPU
        base_memory = np.random.normal(0.5, 0.1)  # 50% average memory
        
        node_data = {
            'node_id': f'worker-{i:03d}',
            'cpu_utilization': np.clip(np.random.normal(base_cpu, 0.1, 100), 0, 1).tolist(),
            'memory_utilization': np.clip(np.random.normal(base_memory, 0.05, 100), 0, 1).tolist(),
            'load_average': np.clip(np.random.normal(1.5, 0.5, 100), 0, 8).tolist(),
            'pod_count': np.random.poisson(15, 100).tolist(),
            'network_bytes_in': np.random.lognormal(10, 1, 100).tolist(),
            'network_bytes_out': np.random.lognormal(9, 1, 100).tolist(),
            'disk_usage_percent': np.clip(np.random.normal(30, 5, 100), 0, 100).tolist(),
            'container_restarts': np.random.poisson(0.5),
            'uptime_hours': np.random.uniform(100, 720),
            'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(100)]
        }
        synthetic_data.append(node_data)
    
    # Generate anomalous node behavior
    for i in range(num_anomalous):
        anomaly_type = np.random.choice(['cpu_spike', 'memory_leak', 'load_anomaly', 'restart_storm', 'network_flood'])
        
        if anomaly_type == 'cpu_spike':
            # CPU spike anomaly
            cpu_util = np.concatenate([
                np.random.normal(0.3, 0.1, 70),
                np.random.normal(0.95, 0.03, 30)  # High CPU spike
            ])
            memory_util = np.random.normal(0.5, 0.1, 100)
            load_avg = np.concatenate([
                np.random.normal(1.5, 0.5, 70),
                np.random.normal(6.0, 1.0, 30)  # High load
            ])
            
        elif anomaly_type == 'memory_leak':
            # Memory leak pattern
            cpu_util = np.random.normal(0.4, 0.1, 100)
            memory_util = np.linspace(0.3, 0.95, 100) + np.random.normal(0, 0.02, 100)  # Growing memory
            load_avg = np.random.normal(2.0, 0.7, 100)
            
        elif anomaly_type == 'load_anomaly':
            # Load average anomaly
            cpu_util = np.random.normal(0.6, 0.2, 100)
            memory_util = np.random.normal(0.7, 0.1, 100)
            load_avg = np.random.normal(7.0, 2.0, 100)  # Very high load
            
        elif anomaly_type == 'restart_storm':
            # High restart rate
            cpu_util = np.random.normal(0.4, 0.2, 100)
            memory_util = np.random.normal(0.5, 0.2, 100)
            load_avg = np.random.normal(2.5, 1.0, 100)
            
        else:  # network_flood
            # Network flooding
            cpu_util = np.random.normal(0.3, 0.1, 100)
            memory_util = np.random.normal(0.4, 0.1, 100)
            load_avg = np.random.normal(1.8, 0.5, 100)
        
        node_data = {
            'node_id': f'anomaly-{i:03d}',
            'cpu_utilization': np.clip(cpu_util, 0, 1).tolist(),
            'memory_utilization': np.clip(memory_util, 0, 1).tolist(),
            'load_average': np.clip(load_avg, 0, 12).tolist(),
            'pod_count': np.random.poisson(20 if anomaly_type != 'restart_storm' else 8, 100).tolist(),
            'network_bytes_in': (np.random.lognormal(12, 2, 100) if anomaly_type == 'network_flood' 
                               else np.random.lognormal(10, 1, 100)).tolist(),
            'network_bytes_out': (np.random.lognormal(11, 2, 100) if anomaly_type == 'network_flood'
                                else np.random.lognormal(9, 1, 100)).tolist(),
            'disk_usage_percent': np.clip(np.random.normal(50, 15, 100), 0, 100).tolist(),
            'container_restarts': np.random.poisson(15) if anomaly_type == 'restart_storm' else np.random.poisson(1),
            'uptime_hours': np.random.uniform(50, 200) if anomaly_type == 'restart_storm' else np.random.uniform(100, 720),
            'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(100)],
            'anomaly_type': anomaly_type  # For validation
        }
        synthetic_data.append(node_data)
    
    return synthetic_data

def main():
    """
    Main training and validation function
    """
    logger.info("Starting Isolation Forest Detector - Le Detective")
    logger.info("Step 6.1: Analyse Comportements Nodes et Feature Engineering")
    
    # Generate synthetic training data
    logger.info("Generating synthetic node behavioral data")
    training_data = generate_synthetic_node_data(num_normal=800, num_anomalous=100)
    
    # Initialize detector
    detector = IsolationForestDetector(contamination=0.1)
    
    # Train anomaly detector
    training_results = detector.train_anomaly_detector(training_data)
    
    # Display results
    print("\n" + "="*60)
    print("ISOLATION FOREST TRAINING RESULTS")
    print("="*60)
    print(f"Samples trained: {training_results['samples_trained']}")
    print(f"Features engineered: {training_results['features_count']}")
    print(f"Anomalies detected: {training_results['anomalies_detected']}")
    print(f"Anomaly rate: {training_results['anomaly_percentage']:.1f}%")
    print(f"Mean anomaly score: {training_results['mean_anomaly_score']:.3f}")
    
    print("\nTop 10 Most Important Features:")
    sorted_features = sorted(training_results['feature_importance'].items(), 
                           key=lambda x: x[1], reverse=True)[:10]
    for feature, importance in sorted_features:
        print(f"  {feature}: {importance:.3f}")
    
    # Test on new synthetic data
    logger.info("Testing anomaly detection on new samples")
    test_data = generate_synthetic_node_data(num_normal=20, num_anomalous=5, random_state=123)
    
    correct_predictions = 0
    total_predictions = len(test_data)
    
    print("\n" + "="*60)
    print("ANOMALY DETECTION TEST RESULTS")
    print("="*60)
    
    for node_data in test_data:
        result = detector.detect_anomalies(node_data)
        actual_anomaly = 'anomaly' in node_data['node_id']
        predicted_anomaly = result['is_anomaly']
        
        if actual_anomaly == predicted_anomaly:
            correct_predictions += 1
        
        print(f"Node {node_data['node_id']}: "
              f"Predicted={'ANOMALY' if predicted_anomaly else 'NORMAL'} "
              f"(Score: {result['anomaly_score']:.3f}, "
              f"Confidence: {result['confidence']:.2f}, "
              f"Risk: {result['risk_level']})")
        
        if result['contributing_factors']:
            print(f"  Contributing factors: {', '.join(result['contributing_factors'])}")
    
    accuracy = correct_predictions / total_predictions
    print(f"\nTest Accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    # Save model results
    results_dir = "./models/isolation_detector"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save training results
    with open(f"{results_dir}/training_results.json", 'w') as f:
        json.dump(training_results, f, indent=2)
    
    logger.info(f"Training results saved to {results_dir}/training_results.json")
    logger.info("Step 6.1 completed successfully - Behavioral analysis and feature engineering done")

if __name__ == "__main__":
    main()