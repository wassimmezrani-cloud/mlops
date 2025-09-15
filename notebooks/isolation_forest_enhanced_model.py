#!/usr/bin/env python3
"""
Isolation Forest Enhanced Model - Le Detective
Enhanced sensitivity for better incident detection
Improved anomaly threshold and feature weighting
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

from isolation_forest_detector import IsolationForestDetector, generate_synthetic_node_data

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedIsolationForestDetector(IsolationForestDetector):
    """
    Enhanced Isolation Forest with improved sensitivity for incident detection
    """
    
    def __init__(self, contamination=0.15, random_state=42):
        """
        Initialize enhanced detector with higher sensitivity
        
        Args:
            contamination: Increased contamination rate for better detection
            random_state: Random seed
        """
        super().__init__(contamination, random_state)
        
        # Enhanced model parameters
        self.model = IsolationForest(
            n_estimators=300,          # More trees for better detection
            contamination=contamination,
            max_samples=0.8,           # Smaller samples for more diversity
            max_features=0.7,          # Feature subsampling
            random_state=random_state,
            n_jobs=-1
        )
        
        # Adjusted anomaly threshold for higher sensitivity
        self.anomaly_threshold = 0.05  # More sensitive threshold
        
        logger.info(f"Enhanced Isolation Forest initialized with contamination={contamination}")
    
    def generate_enhanced_behavioral_features(self, node_data: Dict) -> Dict[str, float]:
        """
        Generate enhanced features with better anomaly discrimination
        
        Args:
            node_data: Node metrics data
            
        Returns:
            Enhanced feature dictionary
        """
        # Start with base features
        features = super().generate_node_behavioral_features(node_data)
        
        # Add enhanced anomaly-sensitive features
        cpu_util = node_data.get('cpu_utilization', [])
        memory_util = node_data.get('memory_utilization', [])
        load_avg = node_data.get('load_average', [])
        
        if cpu_util and len(cpu_util) > 10:
            # CPU anomaly patterns
            cpu_array = np.array(cpu_util)
            
            # Sudden spike detection
            cpu_diff = np.diff(cpu_array)
            features['cpu_sudden_spike'] = np.max(cpu_diff) if len(cpu_diff) > 0 else 0
            features['cpu_sudden_drop'] = abs(np.min(cpu_diff)) if len(cpu_diff) > 0 else 0
            
            # Sustained high utilization
            features['cpu_sustained_high'] = sum(1 for x in cpu_util[-20:] if x > 0.8) / min(20, len(cpu_util))
            
            # CPU variability (instability indicator)
            if len(cpu_util) > 5:
                rolling_std = []
                for i in range(5, len(cpu_util)):
                    rolling_std.append(np.std(cpu_util[i-5:i]))
                features['cpu_instability'] = np.mean(rolling_std) if rolling_std else 0
            
            # Percentile-based features
            features['cpu_p95'] = np.percentile(cpu_util, 95)
            features['cpu_p99'] = np.percentile(cpu_util, 99)
        
        if memory_util and len(memory_util) > 10:
            # Memory anomaly patterns
            memory_array = np.array(memory_util)
            
            # Memory leak detection (improved)
            if len(memory_array) >= 20:
                # Check trend over different windows
                recent_trend = np.polyfit(range(20), memory_array[-20:], 1)[0]
                overall_trend = np.polyfit(range(len(memory_array)), memory_array, 1)[0]
                features['memory_leak_recent'] = max(0, recent_trend)
                features['memory_leak_overall'] = max(0, overall_trend)
                features['memory_leak_acceleration'] = max(0, recent_trend - overall_trend)
            
            # Memory pressure indicators
            features['memory_critical_time'] = sum(1 for x in memory_util if x > 0.9) / len(memory_util)
            features['memory_pressure_gradient'] = np.max(np.diff(memory_array)) if len(memory_array) > 1 else 0
        
        if load_avg and len(load_avg) > 5:
            # Load average anomaly patterns
            load_array = np.array(load_avg)
            
            # Load spike detection
            features['load_spike_intensity'] = np.max(load_array) / (np.mean(load_array) + 0.1)
            features['load_sustained_high'] = sum(1 for x in load_avg if x > 4.0) / len(load_avg)
            
            # Load instability
            features['load_coefficient_variation'] = np.std(load_array) / (np.mean(load_array) + 0.1)
        
        # Cross-metric correlations (anomaly indicators)
        if cpu_util and memory_util and len(cpu_util) == len(memory_util):
            # Unusual CPU-Memory correlation
            correlation = np.corrcoef(cpu_util, memory_util)[0, 1]
            features['cpu_memory_correlation'] = correlation if not np.isnan(correlation) else 0
            
            # Resource competition indicator
            cpu_high = [1 if x > 0.7 else 0 for x in cpu_util]
            mem_high = [1 if x > 0.7 else 0 for x in memory_util]
            features['resource_competition'] = sum(1 for i in range(len(cpu_high)) if cpu_high[i] and mem_high[i]) / len(cpu_high)
        
        # System health indicators
        restart_rate = features.get('restart_rate', 0)
        cpu_mean = features.get('cpu_mean', 0)
        memory_mean = features.get('memory_mean', 0)
        load_mean = features.get('load_mean', 0)
        
        # Composite anomaly score
        features['system_stress_score'] = (
            cpu_mean * 0.3 + 
            memory_mean * 0.3 + 
            (load_mean / 8.0) * 0.2 +  # Normalize load
            restart_rate * 0.2
        )
        
        # Resource efficiency anomaly
        expected_load = max(cpu_mean * 2, 0.5)  # Expected load based on CPU
        actual_load = load_mean
        features['load_efficiency_anomaly'] = abs(actual_load - expected_load) / (expected_load + 0.1)
        
        return features
    
    def detect_anomalies(self, node_data: Dict) -> Dict[str, Any]:
        """
        Enhanced anomaly detection with improved sensitivity
        
        Args:
            node_data: Node metrics data
            
        Returns:
            Enhanced anomaly detection result
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detection")
        
        # Generate enhanced features
        features = self.generate_enhanced_behavioral_features(node_data)
        
        # Convert to array
        feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Get anomaly score and prediction
        anomaly_score = self.model.decision_function(feature_vector_scaled)[0]
        is_anomaly_standard = self.model.predict(feature_vector_scaled)[0] == -1
        
        # Enhanced anomaly detection with multiple thresholds
        is_anomaly_sensitive = anomaly_score < self.anomaly_threshold
        
        # Combine standard and sensitive detection
        is_anomaly = is_anomaly_standard or is_anomaly_sensitive
        
        # Enhanced confidence calculation
        confidence = min(0.95, abs(anomaly_score) * 2.0)
        
        # Risk level with more granular classification
        if anomaly_score < -0.2:
            risk_level = 'CRITICAL'
        elif anomaly_score < -0.1:
            risk_level = 'HIGH'
        elif anomaly_score < 0.05:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Enhanced contributing factors identification
        contributing_factors = self._identify_enhanced_anomaly_factors(features, feature_vector_scaled[0])
        
        result = {
            'is_anomaly': is_anomaly,
            'anomaly_score': float(anomaly_score),
            'confidence': float(confidence),
            'risk_level': risk_level,
            'detection_method': 'SENSITIVE' if is_anomaly_sensitive and not is_anomaly_standard else 'STANDARD',
            'contributing_factors': contributing_factors,
            'node_id': node_data.get('node_id', 'unknown'),
            'timestamp': datetime.now().isoformat(),
            'enhanced_metrics': {
                'system_stress_score': features.get('system_stress_score', 0),
                'cpu_instability': features.get('cpu_instability', 0),
                'memory_leak_recent': features.get('memory_leak_recent', 0),
                'load_spike_intensity': features.get('load_spike_intensity', 1)
            }
        }
        
        return result
    
    def _identify_enhanced_anomaly_factors(self, features: Dict[str, float], 
                                         feature_vector: np.ndarray) -> List[str]:
        """
        Enhanced anomaly factor identification
        
        Args:
            features: Feature dictionary
            feature_vector: Scaled feature vector
            
        Returns:
            List of contributing factors
        """
        factors = []
        
        # CPU-related anomalies
        if features.get('cpu_sustained_high', 0) > 0.5:
            factors.append(f"Sustained high CPU: {features['cpu_sustained_high']:.1%} of time >80%")
        
        if features.get('cpu_sudden_spike', 0) > 0.3:
            factors.append(f"CPU sudden spike: +{features['cpu_sudden_spike']:.1%}")
        
        if features.get('cpu_instability', 0) > 0.1:
            factors.append(f"CPU instability detected: σ={features['cpu_instability']:.3f}")
        
        # Memory-related anomalies
        if features.get('memory_leak_recent', 0) > 0.005:
            factors.append(f"Recent memory leak: +{features['memory_leak_recent']:.1%}/min")
        
        if features.get('memory_critical_time', 0) > 0.1:
            factors.append(f"Memory critical time: {features['memory_critical_time']:.1%}")
        
        if features.get('memory_pressure_gradient', 0) > 0.2:
            factors.append(f"Sharp memory increase: +{features['memory_pressure_gradient']:.1%}")
        
        # Load-related anomalies
        if features.get('load_spike_intensity', 0) > 3.0:
            factors.append(f"Load spike intensity: {features['load_spike_intensity']:.1f}x normal")
        
        if features.get('load_sustained_high', 0) > 0.3:
            factors.append(f"Sustained high load: {features['load_sustained_high']:.1%} of time >4.0")
        
        # System-level anomalies
        if features.get('system_stress_score', 0) > 0.7:
            factors.append(f"High system stress: {features['system_stress_score']:.2f}/1.0")
        
        if features.get('resource_competition', 0) > 0.3:
            factors.append(f"Resource competition: {features['resource_competition']:.1%}")
        
        if features.get('load_efficiency_anomaly', 0) > 1.0:
            factors.append("Load-CPU efficiency anomaly detected")
        
        # Restart and stability issues
        if features.get('restart_rate', 0) > 0.05:
            factors.append(f"High restart rate: {features['restart_rate']:.3f}/hour")
        
        return factors[:7]  # Return top 7 factors

def retrain_enhanced_model() -> str:
    """
    Retrain model with enhanced features and sensitivity
    
    Returns:
        Path to enhanced model
    """
    logger.info("Retraining with enhanced Isolation Forest model")
    
    # Generate more diverse training data
    normal_data = generate_synthetic_node_data(num_normal=1000, num_anomalous=0, random_state=42)
    anomalous_data = generate_synthetic_node_data(num_normal=0, num_anomalous=200, random_state=123)
    
    training_data = normal_data + anomalous_data
    
    # Initialize enhanced detector
    enhanced_detector = EnhancedIsolationForestDetector(contamination=0.15)
    
    # Prepare enhanced features
    feature_dicts = []
    for node_data in training_data:
        features = enhanced_detector.generate_enhanced_behavioral_features(node_data)
        feature_dicts.append(features)
    
    # Convert to DataFrame and train
    df = pd.DataFrame(feature_dicts).fillna(0)
    enhanced_detector.feature_names = list(df.columns)
    X = df.values
    
    # Scale and train
    X_scaled = enhanced_detector.scaler.fit_transform(X)
    enhanced_detector.model.fit(X_scaled)
    enhanced_detector.is_trained = True
    
    # Save enhanced model
    model_dir = "./models/isolation_detector"
    os.makedirs(model_dir, exist_ok=True)
    
    enhanced_model_path = f"{model_dir}/isolation_forest_enhanced.pkl"
    joblib.dump({
        'model': enhanced_detector.model,
        'scaler': enhanced_detector.scaler,
        'feature_names': enhanced_detector.feature_names,
        'contamination': enhanced_detector.contamination,
        'anomaly_threshold': enhanced_detector.anomaly_threshold
    }, enhanced_model_path)
    
    # Save enhanced metadata
    metadata = {
        'model_name': 'enhanced_isolation_forest',
        'version': '2.0.0',
        'features_count': len(enhanced_detector.feature_names),
        'contamination_rate': enhanced_detector.contamination,
        'anomaly_threshold': enhanced_detector.anomaly_threshold,
        'enhancement_date': datetime.now().isoformat(),
        'improvements': [
            'Increased sensitivity with threshold=0.05',
            'Enhanced feature engineering for anomaly patterns',
            'Better CPU instability detection',
            'Improved memory leak detection',
            'Load spike intensity analysis',
            'Cross-metric correlation analysis'
        ]
    }
    
    with open(f"{model_dir}/enhanced_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Enhanced model saved to {enhanced_model_path}")
    logger.info(f"Features count: {len(enhanced_detector.feature_names)}")
    
    return enhanced_model_path

def main():
    """Main enhanced model training"""
    print("="*70)
    print("ISOLATION FOREST ENHANCED MODEL - LE DETECTIVE")
    print("Enhanced Sensitivity for Better Incident Detection")
    print("="*70)
    
    # Retrain with enhanced model
    enhanced_model_path = retrain_enhanced_model()
    
    print(f"\n✅ Enhanced model trained successfully!")
    print(f"Model path: {enhanced_model_path}")
    print("\nEnhancements:")
    print("  • Increased contamination rate: 15% (was 10%)")
    print("  • Lowered anomaly threshold: 0.05 (was -0.1)")
    print("  • Added 15+ new behavioral features")
    print("  • Enhanced CPU instability detection") 
    print("  • Improved memory leak pattern recognition")
    print("  • Load spike intensity analysis")
    print("  • Cross-metric correlation analysis")
    
    print(f"\n🎯 Enhanced Le Detective ready for re-validation!")
    logger.info("Enhanced model training completed")

if __name__ == "__main__":
    main()