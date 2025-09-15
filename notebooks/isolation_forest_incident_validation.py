#!/usr/bin/env python3
"""
Isolation Forest Incident Validation - Le Detective Step 6.3
Advanced validation with realistic historical incident scenarios
Production validation against real-world failure patterns
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging

from isolation_forest_detector import IsolationForestDetector

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncidentValidationSuite:
    """
    Comprehensive validation suite for Isolation Forest against historical incidents
    Simulates real-world Kubernetes cluster failure scenarios
    """
    
    def __init__(self, model_path: str):
        """
        Initialize validation suite with trained model
        
        Args:
            model_path: Path to trained Isolation Forest model
        """
        self.model_path = model_path
        self.detector = self.load_production_model()
        self.incident_scenarios = self.define_incident_scenarios()
        
        logger.info("Incident Validation Suite initialized")
    
    def load_production_model(self) -> IsolationForestDetector:
        """Load production Isolation Forest model"""
        logger.info(f"Loading production model from {self.model_path}")
        
        # Load model artifacts
        artifacts = joblib.load(self.model_path)
        
        # Reconstruct detector
        detector = IsolationForestDetector()
        detector.model = artifacts['model']
        detector.scaler = artifacts['scaler']
        detector.feature_names = artifacts['feature_names']
        detector.contamination = artifacts['contamination']
        detector.is_trained = True
        
        logger.info("Production model loaded successfully")
        return detector
    
    def define_incident_scenarios(self) -> Dict[str, Dict]:
        """
        Define realistic incident scenarios based on production experience
        
        Returns:
            Dictionary of incident scenario definitions
        """
        scenarios = {
            'memory_leak_gradual': {
                'description': 'Gradual memory leak over 2 hours',
                'severity': 'high',
                'duration_minutes': 120,
                'patterns': {
                    'memory_growth_rate': 0.005,  # 0.5% per minute
                    'cpu_correlation': 0.3,       # CPU increases with memory
                    'load_spike_at_end': True     # System becomes unresponsive
                }
            },
            'cpu_spike_sudden': {
                'description': 'Sudden CPU spike to 100% for 15 minutes',
                'severity': 'high',
                'duration_minutes': 15,
                'patterns': {
                    'cpu_target': 0.99,
                    'load_multiplier': 4.0,
                    'memory_stable': True
                }
            },
            'disk_io_storm': {
                'description': 'Disk I/O bottleneck causing system slowdown',
                'severity': 'medium',
                'duration_minutes': 45,
                'patterns': {
                    'load_avg_high': 8.0,
                    'cpu_wait_high': True,
                    'memory_pressure': 0.8
                }
            },
            'network_saturation': {
                'description': 'Network bandwidth saturation',
                'severity': 'medium',
                'duration_minutes': 30,
                'patterns': {
                    'network_multiplier': 20,
                    'cpu_network_processing': 0.6,
                    'load_network_related': 3.5
                }
            },
            'pod_crash_cascade': {
                'description': 'Cascade of pod crashes and restarts',
                'severity': 'high',
                'duration_minutes': 60,
                'patterns': {
                    'restart_storm': 25,
                    'cpu_instability': True,
                    'memory_fragmentation': True
                }
            },
            'resource_exhaustion_combo': {
                'description': 'Combined CPU+Memory+Disk exhaustion',
                'severity': 'critical',
                'duration_minutes': 90,
                'patterns': {
                    'cpu_target': 0.95,
                    'memory_target': 0.95,
                    'disk_growth': True,
                    'load_extreme': 12.0
                }
            },
            'kernel_memory_pressure': {
                'description': 'Kernel memory pressure causing OOM kills',
                'severity': 'critical',
                'duration_minutes': 20,
                'patterns': {
                    'memory_pressure_extreme': True,
                    'oom_kills': True,
                    'cpu_spike_correlation': True
                }
            },
            'thermal_throttling': {
                'description': 'CPU thermal throttling under sustained load',
                'severity': 'medium',
                'duration_minutes': 180,
                'patterns': {
                    'cpu_thermal_pattern': True,
                    'performance_degradation': True,
                    'load_inconsistent': True
                }
            }
        }
        
        logger.info(f"Defined {len(scenarios)} realistic incident scenarios")
        return scenarios
    
    def generate_incident_timeline(self, scenario_name: str, scenario_def: Dict) -> Dict[str, Any]:
        """
        Generate realistic timeline data for incident scenario
        
        Args:
            scenario_name: Name of incident scenario
            scenario_def: Scenario definition
            
        Returns:
            Generated incident timeline data
        """
        duration = scenario_def['duration_minutes']
        patterns = scenario_def['patterns']
        
        # Generate timeline
        timeline_points = 120  # 2-hour timeline with incident in middle
        incident_start = 30    # Incident starts at 30min mark
        incident_end = incident_start + duration
        
        # Initialize baseline patterns
        cpu_baseline = np.random.normal(0.3, 0.1, timeline_points)
        memory_baseline = np.random.normal(0.4, 0.05, timeline_points)
        load_baseline = np.random.normal(1.2, 0.3, timeline_points)
        
        # Apply incident patterns
        if scenario_name == 'memory_leak_gradual':
            # Gradual memory increase
            leak_points = np.arange(incident_start, min(incident_end, timeline_points))
            for i, point in enumerate(leak_points):
                memory_baseline[point] = min(0.99, 
                    memory_baseline[incident_start] + i * patterns['memory_growth_rate'])
            
            # Correlate CPU with memory pressure
            cpu_correlation = patterns.get('cpu_correlation', 0.3)
            for point in leak_points:
                cpu_baseline[point] += memory_baseline[point] * cpu_correlation
            
            # Load spike at end if memory critical
            if patterns.get('load_spike_at_end') and incident_end < timeline_points:
                load_baseline[incident_end-5:incident_end] = np.random.normal(7.0, 1.0, 5)
        
        elif scenario_name == 'cpu_spike_sudden':
            # Sudden CPU spike
            cpu_target = patterns['cpu_target']
            spike_points = np.arange(incident_start, min(incident_end, timeline_points))
            cpu_baseline[spike_points] = np.random.normal(cpu_target, 0.02, len(spike_points))
            
            # Load increase with CPU
            load_multiplier = patterns['load_multiplier']
            load_baseline[spike_points] *= load_multiplier
        
        elif scenario_name == 'disk_io_storm':
            # High load average due to I/O wait
            io_points = np.arange(incident_start, min(incident_end, timeline_points))
            load_baseline[io_points] = np.random.normal(patterns['load_avg_high'], 1.0, len(io_points))
            
            # Memory pressure during I/O storm
            memory_baseline[io_points] = np.random.normal(patterns['memory_pressure'], 0.05, len(io_points))
        
        elif scenario_name == 'network_saturation':
            # Network-related CPU increase
            net_points = np.arange(incident_start, min(incident_end, timeline_points))
            cpu_baseline[net_points] = np.random.normal(patterns['cpu_network_processing'], 0.1, len(net_points))
            load_baseline[net_points] = np.random.normal(patterns['load_network_related'], 0.5, len(net_points))
        
        elif scenario_name == 'pod_crash_cascade':
            # Instability patterns
            crash_points = np.arange(incident_start, min(incident_end, timeline_points))
            cpu_baseline[crash_points] += np.random.normal(0, 0.2, len(crash_points))  # Unstable CPU
            memory_baseline[crash_points] += np.random.normal(0, 0.15, len(crash_points))  # Memory fragmentation
        
        elif scenario_name == 'resource_exhaustion_combo':
            # Combined exhaustion
            exhaust_points = np.arange(incident_start, min(incident_end, timeline_points))
            cpu_baseline[exhaust_points] = np.random.normal(patterns['cpu_target'], 0.03, len(exhaust_points))
            memory_baseline[exhaust_points] = np.random.normal(patterns['memory_target'], 0.02, len(exhaust_points))
            load_baseline[exhaust_points] = np.random.normal(patterns['load_extreme'], 2.0, len(exhaust_points))
        
        elif scenario_name == 'kernel_memory_pressure':
            # Extreme memory pressure
            pressure_points = np.arange(incident_start, min(incident_end, timeline_points))
            memory_baseline[pressure_points] = np.random.normal(0.98, 0.01, len(pressure_points))
            
            # CPU spikes from OOM killer activity
            cpu_baseline[pressure_points] += np.random.exponential(0.3, len(pressure_points))
        
        elif scenario_name == 'thermal_throttling':
            # CPU throttling pattern
            throttle_points = np.arange(incident_start, min(incident_end, timeline_points))
            # Saw-tooth pattern from thermal throttling
            for i, point in enumerate(throttle_points):
                cpu_baseline[point] = 0.4 + 0.4 * np.sin(i * 0.3) + np.random.normal(0, 0.05)
        
        # Clip values to realistic ranges
        cpu_utilization = np.clip(cpu_baseline, 0, 1).tolist()
        memory_utilization = np.clip(memory_baseline, 0, 1).tolist()
        load_average = np.clip(load_baseline, 0, 20).tolist()
        
        # Generate additional metrics
        pod_count = np.random.poisson(15, timeline_points).tolist()
        if scenario_name == 'pod_crash_cascade':
            # Reduce pod count during crash cascade
            crash_points = np.arange(incident_start, min(incident_end, timeline_points))
            for point in crash_points:
                pod_count[point] = max(5, pod_count[point] - np.random.poisson(3))
        
        # Network metrics
        network_multiplier = patterns.get('network_multiplier', 1)
        network_in = (np.random.lognormal(10, 1, timeline_points) * network_multiplier).tolist()
        network_out = (np.random.lognormal(9, 1, timeline_points) * network_multiplier).tolist()
        
        # Disk usage
        disk_usage = np.random.normal(40, 10, timeline_points)
        if patterns.get('disk_growth'):
            growth_points = np.arange(incident_start, min(incident_end, timeline_points))
            for i, point in enumerate(growth_points):
                disk_usage[point] += i * 0.2
        disk_usage = np.clip(disk_usage, 0, 100).tolist()
        
        # Container restarts
        restart_base = 1
        if scenario_name == 'pod_crash_cascade':
            restart_base = patterns['restart_storm']
        
        incident_data = {
            'incident_id': f'{scenario_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'scenario_name': scenario_name,
            'description': scenario_def['description'],
            'severity': scenario_def['severity'],
            'node_id': f'node_incident_{scenario_name}',
            'cpu_utilization': cpu_utilization,
            'memory_utilization': memory_utilization,
            'load_average': load_average,
            'pod_count': pod_count,
            'network_bytes_in': network_in,
            'network_bytes_out': network_out,
            'disk_usage_percent': disk_usage,
            'container_restarts': restart_base,
            'uptime_hours': np.random.uniform(50, 200),
            'timestamps': [(datetime.now() - timedelta(minutes=x)).isoformat() for x in range(timeline_points)],
            'incident_window': {
                'start_minute': incident_start,
                'end_minute': incident_end,
                'duration_minutes': duration
            },
            'ground_truth_anomaly': True
        }
        
        return incident_data
    
    def run_incident_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive incident validation
        
        Returns:
            Detailed validation results
        """
        logger.info("Starting comprehensive incident validation")
        
        validation_results = {
            'scenarios_tested': len(self.incident_scenarios),
            'scenarios_detected': 0,
            'scenarios_missed': 0,
            'detection_details': [],
            'severity_analysis': {'low': [], 'medium': [], 'high': [], 'critical': []},
            'performance_metrics': {}
        }
        
        all_predictions = []
        all_ground_truth = []
        
        for scenario_name, scenario_def in self.incident_scenarios.items():
            logger.info(f"Testing scenario: {scenario_name}")
            
            # Generate incident timeline
            incident_data = self.generate_incident_timeline(scenario_name, scenario_def)
            
            # Detect anomalies
            detection_result = self.detector.detect_anomalies(incident_data)
            
            # Record results
            detected = detection_result['is_anomaly']
            all_predictions.append(detected)
            all_ground_truth.append(True)  # All scenarios should be detected
            
            if detected:
                validation_results['scenarios_detected'] += 1
                status = "DETECTED"
            else:
                validation_results['scenarios_missed'] += 1
                status = "MISSED"
            
            # Detailed analysis
            detail = {
                'scenario_name': scenario_name,
                'description': scenario_def['description'],
                'severity': scenario_def['severity'],
                'detected': detected,
                'status': status,
                'anomaly_score': detection_result['anomaly_score'],
                'confidence': detection_result['confidence'],
                'risk_level': detection_result['risk_level'],
                'contributing_factors': detection_result['contributing_factors'][:5],
                'incident_duration': scenario_def['duration_minutes']
            }
            
            validation_results['detection_details'].append(detail)
            validation_results['severity_analysis'][scenario_def['severity']].append(detail)
        
        # Calculate overall metrics
        detection_rate = validation_results['scenarios_detected'] / validation_results['scenarios_tested']
        
        # Severity-specific metrics
        severity_metrics = {}
        for severity, details in validation_results['severity_analysis'].items():
            if details:
                detected = sum(1 for d in details if d['detected'])
                severity_metrics[severity] = {
                    'total': len(details),
                    'detected': detected,
                    'rate': detected / len(details),
                    'scenarios': [d['scenario_name'] for d in details]
                }
        
        validation_results['performance_metrics'] = {
            'overall_detection_rate': detection_rate,
            'severity_metrics': severity_metrics,
            'mean_confidence': np.mean([d['confidence'] for d in validation_results['detection_details']]),
            'mean_anomaly_score': np.mean([d['anomaly_score'] for d in validation_results['detection_details']])
        }
        
        logger.info(f"Validation completed: {validation_results['scenarios_detected']}/{validation_results['scenarios_tested']} scenarios detected")
        
        return validation_results
    
    def calculate_incident_validation_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate business score based on incident detection performance
        
        Args:
            results: Validation results
            
        Returns:
            Business scoring metrics
        """
        metrics = results['performance_metrics']
        
        # Scoring weights by severity
        severity_weights = {
            'critical': 0.4,  # Critical incidents must be caught
            'high': 0.3,      # High severity very important
            'medium': 0.2,    # Medium severity important
            'low': 0.1        # Low severity less critical
        }
        
        # Calculate severity-weighted score
        weighted_score = 0
        total_weight = 0
        
        for severity, weight in severity_weights.items():
            if severity in metrics['severity_metrics']:
                sev_metrics = metrics['severity_metrics'][severity]
                detection_rate = sev_metrics['rate']
                weighted_score += detection_rate * weight * 100
                total_weight += weight
        
        if total_weight > 0:
            business_score = weighted_score / total_weight
        else:
            business_score = 0
        
        # Performance categories
        performance_level = "EXCELLENT" if business_score >= 90 else \
                          "GOOD" if business_score >= 80 else \
                          "ACCEPTABLE" if business_score >= 70 else "NEEDS_IMPROVEMENT"
        
        return {
            'incident_detection_score': business_score,
            'performance_level': performance_level,
            'severity_breakdown': metrics['severity_metrics'],
            'overall_detection_rate': metrics['overall_detection_rate'],
            'production_ready': business_score >= 75.0,
            'recommendations': self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on results"""
        recommendations = []
        
        severity_metrics = results['performance_metrics']['severity_metrics']
        
        # Check critical incident detection
        if 'critical' in severity_metrics:
            critical_rate = severity_metrics['critical']['rate']
            if critical_rate < 1.0:
                recommendations.append("Improve critical incident detection - currently missing some critical scenarios")
        
        # Check high severity detection
        if 'high' in severity_metrics:
            high_rate = severity_metrics['high']['rate']
            if high_rate < 0.9:
                recommendations.append("Enhance high severity incident detection patterns")
        
        # Overall detection rate
        overall_rate = results['performance_metrics']['overall_detection_rate']
        if overall_rate < 0.85:
            recommendations.append("Increase overall detection sensitivity")
        
        # Confidence analysis
        mean_confidence = results['performance_metrics']['mean_confidence']
        if mean_confidence < 0.7:
            recommendations.append("Improve model confidence calibration")
        
        if not recommendations:
            recommendations.append("Model performs excellently on incident detection")
        
        return recommendations
    
    def save_validation_results(self, results: Dict[str, Any], scoring: Dict[str, Any]) -> str:
        """Save comprehensive validation results"""
        results_dir = "./models/isolation_detector"
        os.makedirs(results_dir, exist_ok=True)
        
        # Comprehensive validation report
        validation_report = {
            'validation_metadata': {
                'validation_type': 'Historical Incident Validation',
                'model_path': self.model_path,
                'validation_date': datetime.now().isoformat(),
                'scenarios_count': len(self.incident_scenarios)
            },
            'validation_results': results,
            'business_scoring': scoring,
            'scenario_definitions': self.incident_scenarios
        }
        
        report_path = f"{results_dir}/incident_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {report_path}")
        return report_path

def main():
    """Main validation execution"""
    print("="*70)
    print("ISOLATION FOREST INCIDENT VALIDATION - LE DETECTIVE")
    print("Step 6.3: Validation avec Incidents Historiques")
    print("="*70)
    
    # Load model and run validation
    model_path = "./models/isolation_detector/isolation_forest_production.pkl"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please run fast training first to generate the model.")
        return
    
    validator = IncidentValidationSuite(model_path)
    
    # Run comprehensive validation
    results = validator.run_incident_validation()
    scoring = validator.calculate_incident_validation_score(results)
    
    # Display results
    print("\n" + "="*50)
    print("INCIDENT VALIDATION RESULTS")
    print("="*50)
    
    print(f"Scenarios tested: {results['scenarios_tested']}")
    print(f"Scenarios detected: {results['scenarios_detected']}")
    print(f"Overall detection rate: {results['performance_metrics']['overall_detection_rate']:.1%}")
    print(f"Business score: {scoring['incident_detection_score']:.1f}/100")
    print(f"Performance level: {scoring['performance_level']}")
    
    print("\nSeverity Analysis:")
    for severity, metrics in scoring['severity_breakdown'].items():
        print(f"  {severity.capitalize()}: {metrics['detected']}/{metrics['total']} ({metrics['rate']:.1%})")
    
    print("\nScenario Details:")
    for detail in results['detection_details']:
        status_icon = "✅" if detail['detected'] else "❌"
        print(f"  {status_icon} {detail['scenario_name']} ({detail['severity']}) - "
              f"Score: {detail['anomaly_score']:.3f}, Confidence: {detail['confidence']:.2f}")
    
    print("\nRecommendations:")
    for rec in scoring['recommendations']:
        print(f"  • {rec}")
    
    # Save results
    report_path = validator.save_validation_results(results, scoring)
    print(f"\nValidation report saved: {report_path}")
    
    if scoring['production_ready']:
        print("\n🎯 SUCCESS - Le Detective validated for production deployment!")
    else:
        print("\n⚠️  Additional improvements needed before production")
    
    logger.info("Step 6.3 completed - Historical incident validation finished")

if __name__ == "__main__":
    main()