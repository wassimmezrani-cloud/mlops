#!/usr/bin/env python3
"""
Trio ML-Scheduler Service - Complete Integration
XGBoost + Q-Learning + Isolation Forest
Revolutionary Kubernetes Scheduler with Three AI Experts
Step 6.4: Complete trio architecture deployment
"""

import os
import json
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrioMLSchedulerService:
    """
    Complete ML-Scheduler with three AI experts:
    1. XGBoost "Le Prophète" - Future load prediction
    2. Q-Learning "L'Optimiseur" - Optimal placement optimization  
    3. Isolation Forest "Le Détective" - Anomaly detection
    """
    
    def __init__(self):
        """Initialize trio ML-Scheduler service"""
        
        # Expert service endpoints
        self.expert_endpoints = {
            'xgboost': 'http://localhost:8080',      # Le Prophète
            'qlearning': 'http://localhost:8081',    # L'Optimiseur  
            'isolation': 'http://localhost:8083'     # Le Détective
        }
        
        # Trio configuration
        self.trio_config = {
            'decision_weights': {
                'prediction_weight': 0.35,    # XGBoost future prediction weight
                'optimization_weight': 0.35,  # Q-Learning optimization weight
                'anomaly_weight': 0.30        # Isolation Forest safety weight
            },
            'risk_thresholds': {
                'high_risk': 0.8,
                'medium_risk': 0.6,
                'low_risk': 0.4
            },
            'consensus_requirements': {
                'minimum_experts': 2,  # At least 2 experts must agree
                'anomaly_veto': True   # Anomaly detection can veto placement
            }
        }
        
        self.service_info = {
            'name': 'Trio ML-Scheduler',
            'version': '1.0.0',
            'experts': ['Le Prophète', "L'Optimiseur", 'Le Détective'],
            'architecture': 'Three-Expert AI Fusion',
            'deployment_date': datetime.now().isoformat()
        }
        
        logger.info("Trio ML-Scheduler Service initialized")
        logger.info(f"Expert endpoints: {self.expert_endpoints}")
    
    def check_experts_health(self) -> Dict[str, bool]:
        """Check health status of all three experts"""
        health_status = {}
        
        for expert, endpoint in self.expert_endpoints.items():
            try:
                response = requests.get(f"{endpoint}/health", timeout=5)
                health_status[expert] = response.status_code == 200
            except Exception as e:
                logger.warning(f"{expert} health check failed: {e}")
                health_status[expert] = False
        
        return health_status
    
    def call_xgboost_predictor(self, features: List[float], 
                              horizon: str = "1h", 
                              metric: str = "cpu") -> Optional[Dict]:
        """
        Call XGBoost Le Prophète for future load prediction
        
        Args:
            features: Node feature vector
            horizon: Prediction horizon (30min, 1h, 2h)
            metric: Metric type (cpu, memory)
            
        Returns:
            XGBoost prediction result
        """
        try:
            payload = {
                "instances": [{
                    "horizon": horizon,
                    "metric": metric,
                    "features": features
                }]
            }
            
            response = requests.post(
                f"{self.expert_endpoints['xgboost']}/v1/models/xgboost-predictor:predict",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'predictions' in result and len(result['predictions']) > 0:
                    return result['predictions'][0]
                    
        except Exception as e:
            logger.error(f"XGBoost prediction failed: {e}")
        
        return None
    
    def call_qlearning_optimizer(self, cluster_state: Dict, 
                                pod_requirements: Dict) -> Optional[Dict]:
        """
        Call Q-Learning L'Optimiseur for optimal placement
        
        Args:
            cluster_state: Current cluster state
            pod_requirements: Pod resource requirements
            
        Returns:
            Q-Learning optimization result
        """
        try:
            payload = {
                "instances": [{
                    "cluster_state": cluster_state,
                    "pod_requirements": pod_requirements
                }]
            }
            
            response = requests.post(
                f"{self.expert_endpoints['qlearning']}/v1/models/qlearning-optimizer:predict",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'predictions' in result and len(result['predictions']) > 0:
                    return result['predictions'][0]
                    
        except Exception as e:
            logger.error(f"Q-Learning optimization failed: {e}")
        
        return None
    
    def call_isolation_detector(self, node_data: Dict) -> Optional[Dict]:
        """
        Call Isolation Forest Le Détective for anomaly detection
        
        Args:
            node_data: Node behavioral data
            
        Returns:
            Anomaly detection result
        """
        try:
            payload = {
                "instances": [node_data]
            }
            
            response = requests.post(
                f"{self.expert_endpoints['isolation']}/v1/models/isolation-detector:predict",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'predictions' in result and len(result['predictions']) > 0:
                    return result['predictions'][0]
                    
        except Exception as e:
            logger.error(f"Isolation detection failed: {e}")
        
        return None
    
    def analyze_node_with_trio(self, node_id: str, 
                              node_data: Dict, 
                              pod_requirements: Dict) -> Dict[str, Any]:
        """
        Complete node analysis using all three experts
        
        Args:
            node_id: Node identifier
            node_data: Node metrics and state
            pod_requirements: Pod resource requirements
            
        Returns:
            Complete trio analysis result
        """
        analysis_start = datetime.now()
        
        # Initialize expert results
        expert_results = {
            'xgboost': None,
            'qlearning': None,
            'isolation': None
        }
        
        # Generate features for XGBoost (simplified)
        current_cpu = node_data.get('cpu_utilization', [0.3])[-1]
        current_memory = node_data.get('memory_utilization', [0.4])[-1]
        current_load = node_data.get('load_average', [1.5])[-1]
        pod_count = node_data.get('pod_count', [15])[-1]
        
        node_features = [
            current_cpu, current_memory, current_load, pod_count,
            pod_requirements.get('cpu_request', 0.1),
            pod_requirements.get('memory_request', 0.2),
            datetime.now().hour, datetime.now().weekday(),
        ] + [0.0] * 20  # Pad to expected feature count
        
        # Expert 1: XGBoost Future Prediction
        cpu_prediction = self.call_xgboost_predictor(node_features, "1h", "cpu")
        memory_prediction = self.call_xgboost_predictor(node_features, "1h", "memory")
        
        if cpu_prediction and memory_prediction:
            expert_results['xgboost'] = {
                'cpu_prediction': cpu_prediction.get('prediction', current_cpu),
                'memory_prediction': memory_prediction.get('prediction', current_memory),
                'confidence': min(cpu_prediction.get('confidence', 0.5),
                               memory_prediction.get('confidence', 0.5)),
                'status': 'SUCCESS'
            }
        else:
            expert_results['xgboost'] = {'status': 'FAILED'}
        
        # Expert 2: Q-Learning Optimization
        cluster_state = {
            node_id: {
                'cpu_utilization': current_cpu,
                'memory_utilization': current_memory,
                'load_average': current_load,
                'pod_count': pod_count
            }
        }
        
        qlearning_result = self.call_qlearning_optimizer(cluster_state, pod_requirements)
        if qlearning_result:
            expert_results['qlearning'] = {
                'recommended_node': qlearning_result.get('recommended_node', node_id),
                'optimization_score': qlearning_result.get('optimization_score', 0.5),
                'confidence': qlearning_result.get('confidence', 0.5),
                'status': 'SUCCESS'
            }
        else:
            expert_results['qlearning'] = {'status': 'FAILED'}
        
        # Expert 3: Isolation Forest Anomaly Detection
        isolation_result = self.call_isolation_detector(node_data)
        if isolation_result:
            expert_results['isolation'] = {
                'is_anomaly': isolation_result.get('is_anomaly', False),
                'anomaly_score': isolation_result.get('anomaly_score', 0.0),
                'risk_level': isolation_result.get('risk_level', 'LOW'),
                'confidence': isolation_result.get('confidence', 0.5),
                'contributing_factors': isolation_result.get('contributing_factors', []),
                'status': 'SUCCESS'
            }
        else:
            expert_results['isolation'] = {'status': 'FAILED'}
        
        # Trio Decision Fusion
        trio_decision = self.fuse_trio_decisions(node_id, expert_results, pod_requirements)
        
        processing_time = (datetime.now() - analysis_start).total_seconds() * 1000
        
        return {
            'node_id': node_id,
            'expert_results': expert_results,
            'trio_decision': trio_decision,
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }
    
    def fuse_trio_decisions(self, node_id: str, 
                           expert_results: Dict[str, Any],
                           pod_requirements: Dict) -> Dict[str, Any]:
        """
        Intelligent fusion of all three expert decisions
        
        Args:
            node_id: Node identifier
            expert_results: Results from all three experts
            pod_requirements: Pod requirements
            
        Returns:
            Fused trio decision
        """
        weights = self.trio_config['decision_weights']
        
        # Count available experts
        available_experts = sum(1 for result in expert_results.values() 
                               if result.get('status') == 'SUCCESS')
        
        if available_experts < self.trio_config['consensus_requirements']['minimum_experts']:
            return {
                'decision': 'REJECT',
                'reason': f'Insufficient experts available ({available_experts}/3)',
                'confidence': 0.2,
                'risk_assessment': 'HIGH'
            }
        
        # Anomaly Detection Veto
        isolation_result = expert_results.get('isolation', {})
        if (isolation_result.get('status') == 'SUCCESS' and 
            isolation_result.get('is_anomaly') and
            self.trio_config['consensus_requirements']['anomaly_veto']):
            
            return {
                'decision': 'REJECT',
                'reason': f"Node anomaly detected: {isolation_result.get('risk_level', 'UNKNOWN')}",
                'confidence': isolation_result.get('confidence', 0.8),
                'risk_assessment': 'HIGH',
                'anomaly_factors': isolation_result.get('contributing_factors', [])[:3]
            }
        
        # Weighted Scoring
        total_score = 0.0
        total_weight = 0.0
        confidence_scores = []
        
        # XGBoost Future Safety Score
        xgboost_result = expert_results.get('xgboost', {})
        if xgboost_result.get('status') == 'SUCCESS':
            # Score based on predicted resource utilization
            cpu_pred = xgboost_result.get('cpu_prediction', 0.5)
            memory_pred = xgboost_result.get('memory_prediction', 0.5)
            
            # Higher utilization = lower score
            future_safety_score = max(0, 1.0 - max(cpu_pred, memory_pred))
            total_score += future_safety_score * weights['prediction_weight']
            total_weight += weights['prediction_weight']
            confidence_scores.append(xgboost_result.get('confidence', 0.5))
        
        # Q-Learning Optimization Score
        qlearning_result = expert_results.get('qlearning', {})
        if qlearning_result.get('status') == 'SUCCESS':
            optimization_score = qlearning_result.get('optimization_score', 0.5)
            total_score += optimization_score * weights['optimization_weight']
            total_weight += weights['optimization_weight']
            confidence_scores.append(qlearning_result.get('confidence', 0.5))
        
        # Isolation Forest Safety Score
        if isolation_result.get('status') == 'SUCCESS':
            # Higher anomaly score = lower safety score
            anomaly_score = isolation_result.get('anomaly_score', 0.0)
            safety_score = max(0, 1.0 + anomaly_score)  # Anomaly scores are negative for normal
            total_score += safety_score * weights['anomaly_weight']
            total_weight += weights['anomaly_weight']
            confidence_scores.append(isolation_result.get('confidence', 0.5))
        
        # Final Decision
        if total_weight > 0:
            final_score = total_score / total_weight
            average_confidence = np.mean(confidence_scores) if confidence_scores else 0.3
        else:
            final_score = 0.3
            average_confidence = 0.3
        
        # Risk Assessment
        risk_thresholds = self.trio_config['risk_thresholds']
        if final_score >= risk_thresholds['high_risk']:
            risk_level = 'LOW'
            decision = 'ACCEPT'
        elif final_score >= risk_thresholds['medium_risk']:
            risk_level = 'MEDIUM'
            decision = 'ACCEPT'
        elif final_score >= risk_thresholds['low_risk']:
            risk_level = 'MEDIUM'
            decision = 'CONDITIONAL'
        else:
            risk_level = 'HIGH'
            decision = 'REJECT'
        
        return {
            'decision': decision,
            'final_score': final_score,
            'confidence': average_confidence,
            'risk_assessment': risk_level,
            'expert_consensus': f"{available_experts}/3 experts available",
            'weighted_scoring': {
                'total_score': total_score,
                'total_weight': total_weight,
                'weights_used': weights
            },
            'reasoning': self._generate_decision_reasoning(expert_results, final_score)
        }
    
    def _generate_decision_reasoning(self, expert_results: Dict[str, Any], 
                                   final_score: float) -> List[str]:
        """Generate human-readable decision reasoning"""
        reasoning = []
        
        # XGBoost reasoning
        xgb = expert_results.get('xgboost', {})
        if xgb.get('status') == 'SUCCESS':
            cpu_pred = xgb.get('cpu_prediction', 0.5)
            memory_pred = xgb.get('memory_prediction', 0.5)
            reasoning.append(f"Le Prophète predicts: CPU {cpu_pred:.1%}, Memory {memory_pred:.1%} in 1h")
        
        # Q-Learning reasoning
        ql = expert_results.get('qlearning', {})
        if ql.get('status') == 'SUCCESS':
            opt_score = ql.get('optimization_score', 0.5)
            reasoning.append(f"L'Optimiseur optimization score: {opt_score:.2f}")
        
        # Isolation reasoning
        iso = expert_results.get('isolation', {})
        if iso.get('status') == 'SUCCESS':
            if iso.get('is_anomaly'):
                reasoning.append(f"Le Détective: ANOMALY detected ({iso.get('risk_level', 'UNKNOWN')})")
            else:
                reasoning.append("Le Détective: Node behavior normal")
        
        reasoning.append(f"Trio consensus score: {final_score:.2f}")
        
        return reasoning
    
    def schedule_pod(self, pod_spec: Dict, cluster_nodes: List[Dict]) -> Dict[str, Any]:
        """
        Complete pod scheduling using trio AI experts
        
        Args:
            pod_spec: Pod specification and requirements
            cluster_nodes: List of available cluster nodes
            
        Returns:
            Scheduling decision with complete analysis
        """
        scheduling_start = datetime.now()
        
        pod_requirements = {
            'cpu_request': pod_spec.get('cpu_request', 0.1),
            'memory_request': pod_spec.get('memory_request', 0.2),
            'pod_type': pod_spec.get('pod_type', 'web')
        }
        
        # Analyze each candidate node
        node_analyses = []
        for node_data in cluster_nodes:
            node_id = node_data.get('node_id', f"node_{len(node_analyses)}")
            analysis = self.analyze_node_with_trio(node_id, node_data, pod_requirements)
            node_analyses.append(analysis)
        
        # Select best node based on trio decisions
        best_node = None
        best_score = -1.0
        
        for analysis in node_analyses:
            trio_decision = analysis['trio_decision']
            if trio_decision['decision'] in ['ACCEPT', 'CONDITIONAL']:
                score = trio_decision['final_score']
                if score > best_score:
                    best_score = score
                    best_node = analysis
        
        # Final scheduling result
        scheduling_time = (datetime.now() - scheduling_start).total_seconds() * 1000
        
        result = {
            'pod_spec': pod_spec,
            'scheduling_result': {
                'selected_node': best_node['node_id'] if best_node else None,
                'decision': best_node['trio_decision']['decision'] if best_node else 'REJECT',
                'confidence': best_node['trio_decision']['confidence'] if best_node else 0.0,
                'final_score': best_node['trio_decision']['final_score'] if best_node else 0.0
            },
            'node_analyses': node_analyses,
            'scheduling_metrics': {
                'total_nodes_analyzed': len(node_analyses),
                'acceptable_nodes': len([a for a in node_analyses 
                                       if a['trio_decision']['decision'] in ['ACCEPT', 'CONDITIONAL']]),
                'processing_time_ms': scheduling_time
            },
            'trio_experts_used': self.service_info['experts'],
            'timestamp': datetime.now().isoformat()
        }
        
        return result

# Flask application
app = Flask(__name__)
trio_service = TrioMLSchedulerService()

@app.route('/health', methods=['GET'])
def health():
    """Health check for trio service"""
    experts_health = trio_service.check_experts_health()
    healthy_experts = sum(experts_health.values())
    
    return jsonify({
        'status': 'healthy' if healthy_experts >= 2 else 'degraded',
        'service': 'trio-ml-scheduler',
        'experts_health': experts_health,
        'healthy_experts': f"{healthy_experts}/3",
        'timestamp': datetime.now().isoformat()
    })

@app.route('/v1/models/trio-scheduler:predict', methods=['POST'])
def schedule():
    """Main scheduling endpoint"""
    try:
        data = request.get_json()
        
        if 'instances' in data:
            # KServe format
            instance = data['instances'][0]
            pod_spec = instance.get('pod_spec', {})
            cluster_nodes = instance.get('cluster_nodes', [])
        else:
            # Direct format
            pod_spec = data.get('pod_spec', {})
            cluster_nodes = data.get('cluster_nodes', [])
        
        # Perform trio scheduling
        result = trio_service.schedule_pod(pod_spec, cluster_nodes)
        
        return jsonify({
            'predictions': [result],
            'model_name': 'trio-scheduler',
            'model_version': trio_service.service_info['version']
        })
        
    except Exception as e:
        logger.error(f"Scheduling error: {e}")
        return jsonify({
            'error': str(e),
            'model_name': 'trio-scheduler'
        }), 400

@app.route('/v1/models/trio-scheduler', methods=['GET'])
def model_info():
    """Trio scheduler information"""
    experts_health = trio_service.check_experts_health()
    
    return jsonify({
        'name': 'trio-scheduler',
        'versions': [trio_service.service_info['version']],
        'description': 'Three-Expert AI Kubernetes Scheduler',
        'experts': {
            'xgboost': {
                'name': 'Le Prophète',
                'role': 'Future load prediction',
                'healthy': experts_health.get('xgboost', False)
            },
            'qlearning': {
                'name': "L'Optimiseur",
                'role': 'Optimal placement optimization',
                'healthy': experts_health.get('qlearning', False)
            },
            'isolation': {
                'name': 'Le Détective',
                'role': 'Anomaly detection',
                'healthy': experts_health.get('isolation', False)
            }
        },
        'architecture': trio_service.service_info['architecture'],
        'fusion_config': trio_service.trio_config,
        'ready': sum(experts_health.values()) >= 2
    })

@app.route('/analyze_node', methods=['POST'])
def analyze_node():
    """Direct node analysis endpoint"""
    try:
        data = request.get_json()
        node_id = data.get('node_id', 'unknown')
        node_data = data.get('node_data', {})
        pod_requirements = data.get('pod_requirements', {})
        
        analysis = trio_service.analyze_node_with_trio(node_id, node_data, pod_requirements)
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def main():
    """Main trio service entry point"""
    print("="*70)
    print("TRIO ML-SCHEDULER SERVICE - REVOLUTIONARY AI SCHEDULER")
    print("Three-Expert AI Fusion for Kubernetes Pod Placement")
    print("="*70)
    print(f"Service: {trio_service.service_info['name']}")
    print(f"Version: {trio_service.service_info['version']}")
    print(f"Architecture: {trio_service.service_info['architecture']}")
    print("\nAI Experts:")
    print("  1. XGBoost 'Le Prophète' - Future load prediction")
    print("  2. Q-Learning 'L'Optimiseur' - Optimal placement")
    print("  3. Isolation Forest 'Le Détective' - Anomaly detection")
    print("\nEndpoints:")
    print("  • POST /v1/models/trio-scheduler:predict (Main scheduling)")
    print("  • GET  /v1/models/trio-scheduler (Service info)")
    print("  • GET  /health (Health check)")
    print("  • POST /analyze_node (Node analysis)")
    print("="*70)
    print("🚀 Trio ML-Scheduler ready on port 8084")
    
    logger.info("Starting Trio ML-Scheduler Service")
    logger.info("Revolutionary three-expert AI scheduling system")
    
    app.run(host='0.0.0.0', port=8084, debug=False)

if __name__ == "__main__":
    main()