#!/usr/bin/env python3
"""
ML-Scheduler Integration Service
Integration between ML Predictor and Q-Learning Optimizer
Orchestrates the two expert AI services
Respect .claude_code_rules - No emojis
"""

from flask import Flask, request, jsonify
import requests
import json
import logging
import os
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Optional
import numpy as np

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class MLSchedulerIntegration:
    """Integration service combining ML Predictor and Q-Learning Optimizer"""
    
    def __init__(self, ml_predictor_url="http://localhost:5000", 
                 qlearning_url="http://localhost:5001"):
        self.ml_predictor_url = ml_predictor_url
        self.qlearning_url = qlearning_url
        self.session = requests.Session()
        self.session.timeout = 30
        
        # Service status
        self.predictor_ready = False
        self.qlearning_ready = False
        
        self.check_services_health()
    
    def check_services_health(self):
        """Check health status of both services"""
        logger.info("Checking ML services health...")
        
        # Check ML Predictor
        try:
            response = self.session.get(f"{self.ml_predictor_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.predictor_ready = data.get('status') == 'healthy'
                logger.info(f"ML Predictor: {'Ready' if self.predictor_ready else 'Not ready'}")
            else:
                self.predictor_ready = False
        except Exception as e:
            logger.warning(f"ML Predictor health check failed: {e}")
            self.predictor_ready = False
        
        # Check Q-Learning Optimizer
        try:
            response = self.session.get(f"{self.qlearning_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.qlearning_ready = data.get('status') == 'healthy'
                logger.info(f"Q-Learning Optimizer: {'Ready' if self.qlearning_ready else 'Not ready'}")
            else:
                self.qlearning_ready = False
        except Exception as e:
            logger.warning(f"Q-Learning Optimizer health check failed: {e}")
            self.qlearning_ready = False
    
    def get_load_predictions(self, node_features: Dict, horizons: List[str] = ['30m']) -> Dict:
        """Get load predictions from ML Predictor"""
        if not self.predictor_ready:
            raise RuntimeError("ML Predictor service not available")
        
        predictions = {}
        
        for horizon in horizons:
            try:
                # Use predict_load endpoint for node-specific predictions
                payload = {'node_features': node_features}
                
                response = self.session.post(
                    f"{self.ml_predictor_url}/predict_load",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    predictions[horizon] = {
                        'predicted_load': data.get('predicted_load', 0.5),
                        'load_level': data.get('load_level', 'MEDIUM'),
                        'recommendation': data.get('recommendation', 'MONITOR'),
                        'confidence': data.get('confidence', 0.5)
                    }
                else:
                    logger.warning(f"Prediction failed for {horizon}: {response.status_code}")
                    predictions[horizon] = {
                        'predicted_load': 0.5,
                        'load_level': 'UNKNOWN',
                        'recommendation': 'MONITOR',
                        'confidence': 0.3
                    }
                
            except Exception as e:
                logger.error(f"Error getting prediction for {horizon}: {e}")
                predictions[horizon] = {
                    'predicted_load': 0.5,
                    'load_level': 'UNKNOWN',
                    'recommendation': 'MONITOR',
                    'confidence': 0.3
                }
        
        return predictions
    
    def get_optimal_placement(self, cluster_state: Dict) -> Dict:
        """Get optimal placement from Q-Learning Optimizer"""
        if not self.qlearning_ready:
            raise RuntimeError("Q-Learning Optimizer service not available")
        
        try:
            payload = {'cluster_state': cluster_state}
            
            response = self.session.post(
                f"{self.qlearning_url}/optimize",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Q-Learning optimization failed: {response.status_code}")
                raise RuntimeError(f"Optimization failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error getting optimal placement: {e}")
            raise
    
    def enhance_cluster_state_with_predictions(self, cluster_state: Dict) -> Dict:
        """Enhance cluster state with ML Predictor forecasts"""
        enhanced_state = cluster_state.copy()
        
        for node_name, node_data in enhanced_state.items():
            try:
                # Get predictions for this node
                node_features = {
                    'node_name': node_name,
                    'cpu_rate': node_data.get('cpu_utilization', 0.1),
                    'memory_utilization': node_data.get('memory_utilization', 0.5),
                    'load1': node_data.get('load1', 1.0),
                    'load5': node_data.get('load5', 1.2),
                    'memory_capacity': node_data.get('memory_capacity', 16e9)
                }
                
                predictions = self.get_load_predictions(node_features, ['30m'])
                
                # Add predictions to node data
                if '30m' in predictions:
                    pred_data = predictions['30m']
                    enhanced_state[node_name].update({
                        'predicted_load_30m': pred_data['predicted_load'],
                        'load_level_30m': pred_data['load_level'],
                        'predictor_recommendation': pred_data['recommendation'],
                        'prediction_confidence': pred_data['confidence']
                    })
                
            except Exception as e:
                logger.warning(f"Failed to get predictions for {node_name}: {e}")
                # Add default predictions
                enhanced_state[node_name].update({
                    'predicted_load_30m': 0.5,
                    'load_level_30m': 'MEDIUM',
                    'predictor_recommendation': 'MONITOR',
                    'prediction_confidence': 0.3
                })
        
        return enhanced_state
    
    def make_integrated_decision(self, cluster_state: Dict, pod_requirements: Dict = None) -> Dict:
        """Make integrated scheduling decision using both AI experts"""
        logger.info("Making integrated ML scheduling decision")
        
        # Step 1: Enhance cluster state with predictions
        enhanced_state = self.enhance_cluster_state_with_predictions(cluster_state)
        
        # Step 2: Get Q-Learning optimal placement
        optimization_result = self.get_optimal_placement(enhanced_state)
        
        # Step 3: Analyze predictor recommendations
        predictor_analysis = self.analyze_predictor_recommendations(enhanced_state)
        
        # Step 4: Combine insights
        integrated_decision = {
            'recommended_node': optimization_result.get('optimal_node'),
            'confidence': optimization_result.get('confidence', 0.5),
            'qlearning_analysis': {
                'optimal_node': optimization_result.get('optimal_node'),
                'q_value': optimization_result.get('q_value', 0),
                'alternatives': optimization_result.get('alternatives', []),
                'reasoning': optimization_result.get('reasoning', {})
            },
            'predictor_analysis': predictor_analysis,
            'integrated_reasoning': self.generate_integrated_reasoning(
                optimization_result, predictor_analysis, enhanced_state
            ),
            'risk_assessment': self.assess_placement_risk(
                optimization_result.get('optimal_node'), enhanced_state
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        return integrated_decision
    
    def analyze_predictor_recommendations(self, enhanced_state: Dict) -> Dict:
        """Analyze ML Predictor recommendations across all nodes"""
        analysis = {
            'avoid_nodes': [],
            'preferred_nodes': [],
            'monitor_nodes': [],
            'high_confidence_predictions': [],
            'load_forecast_summary': {}
        }
        
        for node_name, node_data in enhanced_state.items():
            recommendation = node_data.get('predictor_recommendation', 'MONITOR')
            load_level = node_data.get('load_level_30m', 'MEDIUM')
            confidence = node_data.get('prediction_confidence', 0.5)
            predicted_load = node_data.get('predicted_load_30m', 0.5)
            
            # Categorize nodes based on predictions
            if recommendation == 'AVOID':
                analysis['avoid_nodes'].append(node_name)
            elif recommendation == 'OK' and load_level == 'LOW':
                analysis['preferred_nodes'].append(node_name)
            else:
                analysis['monitor_nodes'].append(node_name)
            
            # High confidence predictions
            if confidence > 0.7:
                analysis['high_confidence_predictions'].append({
                    'node': node_name,
                    'confidence': confidence,
                    'prediction': recommendation
                })
            
            # Load forecast summary
            analysis['load_forecast_summary'][node_name] = {
                'predicted_load': predicted_load,
                'load_level': load_level,
                'trend': 'increasing' if predicted_load > 0.7 else 'stable' if predicted_load > 0.4 else 'decreasing'
            }
        
        return analysis
    
    def generate_integrated_reasoning(self, qlearning_result: Dict, 
                                    predictor_analysis: Dict, 
                                    enhanced_state: Dict) -> Dict:
        """Generate integrated reasoning combining both AI experts"""
        recommended_node = qlearning_result.get('optimal_node')
        
        reasoning = {
            'decision_confidence': 'HIGH',
            'agreement_analysis': {},
            'risk_factors': [],
            'supporting_evidence': [],
            'alternative_considerations': []
        }
        
        # Check agreement between experts
        if recommended_node in predictor_analysis.get('avoid_nodes', []):
            reasoning['agreement_analysis'] = {
                'status': 'CONFLICTED',
                'details': 'Q-Learning recommends node that ML Predictor suggests avoiding'
            }
            reasoning['decision_confidence'] = 'MEDIUM'
            reasoning['risk_factors'].append('Predictor forecasts high load on recommended node')
        
        elif recommended_node in predictor_analysis.get('preferred_nodes', []):
            reasoning['agreement_analysis'] = {
                'status': 'ALIGNED',
                'details': 'Both experts agree on node selection'
            }
            reasoning['supporting_evidence'].append('ML Predictor forecasts favorable conditions')
            reasoning['supporting_evidence'].append('Q-Learning identifies optimal resource utilization')
        
        else:
            reasoning['agreement_analysis'] = {
                'status': 'NEUTRAL',
                'details': 'No strong conflict or agreement between experts'
            }
        
        # Add node-specific insights
        if recommended_node and recommended_node in enhanced_state:
            node_data = enhanced_state[recommended_node]
            
            # Current state analysis
            cpu_util = node_data.get('cpu_utilization', 0)
            memory_util = node_data.get('memory_utilization', 0)
            predicted_load = node_data.get('predicted_load_30m', 0.5)
            
            if cpu_util < 0.4 and memory_util < 0.5:
                reasoning['supporting_evidence'].append('Low current resource utilization')
            
            if predicted_load < 0.6:
                reasoning['supporting_evidence'].append('Favorable load forecast')
            elif predicted_load > 0.8:
                reasoning['risk_factors'].append('High predicted load in 30 minutes')
            
        # Alternative considerations
        alternatives = qlearning_result.get('alternatives', [])
        for alt in alternatives[:2]:  # Top 2 alternatives
            alt_node = alt.get('node')
            if alt_node in predictor_analysis.get('preferred_nodes', []):
                reasoning['alternative_considerations'].append(
                    f"{alt_node} also has favorable predictor forecast"
                )
        
        # Final confidence adjustment
        if len(reasoning['risk_factors']) > len(reasoning['supporting_evidence']):
            reasoning['decision_confidence'] = 'LOW'
        elif len(reasoning['supporting_evidence']) >= 2:
            reasoning['decision_confidence'] = 'HIGH'
        
        return reasoning
    
    def assess_placement_risk(self, recommended_node: str, enhanced_state: Dict) -> Dict:
        """Assess risk of placing pod on recommended node"""
        if not recommended_node or recommended_node not in enhanced_state:
            return {'risk_level': 'UNKNOWN', 'factors': ['Node data unavailable']}
        
        node_data = enhanced_state[recommended_node]
        risk_factors = []
        risk_score = 0
        
        # Current utilization risk
        cpu_util = node_data.get('cpu_utilization', 0)
        memory_util = node_data.get('memory_utilization', 0)
        
        if cpu_util > 0.8:
            risk_factors.append('High CPU utilization')
            risk_score += 30
        elif cpu_util > 0.6:
            risk_score += 15
        
        if memory_util > 0.8:
            risk_factors.append('High memory utilization')
            risk_score += 30
        elif memory_util > 0.6:
            risk_score += 15
        
        # Predicted load risk
        predicted_load = node_data.get('predicted_load_30m', 0.5)
        if predicted_load > 0.8:
            risk_factors.append('High predicted load')
            risk_score += 25
        elif predicted_load > 0.6:
            risk_score += 10
        
        # System load risk
        load1 = node_data.get('load1', 1.0)
        if load1 > 4.0:
            risk_factors.append('High system load')
            risk_score += 20
        
        # Reliability risk
        reliability = node_data.get('reliability_score', 100)
        if reliability < 95:
            risk_factors.append('Below average reliability')
            risk_score += 15
        
        # Determine risk level
        if risk_score >= 60:
            risk_level = 'HIGH'
        elif risk_score >= 30:
            risk_level = 'MEDIUM'
        elif risk_score >= 10:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'factors': risk_factors,
            'mitigation_suggestions': self.get_risk_mitigation_suggestions(risk_factors)
        }
    
    def get_risk_mitigation_suggestions(self, risk_factors: List[str]) -> List[str]:
        """Get risk mitigation suggestions"""
        suggestions = []
        
        if 'High CPU utilization' in risk_factors:
            suggestions.append('Consider CPU-light workload or scale horizontally')
        
        if 'High memory utilization' in risk_factors:
            suggestions.append('Monitor memory requirements and consider memory-optimized nodes')
        
        if 'High predicted load' in risk_factors:
            suggestions.append('Schedule placement for off-peak hours or use alternative node')
        
        if 'High system load' in risk_factors:
            suggestions.append('Wait for system load to decrease before placement')
        
        if not suggestions:
            suggestions.append('No specific mitigation required - monitor placement results')
        
        return suggestions

# Instance globale integration service
integration_service = MLSchedulerIntegration()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Re-check services
    integration_service.check_services_health()
    
    overall_status = 'healthy' if (integration_service.predictor_ready and integration_service.qlearning_ready) else 'unhealthy'
    
    return jsonify({
        'status': overall_status,
        'service': 'ml-scheduler-integration',
        'components': {
            'ml_predictor': 'ready' if integration_service.predictor_ready else 'unavailable',
            'qlearning_optimizer': 'ready' if integration_service.qlearning_ready else 'unavailable'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/info', methods=['GET'])
def service_info():
    """Service information endpoint"""
    return jsonify({
        'service_name': 'ml-scheduler-integration',
        'version': '1.0.0',
        'description': 'Integration service for ML Predictor and Q-Learning Optimizer',
        'ml_predictor_url': integration_service.ml_predictor_url,
        'qlearning_url': integration_service.qlearning_url,
        'services_ready': {
            'ml_predictor': integration_service.predictor_ready,
            'qlearning_optimizer': integration_service.qlearning_ready
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/schedule', methods=['POST'])
def schedule_pod():
    """Main scheduling endpoint using integrated AI decision"""
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'cluster_state' not in data:
            return jsonify({
                'error': 'Format incorrect - expected: {"cluster_state": {...}, "pod_requirements": {...}}'
            }), 400
        
        cluster_state = data['cluster_state']
        pod_requirements = data.get('pod_requirements', {})
        
        # Make integrated decision
        decision = integration_service.make_integrated_decision(cluster_state, pod_requirements)
        
        # Return response
        response = {
            'recommended_node': decision['recommended_node'],
            'confidence': decision['confidence'],
            'decision_type': 'integrated_ai',
            'experts_analysis': {
                'qlearning_optimizer': decision['qlearning_analysis'],
                'ml_predictor': decision['predictor_analysis']
            },
            'integrated_reasoning': decision['integrated_reasoning'],
            'risk_assessment': decision['risk_assessment'],
            'timestamp': decision['timestamp'],
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Scheduling error: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch_schedule', methods=['POST'])
def batch_schedule():
    """Batch scheduling for multiple pods"""
    try:
        data = request.get_json()
        
        if not data or 'requests' not in data:
            return jsonify({
                'error': 'Format incorrect - expected: {"requests": [...]}'
            }), 400
        
        requests = data['requests']
        batch_results = []
        
        for i, req in enumerate(requests):
            if 'cluster_state' not in req:
                batch_results.append({
                    'request_id': i,
                    'error': 'Missing cluster_state'
                })
                continue
            
            try:
                cluster_state = req['cluster_state']
                pod_requirements = req.get('pod_requirements', {})
                
                decision = integration_service.make_integrated_decision(cluster_state, pod_requirements)
                
                batch_results.append({
                    'request_id': i,
                    'recommended_node': decision['recommended_node'],
                    'confidence': decision['confidence'],
                    'risk_level': decision['risk_assessment']['risk_level'],
                    'status': 'success'
                })
                
            except Exception as e:
                batch_results.append({
                    'request_id': i,
                    'error': str(e),
                    'status': 'error'
                })
        
        return jsonify({
            'batch_results': batch_results,
            'total_requests': len(requests),
            'successful': len([r for r in batch_results if r.get('status') == 'success']),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch scheduling error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Check if both services are available
    if not integration_service.predictor_ready or not integration_service.qlearning_ready:
        logger.warning("Not all ML services are ready - some features may be limited")
    
    logger.info("Starting ML-Scheduler Integration Service")
    logger.info(f"ML Predictor: {'Ready' if integration_service.predictor_ready else 'Unavailable'}")
    logger.info(f"Q-Learning Optimizer: {'Ready' if integration_service.qlearning_ready else 'Unavailable'}")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5002, debug=False)