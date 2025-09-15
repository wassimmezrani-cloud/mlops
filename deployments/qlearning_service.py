#!/usr/bin/env python3
"""
Q-Learning Optimizer Service API
Service production pour Q-Learning Second Expert
Respect .claude_code_rules - No emojis
"""

from flask import Flask, request, jsonify
import numpy as np
import json
import logging
import os
from datetime import datetime
import traceback
from collections import defaultdict

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class QLearningService:
    """Service API pour Q-Learning Optimizer"""
    
    def __init__(self, model_path="./data/models/qlearning_optimizer"):
        self.model_path = model_path
        self.q_table = defaultdict(lambda: np.zeros(6))  # 6 nodes default
        self.metadata = None
        self.node_mapping = None
        self.is_ready = False
        
        self.load_model()
    
    def load_model(self):
        """Charger modele Q-Learning et metadonnees"""
        try:
            # Charger Q-table
            model_file = f"{self.model_path}/qlearning_model.json"
            if os.path.exists(model_file):
                with open(model_file, 'r') as f:
                    model_data = json.load(f)
                
                # Reconstituer Q-table
                q_table_data = model_data.get('q_table', {})
                action_size = model_data.get('action_size', 6)
                
                self.q_table = defaultdict(lambda: np.zeros(action_size))
                for state_key, q_values in q_table_data.items():
                    self.q_table[state_key] = np.array(q_values)
                
                logger.info(f"Q-table loaded with {len(self.q_table)} states")
            else:
                raise FileNotFoundError(f"Model not found: {model_file}")
            
            # Charger metadonnees
            metadata_file = f"{self.model_path}/metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded: {self.metadata['model_name']}")
            
            # Charger mapping nodes
            mapping_file = f"{self.model_path}/node_mapping.json"
            if os.path.exists(mapping_file):
                with open(mapping_file, 'r') as f:
                    self.node_mapping = json.load(f)
                logger.info(f"Node mapping loaded: {len(self.node_mapping['nodes'])} nodes")
            else:
                # Default mapping
                self.node_mapping = {
                    'nodes': ['master1', 'master2', 'master3', 'worker1', 'worker2', 'worker3'],
                    'action_to_node': {i: f"node_{i}" for i in range(6)},
                    'node_to_action': {f"node_{i}": i for i in range(6)}
                }
            
            self.is_ready = True
            logger.info("Q-Learning service ready")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_ready = False
            raise
    
    def state_to_key(self, state: np.ndarray) -> str:
        """Convert state vector to Q-table key"""
        discretized = np.round(state * 10) / 10
        return str(discretized.tolist())
    
    def create_state_vector(self, cluster_state: dict) -> np.ndarray:
        """Create state vector from cluster state"""
        state_vector = []
        
        nodes = self.node_mapping['nodes']
        
        for node in nodes:
            node_data = cluster_state.get(node, {})
            
            # Extract metrics
            cpu_util = node_data.get('cpu_utilization', 0.1)
            memory_util = node_data.get('memory_utilization', 0.5)
            load1 = node_data.get('load1', 1.0)
            load5 = node_data.get('load5', 1.2)
            pod_count = node_data.get('pod_count', 10)
            reliability = node_data.get('reliability_score', 100) / 100
            
            # Calculate resource pressure
            resource_pressure = (cpu_util + memory_util) / 2
            
            # Add to state vector (7 features per node)
            state_vector.extend([
                cpu_util, memory_util, load1, load5,
                pod_count / 50,  # Normalized
                reliability,
                resource_pressure
            ])
        
        return np.array(state_vector)
    
    def select_optimal_node(self, cluster_state: dict) -> dict:
        """Select optimal node using Q-Learning policy"""
        if not self.is_ready:
            raise RuntimeError("Service not ready")
        
        # Create state vector
        state_vector = self.create_state_vector(cluster_state)
        state_key = self.state_to_key(state_vector)
        
        # Get Q-values for current state
        q_values = self.q_table[state_key]
        
        # Select action with highest Q-value
        optimal_action = np.argmax(q_values)
        
        # Map action to node
        nodes = self.node_mapping['nodes']
        if optimal_action < len(nodes):
            optimal_node = nodes[optimal_action]
        else:
            optimal_node = nodes[0]  # Fallback
        
        # Calculate confidence
        max_q = np.max(q_values)
        min_q = np.min(q_values)
        confidence = (max_q - min_q) / (max_q + 1e-8) if max_q != min_q else 0.5
        confidence = min(max(confidence, 0.1), 0.99)
        
        # Get alternative nodes
        sorted_indices = np.argsort(q_values)[::-1]
        alternatives = []
        
        for i in range(1, min(4, len(sorted_indices))):  # Top 3 alternatives
            alt_action = sorted_indices[i]
            if alt_action < len(nodes):
                alternatives.append({
                    'node': nodes[alt_action],
                    'q_value': float(q_values[alt_action]),
                    'confidence': float(q_values[alt_action] / max_q) if max_q > 0 else 0.5
                })
        
        return {
            'optimal_node': optimal_node,
            'optimal_action': int(optimal_action),
            'q_value': float(max_q),
            'confidence': confidence,
            'alternatives': alternatives,
            'reasoning': self.explain_decision(cluster_state, optimal_node)
        }
    
    def explain_decision(self, cluster_state: dict, selected_node: str) -> dict:
        """Explain why this node was selected"""
        node_data = cluster_state.get(selected_node, {})
        
        reasons = []
        
        # CPU utilization analysis
        cpu_util = node_data.get('cpu_utilization', 0.1)
        if cpu_util < 0.30:
            reasons.append("Low CPU utilization - good capacity")
        elif cpu_util > 0.80:
            reasons.append("High CPU utilization - consider alternatives")
        else:
            reasons.append("Balanced CPU utilization")
        
        # Memory analysis
        memory_util = node_data.get('memory_utilization', 0.5)
        if memory_util < 0.40:
            reasons.append("Low memory usage - good capacity")
        elif memory_util > 0.80:
            reasons.append("High memory usage - monitor closely")
        else:
            reasons.append("Balanced memory utilization")
        
        # Load analysis
        load1 = node_data.get('load1', 1.0)
        if load1 < 2.0:
            reasons.append("Low system load")
        elif load1 > 4.0:
            reasons.append("High system load")
        else:
            reasons.append("Moderate system load")
        
        # Reliability
        reliability = node_data.get('reliability_score', 100)
        if reliability >= 98:
            reasons.append("High reliability score")
        
        return {
            'node_metrics': {
                'cpu_utilization': cpu_util,
                'memory_utilization': memory_util,
                'load1': load1,
                'reliability': reliability
            },
            'decision_factors': reasons,
            'recommendation_strength': 'HIGH' if len([r for r in reasons if 'good' in r or 'Low' in r]) >= 2 else 'MEDIUM'
        }
    
    def get_service_info(self):
        """Get service information"""
        return {
            'service_name': 'qlearning-optimizer',
            'model_name': self.metadata.get('model_name', 'unknown') if self.metadata else 'unknown',
            'model_type': self.metadata.get('model_type', 'Q-Learning') if self.metadata else 'Q-Learning',
            'version': self.metadata.get('version', '1.0.0') if self.metadata else '1.0.0',
            'state_size': self.metadata.get('state_size', 42) if self.metadata else 42,
            'action_size': self.metadata.get('action_size', 6) if self.metadata else 6,
            'nodes_available': len(self.node_mapping['nodes']) if self.node_mapping else 6,
            'q_table_states': len(self.q_table),
            'is_ready': self.is_ready,
            'timestamp': datetime.now().isoformat()
        }

# Instance globale service
qlearning_service = QLearningService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if qlearning_service.is_ready else 'unhealthy',
        'service': 'qlearning-optimizer',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/info', methods=['GET'])
def service_info():
    """Service information endpoint"""
    try:
        return jsonify(qlearning_service.get_service_info())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/optimize', methods=['POST'])
def optimize_placement():
    """Main optimization endpoint"""
    try:
        # Parse request
        data = request.get_json()
        
        if not data or 'cluster_state' not in data:
            return jsonify({
                'error': 'Format incorrect - expected: {"cluster_state": {...}}'
            }), 400
        
        cluster_state = data['cluster_state']
        
        # Optimize placement
        optimization_result = qlearning_service.select_optimal_node(cluster_state)
        
        # Return result
        response = {
            'optimal_node': optimization_result['optimal_node'],
            'confidence': optimization_result['confidence'],
            'q_value': optimization_result['q_value'],
            'alternatives': optimization_result['alternatives'],
            'reasoning': optimization_result['reasoning'],
            'model_info': {
                'name': qlearning_service.metadata.get('model_name') if qlearning_service.metadata else 'qlearning-optimizer',
                'version': qlearning_service.metadata.get('version') if qlearning_service.metadata else '1.0.0'
            },
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/batch_optimize', methods=['POST'])
def batch_optimize():
    """Batch optimization for multiple placements"""
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
                result = qlearning_service.select_optimal_node(req['cluster_state'])
                batch_results.append({
                    'request_id': i,
                    'optimal_node': result['optimal_node'],
                    'confidence': result['confidence'],
                    'q_value': result['q_value'],
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
        logger.error(f"Batch optimization error: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/nodes', methods=['GET'])
def get_available_nodes():
    """Get available nodes for placement"""
    if not qlearning_service.node_mapping:
        return jsonify({'error': 'Node mapping not available'}), 500
    
    return jsonify({
        'nodes': qlearning_service.node_mapping['nodes'],
        'total_nodes': len(qlearning_service.node_mapping['nodes']),
        'action_mapping': qlearning_service.node_mapping.get('action_to_node', {})
    })

@app.route('/performance', methods=['GET'])
def get_performance_metrics():
    """Get Q-Learning performance metrics"""
    try:
        validation_file = f"{qlearning_service.model_path}/validation_results.json"
        
        if os.path.exists(validation_file):
            with open(validation_file, 'r') as f:
                validation_data = json.load(f)
            
            # Extract key metrics
            performance = {
                'global_improvement': validation_data['performance_metrics']['improvements']['global_improvement'],
                'success_rate': validation_data['validation_summary']['success_rate'],
                'efficiency': validation_data['validation_summary']['efficiency'],
                'business_score': validation_data['business_analysis']['global_score'],
                'status': validation_data['business_analysis']['status'],
                'production_ready': validation_data['business_analysis']['production_ready']
            }
            
            return jsonify(performance)
        else:
            return jsonify({'error': 'Performance metrics not available'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Verification readiness service
    if not qlearning_service.is_ready:
        logger.error("Service not ready - stopping")
        exit(1)
    
    logger.info("Starting Q-Learning Optimizer Service")
    logger.info(f"Model: {qlearning_service.metadata.get('model_name') if qlearning_service.metadata else 'unknown'}")
    logger.info(f"Q-table states: {len(qlearning_service.q_table)}")
    logger.info(f"Available nodes: {len(qlearning_service.node_mapping['nodes']) if qlearning_service.node_mapping else 0}")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5001, debug=False)