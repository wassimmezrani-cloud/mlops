#!/usr/bin/env python3
"""
Q-Learning KServe Service
Déploiement production-ready avec KServe
API <100ms latency pour ML-Scheduler
Respect .claude_code_rules - No emojis
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Any

# KServe dependencies
try:
    from kserve import InferenceService, Model, ModelServer, logging
    KSERVE_AVAILABLE = True
    print("KServe available - production deployment ready")
except ImportError:
    # Fallback to Flask for development
    from flask import Flask, request, jsonify
    KSERVE_AVAILABLE = False
    print("KServe not available - using Flask fallback")

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QLearningOptimizerModel:
    """Q-Learning Optimizer Model for KServe/Flask"""
    
    def __init__(self, model_path: str = "./models/qlearning_optimizer"):
        self.model_path = model_path
        self.q_table = None
        self.params = {}
        self.metadata = {}
        self.is_ready = False
        
        self.load_models()
    
    def load_models(self):
        """Load Q-Learning models and metadata"""
        try:
            # Load Q-table
            q_table_file = f"{self.model_path}/q_table.npy"
            if os.path.exists(q_table_file):
                self.q_table = np.load(q_table_file)
                logger.info(f"Q-table loaded: {self.q_table.shape}")
            
            # Load parameters
            params_file = f"{self.model_path}/params.json"
            if os.path.exists(params_file):
                with open(params_file, 'r') as f:
                    self.params = json.load(f)
                logger.info(f"Parameters loaded: {self.params.get('algorithm', 'Unknown')}")
            
            # Load final results as metadata
            results_file = f"{self.model_path}/final_results.json"
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Results loaded: {self.metadata.get('business_score', 0):.1f}/100")
            
            self.is_ready = self.q_table is not None and len(self.params) > 0
            logger.info(f"Q-Learning Optimizer loaded: {self.is_ready}")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_ready = False
            raise
    
    def encode_state(self, cluster_load: int, pod_type: str) -> int:
        """
        Encode cluster state and pod type into MDP state
        
        Args:
            cluster_load: 0=LOW, 1=MEDIUM, 2=HIGH
            pod_type: 'web', 'db', 'worker', 'ml'
            
        Returns:
            state: Encoded state index
        """
        pod_types = ['web', 'db', 'worker', 'ml']
        pod_type_idx = pod_types.index(pod_type) if pod_type in pod_types else 0
        
        pod_types_count = self.params.get('pod_types', 4)
        state = cluster_load * pod_types_count + pod_type_idx
        
        states_count = self.params.get('states', 12)
        return min(state, states_count - 1)
    
    def predict(self, inputs: Dict) -> Dict:
        """
        Make placement recommendations using Q-Learning
        Input format: {"cluster_load": "medium", "pod_type": "web", "available_nodes": [...]}
        """
        if not self.is_ready:
            raise RuntimeError("Models not ready")
        
        # Validate input format
        if not isinstance(inputs, dict):
            raise ValueError("Input must be a dictionary")
        
        cluster_load_str = inputs.get('cluster_load', 'low')
        pod_type = inputs.get('pod_type', 'web')
        available_nodes = inputs.get('available_nodes', list(range(5)))
        
        # Convert cluster load to numeric
        load_mapping = {'low': 0, 'medium': 1, 'high': 2}
        cluster_load = load_mapping.get(cluster_load_str.lower(), 0)
        
        # Encode state
        state = self.encode_state(cluster_load, pod_type)
        
        # Get Q-values for this state
        if state >= len(self.q_table):
            state = 0  # Fallback to first state
        
        q_values = self.q_table[state]
        
        # Select best action (node)
        if len(available_nodes) == 0:
            available_nodes = list(range(self.params.get('actions', 5)))
        
        # Filter Q-values for available nodes only
        available_q_values = []
        valid_nodes = []
        
        for node in available_nodes:
            if node < len(q_values):
                available_q_values.append(q_values[node])
                valid_nodes.append(node)
        
        if len(available_q_values) == 0:
            # Fallback: select first available node
            recommended_node = available_nodes[0] if available_nodes else 0
            confidence = 0.5
        else:
            # Select node with highest Q-value
            best_idx = np.argmax(available_q_values)
            recommended_node = valid_nodes[best_idx]
            
            # Calculate confidence based on Q-value difference
            max_q = max(available_q_values)
            min_q = min(available_q_values)
            if max_q == min_q:
                confidence = 0.7
            else:
                confidence = min(0.99, 0.5 + (max_q - min_q) / 100.0)
        
        # Generate placement strategy
        if cluster_load == 0:  # LOW
            strategy = "Load balancing - distribute workload"
        elif cluster_load == 1:  # MEDIUM
            strategy = "Optimal placement - balance resources"
        else:  # HIGH
            strategy = "Conservative placement - avoid overload"
        
        # Risk assessment
        if confidence > 0.8:
            risk_level = "LOW"
            recommendation = f"OPTIMAL NODE {recommended_node} - High confidence placement"
        elif confidence > 0.6:
            risk_level = "MEDIUM"
            recommendation = f"GOOD NODE {recommended_node} - Balanced placement"
        else:
            risk_level = "HIGH"
            recommendation = f"FALLBACK NODE {recommended_node} - Monitor closely"
        
        return {
            'recommended_node': recommended_node,
            'cluster_load': cluster_load_str,
            'pod_type': pod_type,
            'state': int(state),
            'confidence': float(confidence),
            'recommendation': recommendation,
            'risk_level': risk_level,
            'strategy': strategy,
            'q_value': float(q_values[recommended_node] if recommended_node < len(q_values) else 0),
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'type': self.params.get('algorithm', 'Q-Learning'),
                'states': self.params.get('states', 12),
                'actions': self.params.get('actions', 5),
                'business_score': self.metadata.get('business_score', 0)
            }
        }


# KServe Model Class
if KSERVE_AVAILABLE:
    class QLearningKServeModel(Model):
        """KServe Model wrapper for Q-Learning Optimizer"""
        
        def __init__(self, name: str):
            super().__init__(name)
            self.optimizer = QLearningOptimizerModel()
            self.ready = self.optimizer.is_ready
        
        def predict(self, request: Dict) -> Dict:
            """KServe predict interface"""
            try:
                # Extract inputs from KServe request format
                if 'instances' in request:
                    inputs = request['instances'][0]
                else:
                    inputs = request
                
                # Make prediction
                result = self.optimizer.predict(inputs)
                
                # Return in KServe format
                return {
                    'predictions': [result],
                    'model_name': self.name,
                    'model_version': '1.0.0'
                }
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return {
                    'predictions': [],
                    'error': str(e),
                    'model_name': self.name
                }

# Flask Fallback Service
else:
    app = Flask(__name__)
    optimizer = QLearningOptimizerModel()
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy' if optimizer.is_ready else 'unhealthy',
            'service': 'qlearning-optimizer',
            'model_ready': optimizer.is_ready,
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/v1/models/qlearning-optimizer:predict', methods=['POST'])
    def predict():
        """Main prediction endpoint (KServe compatible)"""
        try:
            data = request.get_json()
            
            # Handle both KServe and direct formats
            if 'instances' in data:
                inputs = data['instances'][0]
            else:
                inputs = data
            
            result = optimizer.predict(inputs)
            
            return jsonify({
                'predictions': [result],
                'model_name': 'qlearning-optimizer',
                'model_version': '1.0.0'
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({
                'error': str(e),
                'model_name': 'qlearning-optimizer'
            }), 400
    
    @app.route('/v1/models/qlearning-optimizer', methods=['GET'])
    def model_info():
        """Model information endpoint"""
        return jsonify({
            'name': 'qlearning-optimizer',
            'versions': ['1.0.0'],
            'ready': optimizer.is_ready,
            'metadata': optimizer.metadata,
            'parameters': optimizer.params,
            'q_table_shape': list(optimizer.q_table.shape) if optimizer.q_table is not None else None
        })

def main():
    """Main entry point"""
    if KSERVE_AVAILABLE:
        # KServe deployment
        logger.info("Starting Q-Learning Optimizer with KServe...")
        model = QLearningKServeModel("qlearning-optimizer")
        ModelServer().start([model])
    else:
        # Flask fallback
        logger.info("Starting Q-Learning Optimizer with Flask fallback...")
        logger.info(f"Models ready: {optimizer.is_ready}")
        logger.info(f"Business score: {optimizer.metadata.get('business_score', 0):.1f}/100")
        app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == "__main__":
    main()