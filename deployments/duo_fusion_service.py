#!/usr/bin/env python3
"""
Duo Fusion Service - XGBoost + Q-Learning Integration
Fusion synergique des deux experts IA ML-Scheduler
Respect .claude_code_rules - No emojis
"""

import os
import json
import logging
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from flask import Flask, request, jsonify

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DuoFusionService:
    """Service de fusion intelligent XGBoost + Q-Learning"""
    
    def __init__(self, 
                 xgboost_url: str = "http://localhost:8080",
                 qlearning_url: str = "http://localhost:8081"):
        """
        Initialize Duo Fusion Service
        
        Args:
            xgboost_url: XGBoost service endpoint
            qlearning_url: Q-Learning service endpoint
        """
        self.xgboost_url = xgboost_url
        self.qlearning_url = qlearning_url
        self.fusion_weights = {
            'xgboost': 0.6,  # XGBoost weight for filtering
            'qlearning': 0.4  # Q-Learning weight for optimization
        }
        
        logger.info("Duo Fusion Service initialized")
        logger.info(f"XGBoost endpoint: {xgboost_url}")
        logger.info(f"Q-Learning endpoint: {qlearning_url}")
    
    def check_services_health(self) -> Dict[str, bool]:
        """Check health of both expert services"""
        health_status = {}
        
        # Check XGBoost health
        try:
            response = requests.get(f"{self.xgboost_url}/health", timeout=5)
            health_status['xgboost'] = response.status_code == 200
        except Exception as e:
            logger.warning(f"XGBoost health check failed: {e}")
            health_status['xgboost'] = False
        
        # Check Q-Learning health
        try:
            response = requests.get(f"{self.qlearning_url}/health", timeout=5)
            health_status['qlearning'] = response.status_code == 200
        except Exception as e:
            logger.warning(f"Q-Learning health check failed: {e}")
            health_status['qlearning'] = False
        
        return health_status
    
    def call_xgboost_prediction(self, features: List[float], 
                               horizon: str = "30min", 
                               metric: str = "memory") -> Optional[Dict]:
        """
        Call XGBoost "Le Prophète" for future load prediction
        
        Args:
            features: Feature vector for prediction
            horizon: Prediction horizon (30min, 1h, 2h)
            metric: Metric type (cpu, memory)
            
        Returns:
            XGBoost prediction result or None if failed
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
                f"{self.xgboost_url}/v1/models/xgboost-predictor:predict",
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
    
    def call_qlearning_recommendation(self, cluster_load: str, 
                                    pod_type: str,
                                    available_nodes: List[int]) -> Optional[Dict]:
        """
        Call Q-Learning "L'Optimiseur" for optimal placement
        
        Args:
            cluster_load: Cluster load state (low, medium, high)
            pod_type: Pod type (web, db, worker, ml)
            available_nodes: List of available node indices
            
        Returns:
            Q-Learning recommendation result or None if failed
        """
        try:
            payload = {
                "instances": [{
                    "cluster_load": cluster_load,
                    "pod_type": pod_type,
                    "available_nodes": available_nodes
                }]
            }
            
            response = requests.post(
                f"{self.qlearning_url}/v1/models/qlearning-optimizer:predict",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'predictions' in result and len(result['predictions']) > 0:
                    return result['predictions'][0]
                    
        except Exception as e:
            logger.error(f"Q-Learning recommendation failed: {e}")
            
        return None
    
    def filter_nodes_with_xgboost(self, nodes: List[int], 
                                 features_base: List[float],
                                 horizon: str = "1h") -> List[int]:
        """
        Filter out nodes that XGBoost predicts will be saturated
        
        Args:
            nodes: List of candidate nodes
            features_base: Base feature vector
            horizon: Prediction horizon
            
        Returns:
            Filtered list of safe nodes
        """
        safe_nodes = []
        saturation_threshold = 0.8  # 80% saturation threshold
        
        for node_idx in nodes:
            # Modify features for this specific node (simplified)
            node_features = features_base.copy()
            if len(node_features) > 10:
                node_features[10] = float(node_idx)  # Node identifier feature
            
            # Check CPU saturation prediction
            cpu_pred = self.call_xgboost_prediction(node_features, horizon, "cpu")
            memory_pred = self.call_xgboost_prediction(node_features, horizon, "memory")
            
            node_safe = True
            
            # Check CPU prediction
            if cpu_pred and cpu_pred.get('prediction', 0) > saturation_threshold:
                logger.info(f"Node {node_idx} filtered out - CPU saturation predicted: {cpu_pred['prediction']:.2f}")
                node_safe = False
            
            # Check Memory prediction  
            if memory_pred and memory_pred.get('prediction', 0) > saturation_threshold:
                logger.info(f"Node {node_idx} filtered out - Memory saturation predicted: {memory_pred['prediction']:.2f}")
                node_safe = False
            
            if node_safe:
                safe_nodes.append(node_idx)
        
        # Fallback: if all nodes filtered, return original list
        if not safe_nodes:
            logger.warning("All nodes filtered by XGBoost - returning original list")
            return nodes
            
        return safe_nodes
    
    def fuse_duo_decisions(self, pod_spec: Dict) -> Dict:
        """
        FUSION SYNERGIQUE - Combine XGBoost filtering + Q-Learning optimization
        
        Args:
            pod_spec: Pod specification with requirements
            
        Returns:
            Fused placement decision
        """
        start_time = datetime.now()
        
        # Extract pod information
        pod_type = pod_spec.get('pod_type', 'web')
        cpu_request = pod_spec.get('cpu_request', 0.1)
        memory_request = pod_spec.get('memory_request', 0.2)
        available_nodes = pod_spec.get('available_nodes', [0, 1, 2, 3, 4])
        cluster_load = pod_spec.get('cluster_load', 'medium')
        
        # Generate features for XGBoost (simplified)
        features_base = [
            cpu_request, memory_request,
            len(available_nodes), 
            {'low': 0, 'medium': 1, 'high': 2}.get(cluster_load, 1),
            {'web': 0, 'db': 1, 'worker': 2, 'ml': 3}.get(pod_type, 0),
            datetime.now().hour, datetime.now().weekday(),
            np.random.random(),  # Simulated load features
            np.random.random(), np.random.random(),
            0,  # Node identifier (will be modified per node)
            # Additional features to reach expected count
        ] + [0.0] * 17  # Pad to 28 features total
        
        # PHASE 1: XGBoost Filtering - "Le Prophète" élimine nodes dangereux
        logger.info(f"PHASE 1: XGBoost filtering {len(available_nodes)} candidate nodes")
        safe_nodes = self.filter_nodes_with_xgboost(
            available_nodes, features_base, horizon="1h"
        )
        
        xgboost_filtered = len(available_nodes) - len(safe_nodes)
        logger.info(f"XGBoost filtered out {xgboost_filtered} nodes, {len(safe_nodes)} remain safe")
        
        # PHASE 2: Q-Learning Optimization - "L'Optimiseur" choisit optimal
        logger.info(f"PHASE 2: Q-Learning optimization among {len(safe_nodes)} safe nodes")
        qlearning_recommendation = self.call_qlearning_recommendation(
            cluster_load, pod_type, safe_nodes
        )
        
        # PHASE 3: Fusion Decision
        if qlearning_recommendation and qlearning_recommendation.get('recommended_node') in safe_nodes:
            # Perfect fusion - Q-Learning recommendation is among XGBoost safe nodes
            recommended_node = qlearning_recommendation['recommended_node']
            confidence = min(0.95, qlearning_recommendation.get('confidence', 0.7) * 1.1)
            strategy = "FUSION OPTIMALE - XGBoost safe + Q-Learning optimal"
            risk_level = "LOW"
            
        elif safe_nodes:
            # XGBoost filtering worked, choose first safe node
            recommended_node = safe_nodes[0]
            confidence = 0.75
            strategy = "XGBOOST SAFE - Q-Learning indisponible"
            risk_level = "MEDIUM"
            
        elif qlearning_recommendation:
            # Q-Learning only (XGBoost failed/filtered all)
            recommended_node = qlearning_recommendation['recommended_node']
            confidence = max(0.5, qlearning_recommendation.get('confidence', 0.7) * 0.8)
            strategy = "Q-LEARNING ONLY - XGBoost indisponible"
            risk_level = "MEDIUM"
            
        else:
            # Complete fallback
            recommended_node = available_nodes[0] if available_nodes else 0
            confidence = 0.4
            strategy = "FALLBACK - Experts indisponibles"
            risk_level = "HIGH"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Construct fusion result
        fusion_result = {
            'recommended_node': recommended_node,
            'confidence': confidence,
            'strategy': strategy,
            'risk_level': risk_level,
            'processing_time_ms': processing_time,
            'fusion_details': {
                'xgboost_filtered_nodes': xgboost_filtered,
                'safe_nodes_count': len(safe_nodes),
                'qlearning_available': qlearning_recommendation is not None,
                'fusion_weights': self.fusion_weights
            },
            'expert_inputs': {
                'pod_type': pod_type,
                'cluster_load': cluster_load,
                'available_nodes_original': available_nodes,
                'safe_nodes_filtered': safe_nodes
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"FUSION COMPLETE: Node {recommended_node} selected with {confidence:.2f} confidence ({processing_time:.1f}ms)")
        return fusion_result

# Flask Application
app = Flask(__name__)
fusion_service = DuoFusionService()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    services_health = fusion_service.check_services_health()
    overall_health = any(services_health.values())
    
    return jsonify({
        'status': 'healthy' if overall_health else 'degraded',
        'service': 'duo-fusion',
        'experts_health': services_health,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/v1/models/duo-fusion:predict', methods=['POST'])
def predict():
    """Main fusion prediction endpoint"""
    try:
        data = request.get_json()
        
        # Handle both KServe and direct formats
        if 'instances' in data:
            pod_spec = data['instances'][0]
        else:
            pod_spec = data
        
        # Perform duo fusion
        result = fusion_service.fuse_duo_decisions(pod_spec)
        
        return jsonify({
            'predictions': [result],
            'model_name': 'duo-fusion',
            'model_version': '1.0.0'
        })
        
    except Exception as e:
        logger.error(f"Fusion prediction error: {e}")
        return jsonify({
            'error': str(e),
            'model_name': 'duo-fusion'
        }), 400

@app.route('/v1/models/duo-fusion', methods=['GET'])
def model_info():
    """Model information endpoint"""
    services_health = fusion_service.check_services_health()
    
    return jsonify({
        'name': 'duo-fusion',
        'versions': ['1.0.0'],
        'description': 'XGBoost + Q-Learning Fusion Service',
        'experts': {
            'xgboost': {
                'name': 'Le Prophète',
                'role': 'Future load prediction and node filtering',
                'endpoint': fusion_service.xgboost_url,
                'healthy': services_health.get('xgboost', False)
            },
            'qlearning': {
                'name': 'L\'Optimiseur', 
                'role': 'Optimal placement recommendation',
                'endpoint': fusion_service.qlearning_url,
                'healthy': services_health.get('qlearning', False)
            }
        },
        'fusion_weights': fusion_service.fusion_weights,
        'ready': any(services_health.values())
    })

def main():
    """Main entry point"""
    logger.info("Starting Duo Fusion Service - XGBoost + Q-Learning")
    logger.info("Fusion synergique des experts ML-Scheduler")
    app.run(host='0.0.0.0', port=8082, debug=False)

if __name__ == "__main__":
    main()