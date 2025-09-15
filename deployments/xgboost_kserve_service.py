#!/usr/bin/env python3
"""
XGBoost KServe Service
Déploiement production-ready avec KServe
API <50ms latency pour ML-Scheduler
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

class XGBoostPredictorModel:
    """XGBoost Predictor Model for KServe/Flask"""
    
    def __init__(self, model_path: str = "./models/xgboost_predictor"):
        self.model_path = model_path
        self.models = {}
        self.features = []
        self.metadata = {}
        self.is_ready = False
        
        self.load_models()
    
    def load_models(self):
        """Load all XGBoost models and metadata"""
        try:
            # Load metadata
            metadata_file = f"{self.model_path}/metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadata loaded: {self.metadata['model_name']}")
            
            # Load features
            features_file = f"{self.model_path}/features.json"
            if os.path.exists(features_file):
                with open(features_file, 'r') as f:
                    self.features = json.load(f)
                logger.info(f"Features loaded: {len(self.features)} columns")
            
            # Load all trained models
            model_files = [
                'cpu_30min_model.pkl', 'memory_30min_model.pkl',
                'cpu_1h_model.pkl', 'memory_1h_model.pkl',
                'cpu_2h_model.pkl', 'memory_2h_model.pkl'
            ]
            
            for model_file in model_files:
                model_path = f"{self.model_path}/{model_file}"
                if os.path.exists(model_path):
                    model_name = model_file.replace('_model.pkl', '')
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Model loaded: {model_name}")
            
            self.is_ready = len(self.models) > 0
            logger.info(f"XGBoost Predictor loaded: {len(self.models)} models ready")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_ready = False
            raise
    
    def predict(self, inputs: Dict) -> Dict:
        """
        Make predictions using XGBoost models
        Input format: {"horizon": "30min", "metric": "memory", "features": [...]}
        """
        if not self.is_ready:
            raise RuntimeError("Models not ready")
        
        # Validate input format
        if not isinstance(inputs, dict):
            raise ValueError("Input must be a dictionary")
        
        horizon = inputs.get('horizon', '30min')
        metric = inputs.get('metric', 'memory')
        features = inputs.get('features', [])
        
        if not features:
            raise ValueError("Features array required")
        
        # Construct model name
        model_name = f"{metric}_{horizon}"
        
        if model_name not in self.models:
            available_models = list(self.models.keys())
            raise ValueError(f"Model {model_name} not found. Available: {available_models}")
        
        # Get model data
        model_data = self.models[model_name]
        model = model_data['model']
        scaler = model_data.get('scaler')
        
        # Validate features count
        features_array = np.array(features).reshape(1, -1)
        expected_features = len(self.features)
        
        if features_array.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {features_array.shape[1]}")
        
        # Scale features if scaler available
        if scaler:
            features_scaled = scaler.transform(features_array)
        else:
            features_scaled = features_array
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Calculate confidence (simplified)
        if hasattr(model, 'predict_proba'):
            try:
                confidence = np.max(model.predict_proba(features_scaled))
            except:
                confidence = 0.85  # Default confidence
        else:
            confidence = 0.85
        
        # Generate recommendation
        if metric == 'cpu':
            if prediction > 0.8:
                recommendation = "HIGH CPU - Avoid placement"
                risk_level = "HIGH"
            elif prediction > 0.6:
                recommendation = "MEDIUM CPU - Monitor closely"
                risk_level = "MEDIUM"
            else:
                recommendation = "LOW CPU - Safe for placement"
                risk_level = "LOW"
        else:  # memory
            if prediction > 0.85:
                recommendation = "HIGH MEMORY - Avoid placement"
                risk_level = "HIGH"
            elif prediction > 0.70:
                recommendation = "MEDIUM MEMORY - Monitor closely"
                risk_level = "MEDIUM"
            else:
                recommendation = "LOW MEMORY - Safe for placement"
                risk_level = "LOW"
        
        return {
            'prediction': float(prediction),
            'horizon': horizon,
            'metric': metric,
            'model_used': model_name,
            'confidence': confidence,
            'recommendation': recommendation,
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'type': model_data.get('model_type', 'XGBoost'),
                'features_count': len(self.features)
            }
        }

# KServe Model Class
if KSERVE_AVAILABLE:
    class XGBoostKServeModel(Model):
        """KServe Model wrapper for XGBoost Predictor"""
        
        def __init__(self, name: str):
            super().__init__(name)
            self.predictor = XGBoostPredictorModel()
            self.ready = self.predictor.is_ready
        
        def predict(self, request: Dict) -> Dict:
            """KServe predict interface"""
            try:
                # Extract inputs from KServe request format
                if 'instances' in request:
                    inputs = request['instances'][0]
                else:
                    inputs = request
                
                # Make prediction
                result = self.predictor.predict(inputs)
                
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
    predictor = XGBoostPredictorModel()
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy' if predictor.is_ready else 'unhealthy',
            'service': 'xgboost-predictor',
            'models_loaded': len(predictor.models),
            'timestamp': datetime.now().isoformat()
        })
    
    @app.route('/v1/models/xgboost-predictor:predict', methods=['POST'])
    def predict():
        """Main prediction endpoint (KServe compatible)"""
        try:
            data = request.get_json()
            
            # Handle both KServe and direct formats
            if 'instances' in data:
                inputs = data['instances'][0]
            else:
                inputs = data
            
            result = predictor.predict(inputs)
            
            return jsonify({
                'predictions': [result],
                'model_name': 'xgboost-predictor',
                'model_version': '1.0.0'
            })
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({
                'error': str(e),
                'model_name': 'xgboost-predictor'
            }), 400
    
    @app.route('/v1/models/xgboost-predictor', methods=['GET'])
    def model_info():
        """Model information endpoint"""
        return jsonify({
            'name': 'xgboost-predictor',
            'versions': ['1.0.0'],
            'ready': predictor.is_ready,
            'metadata': predictor.metadata,
            'models_available': list(predictor.models.keys()),
            'features_count': len(predictor.features)
        })

def main():
    """Main entry point"""
    if KSERVE_AVAILABLE:
        # KServe deployment
        logger.info("Starting XGBoost Predictor with KServe...")
        model = XGBoostKServeModel("xgboost-predictor")
        ModelServer().start([model])
    else:
        # Flask fallback
        logger.info("Starting XGBoost Predictor with Flask fallback...")
        logger.info(f"Models ready: {predictor.is_ready}")
        logger.info(f"Available models: {list(predictor.models.keys())}")
        app.run(host='0.0.0.0', port=8080, debug=False)

if __name__ == "__main__":
    main()