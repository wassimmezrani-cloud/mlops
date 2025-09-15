#!/usr/bin/env python3
"""
ML Predictor Service API
Service production pour ML-Scheduler Premier Expert
Respect .claude_code_rules - Pas d'emojis
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import json
import logging
import os
from datetime import datetime
import traceback

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class MLPredictorService:
    """Service API pour ML Predictor"""
    
    def __init__(self, model_path="./models/ml_predictor"):
        self.model_path = model_path
        self.model = None
        self.features = None
        self.metadata = None
        self.is_ready = False
        self.load_model()
    
    def load_model(self):
        """Charger modele et metadonnees"""
        try:
            # Charger modele
            model_file = f"{self.model_path}/model.pkl"
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
                logger.info("Modele charge avec succes")
            else:
                raise FileNotFoundError(f"Modele non trouve: {model_file}")
            
            # Charger features
            features_file = f"{self.model_path}/features.json"
            if os.path.exists(features_file):
                with open(features_file, 'r') as f:
                    self.features = json.load(f)
                logger.info(f"Features chargees: {len(self.features)} colonnes")
            else:
                raise FileNotFoundError(f"Features non trouvees: {features_file}")
            
            # Charger metadonnees
            metadata_file = f"{self.model_path}/metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Metadonnees chargees: {self.metadata['model_name']}")
            
            self.is_ready = True
            logger.info("Service ML Predictor ready")
            
        except Exception as e:
            logger.error(f"Erreur chargement modele: {e}")
            self.is_ready = False
            raise
    
    def validate_input(self, features_array):
        """Valider format input"""
        if not isinstance(features_array, (list, np.ndarray)):
            raise ValueError("Input doit etre list ou array")
        
        features_array = np.array(features_array)
        
        if len(features_array.shape) == 1:
            features_array = features_array.reshape(1, -1)
        
        if features_array.shape[1] != len(self.features):
            raise ValueError(
                f"Nombre features incorrect: {features_array.shape[1]}, "
                f"attendu: {len(self.features)}"
            )
        
        return features_array
    
    def predict(self, features_array):
        """Effectuer prediction"""
        if not self.is_ready:
            raise RuntimeError("Service non ready")
        
        # Validation input
        validated_features = self.validate_input(features_array)
        
        # Prediction
        predictions = self.model.predict(validated_features)
        
        return predictions.tolist()
    
    def get_model_info(self):
        """Obtenir informations modele"""
        return {
            'model_name': self.metadata.get('model_name', 'unknown') if self.metadata else 'unknown',
            'model_type': self.metadata.get('model_type', 'unknown') if self.metadata else 'unknown',
            'version': self.metadata.get('version', '1.0.0') if self.metadata else '1.0.0',
            'features_count': len(self.features) if self.features else 0,
            'is_ready': self.is_ready,
            'timestamp': datetime.now().isoformat()
        }

# Instance globale service
predictor_service = MLPredictorService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if predictor_service.is_ready else 'unhealthy',
        'service': 'ml-predictor',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/info', methods=['GET'])
def model_info():
    """Informations modele endpoint"""
    try:
        return jsonify(predictor_service.get_model_info())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint principal"""
    try:
        # Parse request JSON
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Format incorrect - attendu: {"features": [...]}'
            }), 400
        
        features = data['features']
        
        # Effectuer prediction
        predictions = predictor_service.predict(features)
        
        # Retourner resultats
        response = {
            'predictions': predictions,
            'model_name': predictor_service.metadata.get('model_name') if predictor_service.metadata else 'ml-predictor',
            'version': predictor_service.metadata.get('version', '1.0.0') if predictor_service.metadata else '1.0.0',
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erreur prediction: {e}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict_load', methods=['POST'])
def predict_load():
    """Endpoint specialise prediction charge nodes"""
    try:
        data = request.get_json()
        
        if not data or 'node_features' not in data:
            return jsonify({
                'error': 'Format incorrect - attendu: {"node_features": {...}}'
            }), 400
        
        node_features = data['node_features']
        
        # Convertir features node selon ordre exact features.json
        if isinstance(node_features, dict):
            feature_values = []
            
            # 1-2: load1, load5
            feature_values.extend([
                node_features.get('load1', 1.0),
                node_features.get('load5', 1.2)
            ])
            
            # 3-8: Features temporelles
            current_hour = datetime.now().hour
            current_dow = datetime.now().weekday()
            
            feature_values.extend([
                current_hour,  # hour
                current_dow,   # day_of_week
                datetime.now().day,  # day_of_month
                current_dow * 24 + current_hour,  # hour_of_week
                1 if (9 <= current_hour <= 17 and current_dow < 5) else 0,  # is_business_hours
                1 if current_dow >= 5 else 0  # is_weekend
            ])
            
            # 9-13: Features lag
            current_cpu = node_features.get('cpu_rate', 0.1)
            current_mem = node_features.get('memory_utilization', 0.5)
            
            feature_values.extend([
                current_cpu, current_cpu, current_cpu,  # cpu_rate_lag_1, 5, 10
                current_mem, current_mem  # memory_util_lag_1, 5
            ])
            
            # 14-18: Features moyennes mobiles
            feature_values.extend([
                current_cpu, current_cpu, current_cpu,  # cpu_ma_5, 15, 30
                current_mem, current_mem  # memory_ma_10, 30
            ])
            
            # 19-20: Features node contextuelles
            memory_capacity = node_features.get('memory_capacity', 16e9)
            feature_values.extend([
                memory_capacity,  # node_memory_capacity
                0.5  # node_capacity_normalized
            ])
            
            # 21-26: One-hot encoding nodes (node_is_0 to node_is_5)
            node_id = hash(node_features.get('node_name', 'default')) % 6
            for i in range(6):
                feature_values.append(1 if i == node_id else 0)
            
            # 27-28: Features interaction
            current_load1 = node_features.get('load1', 1.0)
            feature_values.extend([
                current_cpu * current_mem,  # cpu_memory_interaction
                current_load1 / (current_mem + 0.001)  # load_memory_ratio
            ])
            
        else:
            # Features directes en array
            feature_values = node_features
        
        # Predire avec features construites
        predictions = predictor_service.predict([feature_values])
        
        # Interpreter prediction
        predicted_load = predictions[0] if predictions else 0
        
        # Classification charge
        if predicted_load > 0.8:
            load_level = "HIGH"
            recommendation = "AVOID"
        elif predicted_load > 0.6:
            load_level = "MEDIUM"
            recommendation = "MONITOR"
        else:
            load_level = "LOW"
            recommendation = "OK"
        
        response = {
            'node': node_features.get('node_name', 'unknown'),
            'predicted_load': predicted_load,
            'load_level': load_level,
            'recommendation': recommendation,
            'confidence': 0.85,  # Static pour demo
            'horizon_minutes': 30,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Erreur prediction charge: {e}")
        
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/features', methods=['GET'])
def get_features():
    """Obtenir liste features attendues"""
    if not predictor_service.features:
        return jsonify({'error': 'Features non chargees'}), 500
    
    return jsonify({
        'features': predictor_service.features,
        'features_count': len(predictor_service.features),
        'description': 'Features attendues pour prediction'
    })

if __name__ == '__main__':
    # Verification readiness service
    if not predictor_service.is_ready:
        logger.error("Service non ready - Arret")
        exit(1)
    
    logger.info("Demarrage ML Predictor Service")
    logger.info(f"Modele: {predictor_service.metadata.get('model_name') if predictor_service.metadata else 'unknown'}")
    logger.info(f"Features: {len(predictor_service.features)} colonnes")
    
    # Demarrer serveur Flask
    app.run(host='0.0.0.0', port=5000, debug=False)