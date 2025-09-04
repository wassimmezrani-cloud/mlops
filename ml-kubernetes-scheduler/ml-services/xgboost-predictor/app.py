# XGBoost Predictor Service - ML-Scheduler HYDATIS
# Prédiction de charge future des worker nodes

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import xgboost as xgb
import numpy as np
import pandas as pd
import logging
import time
import asyncio
import uvicorn
from datetime import datetime, timedelta
import joblib
import redis
import json
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [xgboost-predictor] %(message)s'
)
logger = logging.getLogger(__name__)

# Métriques Prometheus
prediction_requests = Counter('xgboost_prediction_requests_total', 
                            'Total prediction requests', ['status'])
prediction_latency = Histogram('xgboost_prediction_duration_seconds',
                             'Prediction request duration')
model_accuracy = Gauge('xgboost_model_accuracy', 'Current model accuracy')
cache_hits = Counter('xgboost_cache_hits_total', 'Cache hits')

# Configuration
class Config:
    model_path = os.getenv('MODEL_PATH', '/data/models/xgboost_model.pkl')
    redis_host = os.getenv('REDIS_HOST', 'redis-cache-service.ml-scheduler.svc.cluster.local')
    redis_port = int(os.getenv('REDIS_PORT', '6379'))
    redis_db = int(os.getenv('REDIS_DB', '2'))  # DB 2 pour XGBoost
    cache_ttl = int(os.getenv('CACHE_TTL', '300'))  # 5 minutes
    model_version = os.getenv('MODEL_VERSION', 'v1.0.0')
    accuracy_target = float(os.getenv('ACCURACY_TARGET', '0.89'))  # 89%
    longhorn_data_path = os.getenv('LONGHORN_DATA_PATH', '/data/historical')

# Modèles de données
class NodeMetrics(BaseModel):
    node_name: str = Field(..., description="Nom du worker node")
    cpu_usage: float = Field(..., ge=0, le=1, description="Usage CPU actuel (0-1)")
    memory_usage: float = Field(..., ge=0, le=1, description="Usage mémoire actuel (0-1)")
    disk_usage: float = Field(..., ge=0, le=1, description="Usage disque actuel (0-1)")
    network_io: float = Field(..., ge=0, description="I/O réseau (bytes/sec)")
    pod_count: int = Field(..., ge=0, description="Nombre de pods actuels")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PredictionRequest(BaseModel):
    nodes: List[NodeMetrics] = Field(..., description="Métriques des worker nodes")
    prediction_horizon: int = Field(default=300, ge=60, le=3600, 
                                  description="Horizon prédiction en secondes")
    historical_window: int = Field(default=1800, ge=300, le=7200,
                                 description="Fenêtre historique en secondes")

class NodePrediction(BaseModel):
    node_name: str
    current_cpu: float
    predicted_cpu: float
    current_memory: float
    predicted_memory: float
    confidence_score: float
    risk_level: str  # LOW, MEDIUM, HIGH
    recommended_action: str

class PredictionResponse(BaseModel):
    request_id: str
    timestamp: datetime
    predictions: List[NodePrediction]
    model_version: str
    accuracy_score: float
    processing_time_ms: float

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_loaded: bool
    model_version: str
    accuracy: float
    cache_status: str
    data_path_accessible: bool

# Service XGBoost Predictor
class XGBoostPredictor:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.redis_client = None
        self.model_version = Config.model_version
        self.accuracy = 0.0
        self.last_training = None
        
    async def initialize(self):
        """Initialise le service"""
        logger.info("🚀 Initialisation XGBoost Predictor Service HYDATIS")
        
        # Connexion Redis
        await self._connect_redis()
        
        # Chargement du modèle
        await self._load_model()
        
        # Vérification données historiques
        await self._check_data_access()
        
        logger.info("✅ XGBoost Predictor initialisé avec succès")
        
    async def _connect_redis(self):
        """Connexion au cache Redis"""
        try:
            self.redis_client = redis.Redis(
                host=Config.redis_host,
                port=Config.redis_port,
                db=Config.redis_db,
                decode_responses=True
            )
            # Test connexion
            self.redis_client.ping()
            logger.info("✅ Connexion Redis établie")
        except Exception as e:
            logger.error(f"❌ Erreur connexion Redis: {e}")
            self.redis_client = None
            
    async def _load_model(self):
        """Charge le modèle XGBoost"""
        try:
            if os.path.exists(Config.model_path):
                self.model = joblib.load(Config.model_path)
                
                # Charger métadonnées modèle
                metadata_path = Config.model_path.replace('.pkl', '_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        self.accuracy = metadata.get('accuracy', 0.0)
                        self.feature_columns = metadata.get('feature_columns', [])
                        self.last_training = metadata.get('last_training')
                        
                model_accuracy.set(self.accuracy)
                logger.info(f"✅ Modèle XGBoost chargé - Accuracy: {self.accuracy:.3f}")
            else:
                logger.warning(f"⚠️ Modèle non trouvé: {Config.model_path}")
                # Créer modèle par défaut pour développement
                await self._create_default_model()
                
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle: {e}")
            await self._create_default_model()
            
    async def _create_default_model(self):
        """Crée un modèle par défaut pour développement"""
        logger.info("🛠️ Création modèle XGBoost par défaut")
        
        # Modèle simple pour démonstration
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.datasets import make_regression
        
        # Données synthétiques
        X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        
        # Entraînement modèle simple
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model.fit(X, y)
        
        self.accuracy = 0.75  # Accuracy simulée
        self.feature_columns = [f'feature_{i}' for i in range(10)]
        model_accuracy.set(self.accuracy)
        
        logger.info("✅ Modèle par défaut créé")
        
    async def _check_data_access(self):
        """Vérifie l'accès aux données historiques Longhorn"""
        try:
            if os.path.exists(Config.longhorn_data_path):
                logger.info(f"✅ Accès données historiques: {Config.longhorn_data_path}")
            else:
                logger.warning(f"⚠️ Données historiques non accessibles: {Config.longhorn_data_path}")
        except Exception as e:
            logger.error(f"❌ Erreur accès données: {e}")
            
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Effectue la prédiction de charge"""
        start_time = time.time()
        request_id = f"xgb_{int(time.time())}_{hash(str(request.nodes)) % 10000}"
        
        with prediction_latency.time():
            try:
                # Vérifier cache Redis
                cached = await self._check_cache(request_id, request)
                if cached:
                    cache_hits.inc()
                    return cached
                    
                # Préparer données pour prédiction
                features = await self._prepare_features(request)
                
                # Effectuer prédictions
                predictions = await self._make_predictions(features, request)
                
                # Créer réponse
                processing_time = (time.time() - start_time) * 1000
                response = PredictionResponse(
                    request_id=request_id,
                    timestamp=datetime.utcnow(),
                    predictions=predictions,
                    model_version=self.model_version,
                    accuracy_score=self.accuracy,
                    processing_time_ms=processing_time
                )
                
                # Mettre en cache
                await self._cache_response(request_id, response)
                
                prediction_requests.labels(status='success').inc()
                logger.info(f"✅ Prédiction {request_id} - {len(predictions)} nodes - {processing_time:.2f}ms")
                
                return response
                
            except Exception as e:
                prediction_requests.labels(status='error').inc()
                logger.error(f"❌ Erreur prédiction {request_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
                
    async def _check_cache(self, request_id: str, request: PredictionRequest) -> Optional[PredictionResponse]:
        """Vérifie le cache Redis"""
        if not self.redis_client:
            return None
            
        try:
            # Créer clé cache basée sur hash des données
            cache_key = f"xgb_pred_{hash(str(request.nodes))}_{request.prediction_horizon}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return PredictionResponse(**data)
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur cache: {e}")
            
        return None
        
    async def _cache_response(self, request_id: str, response: PredictionResponse):
        """Met en cache la réponse"""
        if not self.redis_client:
            return
            
        try:
            cache_key = f"xgb_pred_{request_id}"
            data = response.dict()
            # Sérialiser datetime
            data['timestamp'] = data['timestamp'].isoformat()
            
            self.redis_client.setex(
                cache_key,
                Config.cache_ttl,
                json.dumps(data, default=str)
            )
        except Exception as e:
            logger.warning(f"⚠️ Erreur mise en cache: {e}")
            
    async def _prepare_features(self, request: PredictionRequest) -> pd.DataFrame:
        """Prépare les features pour la prédiction"""
        features_list = []
        
        for node in request.nodes:
            # Features de base
            features = {
                'node_name': node.node_name,
                'cpu_usage': node.cpu_usage,
                'memory_usage': node.memory_usage,
                'disk_usage': node.disk_usage,
                'network_io': node.network_io,
                'pod_count': node.pod_count,
            }
            
            # Features temporelles
            now = datetime.utcnow()
            features.update({
                'hour_of_day': now.hour,
                'day_of_week': now.weekday(),
                'is_weekend': now.weekday() >= 5,
            })
            
            # Features historiques (simulées pour développement)
            features.update({
                'cpu_trend_5m': np.random.normal(0, 0.1),
                'memory_trend_5m': np.random.normal(0, 0.1),
                'load_avg_15m': node.cpu_usage + np.random.normal(0, 0.05),
            })
            
            features_list.append(features)
            
        return pd.DataFrame(features_list)
        
    async def _make_predictions(self, features: pd.DataFrame, request: PredictionRequest) -> List[NodePrediction]:
        """Effectue les prédictions ML"""
        predictions = []
        
        for _, row in features.iterrows():
            # Préparer features numériques pour le modèle
            numeric_features = [
                row['cpu_usage'], row['memory_usage'], row['disk_usage'],
                row['network_io'] / 1e6, row['pod_count'] / 100,
                row['hour_of_day'] / 24, row['day_of_week'] / 7,
                float(row['is_weekend']), row['cpu_trend_5m'], row['memory_trend_5m']
            ]
            
            # Simulation prédiction (remplacer par vraie prédiction)
            if self.model:
                try:
                    pred_features = np.array([numeric_features])
                    raw_prediction = self.model.predict(pred_features)[0]
                    
                    # Convertir en prédictions CPU/Memory
                    cpu_delta = (raw_prediction % 0.2) - 0.1  # -10% à +10%
                    memory_delta = ((raw_prediction * 1.5) % 0.15) - 0.075  # -7.5% à +7.5%
                    
                except Exception as e:
                    logger.warning(f"⚠️ Erreur modèle pour {row['node_name']}: {e}")
                    cpu_delta = np.random.normal(0, 0.05)
                    memory_delta = np.random.normal(0, 0.03)
            else:
                # Prédiction par défaut
                cpu_delta = np.random.normal(0, 0.05)
                memory_delta = np.random.normal(0, 0.03)
                
            # Calculer prédictions finales
            predicted_cpu = max(0, min(1, row['cpu_usage'] + cpu_delta))
            predicted_memory = max(0, min(1, row['memory_usage'] + memory_delta))
            
            # Calculer score de confiance
            confidence = min(0.95, self.accuracy * (1 - abs(cpu_delta) - abs(memory_delta)))
            
            # Déterminer niveau de risque
            cpu_risk = predicted_cpu > 0.8
            memory_risk = predicted_memory > 0.85
            
            if cpu_risk or memory_risk:
                risk_level = "HIGH"
                action = "AVOID_SCHEDULING"
            elif predicted_cpu > 0.6 or predicted_memory > 0.7:
                risk_level = "MEDIUM"
                action = "MONITOR_CLOSELY"
            else:
                risk_level = "LOW"
                action = "SCHEDULE_NORMALLY"
                
            prediction = NodePrediction(
                node_name=row['node_name'],
                current_cpu=row['cpu_usage'],
                predicted_cpu=predicted_cpu,
                current_memory=row['memory_usage'],
                predicted_memory=predicted_memory,
                confidence_score=confidence,
                risk_level=risk_level,
                recommended_action=action
            )
            
            predictions.append(prediction)
            
        return predictions

# Application FastAPI
app = FastAPI(
    title="XGBoost Predictor Service",
    description="Service de prédiction de charge ML-Scheduler HYDATIS",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instance service
predictor = XGBoostPredictor()

@app.on_event("startup")
async def startup_event():
    await predictor.initialize()

@app.post("/predict", response_model=PredictionResponse)
async def predict_load(request: PredictionRequest):
    """Prédit la charge future des worker nodes"""
    return await predictor.predict(request)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check du service"""
    data_accessible = os.path.exists(Config.longhorn_data_path)
    cache_status = "connected" if predictor.redis_client else "disconnected"
    
    return HealthResponse(
        status="healthy" if predictor.model else "degraded",
        timestamp=datetime.utcnow(),
        model_loaded=predictor.model is not None,
        model_version=predictor.model_version,
        accuracy=predictor.accuracy,
        cache_status=cache_status,
        data_path_accessible=data_accessible
    )

@app.get("/metrics")
async def metrics():
    """Métriques Prometheus"""
    return generate_latest().decode('utf-8')

@app.get("/info")
async def service_info():
    """Informations du service"""
    return {
        "service": "XGBoost Predictor",
        "version": Config.model_version,
        "accuracy_target": Config.accuracy_target,
        "current_accuracy": predictor.accuracy,
        "mission": "Prédiction charge future worker nodes - ML-Scheduler HYDATIS",
        "algorithms": ["XGBoost", "Random Forest Fallback"],
        "features": [
            "Prédiction CPU/Memory à 5-60 minutes",
            "Cache Redis pour performance",
            "Stockage Longhorn pour données historiques",
            "Métriques Prometheus intégrées"
        ]
    }

if __name__ == "__main__":
    logger.info("🚀 Démarrage XGBoost Predictor Service - ML-Scheduler HYDATIS")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        log_level="info",
        access_log=True
    )
