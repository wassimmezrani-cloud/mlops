"""
ML-Scheduler HYDATIS - Package principal
Algorithmes ML pour optimisation scheduling Kubernetes
"""
__version__ = "1.0.0"
__author__ = "HYDATIS Team"

from .xgboost_predictor import XGBoostPredictor
from .qlearning_optimizer import QLearningOptimizer
from .isolation_detector import IsolationDetector

__all__ = [
    'XGBoostPredictor',
    'QLearningOptimizer', 
    'IsolationDetector'
]