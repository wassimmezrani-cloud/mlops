# ML-Scheduler HYDATIS

Système ML pour optimisation intelligente du scheduling Kubernetes sur cluster HYDATIS.

## Architecture

### Algorithmes ML
- **XGBoost Predictor**: Prédiction charge temporelle des nodes
- **Q-Learning Optimizer**: Optimisation décisions de scheduling
- **Isolation Forest**: Détection anomalies comportement cluster

### Infrastructure
- **Cluster**: 6 nodes HYDATIS (3 masters + 3 workers)
- **Monitoring**: Prometheus + Grafana
- **Development**: Jupyter Notebook (wassimmezrani namespace)
- **Storage**: 104Gi volumes persistants

## Structure Projet

```
mlops/
├── src/                               # Code source
│   ├── ml_scheduler/                  # Package principal algorithmes ML
│   └── utils/                         # Utilitaires et clients
├── tests/                             # Tests unitaires et intégration
├── configs/                           # Configurations cluster et ML
├── docs/                              # Documentation technique
├── notebooks/                         # Jupyter notebooks développement
├── scripts/                           # Scripts déploiement et maintenance
├── data/                              # Données collectées (50Gi)
├── models/                            # Modèles entraînés (20Gi)
├── experiments/                       # Expérimentations ML (15Gi)
├── logs/                              # Logs système et ML
└── deployments/                       # Manifests Kubernetes
```

## Quick Start

### 1. Collecte Données
```bash
jupyter notebook notebooks/data_collection_prometheus.ipynb
```

### 2. Entraînement Modèles
```bash
python src/ml_scheduler/xgboost_predictor.py --train
python src/ml_scheduler/qlearning_optimizer.py --train
python src/ml_scheduler/isolation_detector.py --train
```

### 3. Tests
```bash
pytest tests/ -v
```

### 4. Déploiement
```bash
kubectl apply -f deployments/
```

## Configuration

### Cluster HYDATIS
- **Prometheus**: http://10.110.190.83:9090
- **Grafana**: http://10.110.190.84:3000
- **Kubeflow**: http://10.110.190.82

### Mapping Nodes
```
10.110.190.32 → master1
10.110.190.33 → master2  
10.110.190.34 → master3
10.110.190.35 → worker1
10.110.190.36 → worker2
10.110.190.37 → worker3
```

## Développement

### Règles Code
- **INTERDIT**: Emojis dans code/commentaires
- **FORMAT**: SUCCESS/ERROR/WARNING/INFO uniquement
- **STYLE**: PEP8, Black formatting
- **TESTS**: Couverture minimum 80%

### Environnement
```bash
pip install -r requirements.txt
pre-commit install
```

## Monitoring

### Métriques Collectées
- CPU/Memory nodes (8 métriques)
- Pods scheduling (6 métriques)  
- Network/Storage (4 métriques)

### Qualité Données
- **Période**: 15 jours historique
- **Résolution**: 1-2h
- **Format**: CSV + Parquet

## Status

- **Infrastructure**: READY
- **Data Collection**: OPERATIONAL
- **ML Development**: IN PROGRESS
- **Production**: PLANNED

## Contact

**Team**: HYDATIS MLOps
**Environment**: Kubernetes 1.32.7
**Last Update**: 2025-09-04