# XGBoost Predictor Service Dockerfile
# ML-Scheduler HYDATIS - Service de prédiction de charge

FROM python:3.9-slim

# Métadonnées
LABEL maintainer="HYDATIS DevOps Team <devops@hydatis.tn>"
LABEL description="XGBoost Predictor Service for ML-Scheduler"
LABEL version="1.0.0"
LABEL component="ml-services"
LABEL algorithm="xgboost"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Répertoire de travail
WORKDIR /app

# Installation dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie des requirements et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY app.py .
COPY model.py .
COPY trainer.py .
COPY config.yaml .

# Création des répertoires pour données
RUN mkdir -p /data/models /data/historical /data/cache /app/logs

# Utilisateur non-root pour sécurité
RUN groupadd -r xgboost && useradd -r -g xgboost xgboost
RUN chown -R xgboost:xgboost /app /data
USER xgboost

# Port d'exposition
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Commande de démarrage
CMD ["python", "app.py"]
