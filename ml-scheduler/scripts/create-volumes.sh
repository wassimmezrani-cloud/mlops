#!/bin/bash

# 🚀 ML-Scheduler Volumes Configuration
# Create optimized storage volumes for ML development

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║              📦 ML-SCHEDULER VOLUMES SETUP                  ║
║                                                               ║
║  Creating optimized storage for ML development:              ║
║  ├── 🗄️  Historical Data Storage (50Gi)                     ║
║  ├── 🤖 ML Models Repository (30Gi)                          ║
║  ├── 📊 Experiments & Artifacts (20Gi)                      ║
║  ├── 💾 Workspace & Notebooks (20Gi)                        ║
║  └── ⚡ Cache & Temporary Data (10Gi)                       ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

NAMESPACE="wassimmezrani"

echo -e "${BLUE}[INFO]${NC} Creating ML-Scheduler storage volumes in namespace: ${NAMESPACE}"

# Create PersistentVolumeClaims for ML-Scheduler
kubectl apply -f - << EOF
---
# 1. Historical Data Storage (Prometheus metrics, K8s events)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-historical-data
  namespace: ${NAMESPACE}
  labels:
    app: ml-scheduler
    component: data-storage
    data-type: historical
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: longhorn

---
# 2. ML Models Storage (XGBoost, Q-Learning, Isolation Forest)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-models-storage
  namespace: ${NAMESPACE}
  labels:
    app: ml-scheduler
    component: model-storage
    data-type: models
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 30Gi
  storageClassName: longhorn

---
# 3. Experiments & Artifacts (MLflow, Katib, results)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-experiments
  namespace: ${NAMESPACE}
  labels:
    app: ml-scheduler
    component: experiments
    data-type: artifacts
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: longhorn

---
# 4. Workspace & Notebooks (Jupyter notebooks, code, configs)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-scheduler-workspace
  namespace: ${NAMESPACE}
  labels:
    app: ml-scheduler
    component: workspace
    data-type: development
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: longhorn

---
# 5. Cache & Temporary Data (Feast cache, temp files)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-cache-storage
  namespace: ${NAMESPACE}
  labels:
    app: ml-scheduler
    component: cache
    data-type: temporary
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: longhorn

---
# 6. Feature Store Data (Feast features, processed data)
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ml-feature-store
  namespace: ${NAMESPACE}
  labels:
    app: ml-scheduler
    component: feature-store
    data-type: features
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 25Gi
  storageClassName: longhorn
EOF

echo -e "${YELLOW}[INFO]${NC} Waiting for volumes to be bound..."
sleep 5

# Check volume status
kubectl get pvc -n ${NAMESPACE}

echo -e "${GREEN}[SUCCESS]${NC} All storage volumes created successfully!"

echo -e "\n${BLUE}📊 Storage Summary:${NC}"
echo -e "├── 🗄️  Historical Data: 50Gi (Prometheus metrics, K8s events)"
echo -e "├── 🤖 ML Models: 30Gi (XGBoost, Q-Learning, Isolation Forest)"
echo -e "├── 📊 Experiments: 20Gi (MLflow artifacts, Katib results)"
echo -e "├── 💾 Workspace: 20Gi (Jupyter notebooks, development)"
echo -e "├── ⚡ Cache: 10Gi (Temporary data, Redis cache)"
echo -e "└── 🍽️  Feature Store: 25Gi (Feast features, processed data)"
echo -e "\n${YELLOW}Total Storage: 155Gi${NC}"

echo -e "\n${BLUE}🚀 Next: Update Jupyter notebook to use these volumes${NC}"
EOF
