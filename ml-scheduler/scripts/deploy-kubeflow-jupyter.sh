#!/bin/bash

# 🧠 Deploy ML-Scheduler Jupyter Notebook in Kubeflow (wassimmezrani namespace)
# Optimized for ML development with all necessary libraries

set -euo pipefail

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║              🧠 ML-SCHEDULER JUPYTER DEPLOYMENT              ║
║                                                               ║
║  Deploying advanced Jupyter environment in Kubeflow         ║
║  ├── Custom ML image with XGBoost, PyTorch, Scikit-learn     ║
║  ├── Kubernetes clients & Prometheus integration             ║
║  ├── Feast feature store connectivity                        ║
║  └── Pre-configured notebooks for ML development             ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

NAMESPACE="wassimmezrani"
NOTEBOOK_NAME="ml-scheduler-dev"
NOTEBOOK_CPU="4"
NOTEBOOK_MEMORY="16Gi"
NOTEBOOK_STORAGE="100Gi"

echo -e "${BLUE}[INFO]${NC} Deploying ML-Scheduler notebook in namespace: ${NAMESPACE}"

# Create the enhanced Kubeflow Notebook with ML libraries
kubectl apply -f - << EOF
apiVersion: kubeflow.org/v1
kind: Notebook
metadata:
  name: ${NOTEBOOK_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ml-scheduler
    component: jupyter-development
    ml-scheduler.hydatis.com/enabled: "true"
  annotations:
    description: "Advanced ML development environment for ML-Scheduler"
    contact: "wassim.mezrani@hydatis.com"
spec:
  template:
    spec:
      serviceAccountName: default-editor
      containers:
      - name: ${NOTEBOOK_NAME}
        image: jupyter/datascience-notebook:python-3.11
        command:
          - sh
          - -c
          - |
            # Install additional ML libraries for ML-Scheduler
            pip install --quiet --no-cache-dir \\
              xgboost==2.0.3 \\
              torch==2.1.2 \\
              torchvision==0.16.2 \\
              scikit-learn==1.3.2 \\
              pandas==2.1.4 \\
              numpy==1.24.4 \\
              matplotlib==3.8.2 \\
              seaborn==0.13.0 \\
              plotly==5.17.0 \\
              kubernetes==28.1.0 \\
              prometheus-api-client==0.5.3 \\
              feast[redis]==0.34.1 \\
              mlflow==2.8.1 \\
              optuna==3.4.0 \\
              shap==0.44.0 \\
              lime==0.2.0.1 \\
              gym==0.29.1 \\
              stable-baselines3==2.2.1 \\
              redis==5.0.1 \\
              psutil==5.9.6 \\
              jupyter-ai==2.8.0
            
            # Install Kubeflow SDK
            pip install --quiet --no-cache-dir \\
              kfp==2.5.0 \\
              kfp-server-api==2.0.5
            
            # Create ML-Scheduler notebooks directory
            mkdir -p /home/jovyan/ml-scheduler/{data,notebooks,models,experiments}
            
            # Download initial notebook templates
            echo "Creating ML-Scheduler notebook templates..."
            
            # Start Jupyter Lab
            start-notebook.sh --ServerApp.ip=0.0.0.0 \\
                            --ServerApp.port=8888 \\
                            --ServerApp.token='' \\
                            --ServerApp.password='' \\
                            --ServerApp.allow_root=True \\
                            --ServerApp.base_url='/notebook/${NAMESPACE}/${NOTEBOOK_NAME}/'
        
        env:
        - name: JUPYTER_ENABLE_LAB
          value: "yes"
        - name: CHOWN_HOME
          value: "yes"
        - name: CHOWN_HOME_OPTS
          value: "-R"
        - name: NB_UID
          value: "1000"
        - name: NB_GID
          value: "100"
        - name: PROMETHEUS_URL
          value: "http://prometheus-k8s.monitoring.svc.cluster.local:9090"
        - name: FEAST_REGISTRY_PATH
          value: "/home/jovyan/ml-scheduler/data/feast-registry"
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server.kubeflow.svc.cluster.local:5000"
        
        ports:
        - containerPort: 8888
          name: notebook-port
          protocol: TCP
        
        resources:
          requests:
            cpu: "${NOTEBOOK_CPU}"
            memory: "${NOTEBOOK_MEMORY}"
          limits:
            cpu: "8"
            memory: "32Gi"
        
        volumeMounts:
        - name: workspace-${NOTEBOOK_NAME}
          mountPath: /home/jovyan
        - name: dshm
          mountPath: /dev/shm
        
        workingDir: /home/jovyan
        
      volumes:
      - name: workspace-${NOTEBOOK_NAME}
        persistentVolumeClaim:
          claimName: workspace-${NOTEBOOK_NAME}
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: 2Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: workspace-${NOTEBOOK_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: ml-scheduler
    component: storage
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: ${NOTEBOOK_STORAGE}
  storageClassName: longhorn
EOF

echo -e "${YELLOW}[INFO]${NC} Waiting for notebook to be ready..."
kubectl wait --for=condition=ready pod -l app=ml-scheduler -n ${NAMESPACE} --timeout=300s

echo -e "${GREEN}[SUCCESS]${NC} ML-Scheduler Jupyter notebook deployed successfully!"

# Get the notebook URL
KUBEFLOW_URL=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
if [ "$KUBEFLOW_URL" = "localhost" ]; then
    KUBEFLOW_URL=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.spec.clusterIP}')
fi

echo -e "\n${BLUE}📊 Access Information:${NC}"
echo -e "├── ${YELLOW}Kubeflow Central Dashboard:${NC} http://${KUBEFLOW_URL}"
echo -e "├── ${YELLOW}Notebook Direct URL:${NC} http://${KUBEFLOW_URL}/notebook/${NAMESPACE}/${NOTEBOOK_NAME}/"
echo -e "└── ${YELLOW}Namespace:${NC} ${NAMESPACE}"

echo -e "\n${BLUE}🧠 Next Steps:${NC}"
echo -e "├── 1️⃣  Access Kubeflow Dashboard"
echo -e "├── 2️⃣  Navigate to Notebooks section"
echo -e "├── 3️⃣  Open '${NOTEBOOK_NAME}' notebook"
echo -e "├── 4️⃣  Start developing ML algorithms"
echo -e "└── 5️⃣  Create notebooks for EDA, XGBoost, Q-Learning, Isolation Forest"

echo -e "\n${GREEN}🚀 Ready to develop the world's first AI-powered Kubernetes scheduler! 🧠⚡${NC}"
