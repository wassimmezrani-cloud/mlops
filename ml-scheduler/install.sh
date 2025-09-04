#!/bin/bash

# 🧠 ML-Scheduler Installation Script
# Automated setup for Charmed Kubeflow + ML-Scheduler

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Banner
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                    🧠 ML-SCHEDULER INSTALLER                  ║
║                                                               ║
║  Revolutionary AI-Powered Kubernetes Scheduler               ║
║  ├── Charmed Kubeflow MLOps Platform                         ║
║  ├── 3 ML Algorithms (XGBoost + Q-Learning + Isolation)      ║
║  ├── Real-time Feast Feature Store                           ║
║  └── Production-Ready Scheduler Plugin                       ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Configuration
KUBEFLOW_VERSION="1.8/stable"
NAMESPACE_KUBEFLOW="kubeflow"
NAMESPACE_ML_SCHEDULER="ml-scheduler"
LONGHORN_VERSION="v1.6.0"

# Pre-requisites check
check_prerequisites() {
    log_info "🔍 Checking prerequisites..."
    
    # Check if kubectl is installed and configured
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi
    
    # Check Kubernetes cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    # Check if juju is installed
    if ! command -v juju &> /dev/null; then
        log_warn "Juju is not installed. Installing Juju..."
        install_juju
    fi
    
    # Check available resources
    log_info "Checking cluster resources..."
    kubectl top nodes || log_warn "Metrics server not available - continuing anyway"
    
    log_success "Prerequisites check completed"
}

# Install Juju if not present
install_juju() {
    log_info "📦 Installing Juju..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo snap install juju --classic
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install juju
    else
        log_error "Unsupported operating system. Please install Juju manually."
        exit 1
    fi
    
    log_success "Juju installed successfully"
}

# Setup Longhorn Storage
setup_longhorn() {
    log_info "💾 Setting up Longhorn storage..."
    
    # Check if Longhorn is already installed
    if kubectl get namespace longhorn-system &> /dev/null; then
        log_warn "Longhorn already installed, skipping..."
        return
    fi
    
    # Install Longhorn
    kubectl apply -f https://raw.githubusercontent.com/longhorn/longhorn/${LONGHORN_VERSION}/deploy/longhorn.yaml
    
    # Wait for Longhorn to be ready
    log_info "⏳ Waiting for Longhorn to be ready..."
    kubectl wait --for=condition=ready pod --all -n longhorn-system --timeout=600s
    
    # Set Longhorn as default storage class
    kubectl patch storageclass longhorn -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
    
    log_success "Longhorn storage configured successfully"
}

# Deploy Charmed Kubeflow
deploy_kubeflow() {
    log_info "🚀 Deploying Charmed Kubeflow..."
    
    # Bootstrap Juju if needed
    if ! juju list-controllers | grep -q microk8s; then
        log_info "Bootstrapping Juju controller..."
        juju bootstrap microk8s mk8s
    fi
    
    # Add model for Kubeflow
    if ! juju list-models | grep -q kubeflow; then
        log_info "Creating Kubeflow model..."
        juju add-model kubeflow
    fi
    
    # Deploy Kubeflow bundle
    log_info "📦 Deploying Kubeflow components..."
    juju deploy kubeflow --channel=${KUBEFLOW_VERSION} --trust
    
    # Wait for deployment
    log_info "⏳ Waiting for Kubeflow deployment (this may take 15-30 minutes)..."
    juju wait-for application kubeflow-dashboard --query='status=="active"' --timeout=1800s
    
    # Configure ingress
    setup_kubeflow_ingress
    
    log_success "Kubeflow deployed successfully"
}

# Setup Kubeflow Ingress
setup_kubeflow_ingress() {
    log_info "🌐 Configuring Kubeflow ingress..."
    
    # Create ingress for Kubeflow dashboard
    kubectl apply -f - << EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: kubeflow-gateway
  namespace: kubeflow
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
  - host: kubeflow.mlops.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: kubeflow-dashboard
            port:
              number: 80
EOF
    
    log_success "Kubeflow ingress configured"
}

# Setup Prometheus for extended monitoring
setup_prometheus() {
    log_info "📊 Configuring Prometheus for ML-Scheduler..."
    
    # Extend Prometheus retention and add custom scrape configs
    kubectl apply -f - << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-ml-config
  namespace: monitoring
data:
  additional.yml: |
    - job_name: 'kubernetes-pods-ml'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - ml-scheduler
          - kubeflow
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
EOF
    
    # Increase retention to 30 days
    kubectl patch prometheus prometheus-k8s -n monitoring --type='merge' -p='{"spec":{"retention":"30d","retentionSize":"50GB"}}'
    
    log_success "Prometheus configured for ML data collection"
}

# Deploy Feast Feature Store
deploy_feast() {
    log_info "🍽️ Deploying Feast Feature Store..."
    
    kubectl create namespace feast-system 2>/dev/null || true
    
    # Deploy Feast using Helm
    helm repo add feast-charts https://feast-helm-charts.storage.googleapis.com
    helm repo update
    
    helm install feast feast-charts/feast-feature-server \
        --namespace feast-system \
        --set feast.core.enabled=true \
        --set feast.online_store.enabled=true \
        --set feast.offline_store.enabled=true
    
    log_success "Feast Feature Store deployed"
}

# Create ML-Scheduler namespace and RBAC
setup_ml_scheduler() {
    log_info "🤖 Setting up ML-Scheduler namespace..."
    
    kubectl apply -f - << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE_ML_SCHEDULER}
  labels:
    app: ml-scheduler
    version: v1.0.0
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ml-scheduler
  namespace: ${NAMESPACE_ML_SCHEDULER}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: ml-scheduler
rules:
- apiGroups: [""]
  resources: ["nodes", "pods", "services", "endpoints"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch"]
- apiGroups: ["metrics.k8s.io"]
  resources: ["nodes", "pods"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: ml-scheduler
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: ml-scheduler
subjects:
- kind: ServiceAccount
  name: ml-scheduler
  namespace: ${NAMESPACE_ML_SCHEDULER}
EOF
    
    log_success "ML-Scheduler namespace and RBAC configured"
}

# Deploy Jupyter development environment
deploy_jupyter() {
    log_info "📓 Setting up Jupyter development environment..."
    
    kubectl apply -f - << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-jupyter
  namespace: ${NAMESPACE_ML_SCHEDULER}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ml-jupyter
  template:
    metadata:
      labels:
        app: ml-jupyter
    spec:
      containers:
      - name: jupyter
        image: jupyter/datascience-notebook:latest
        ports:
        - containerPort: 8888
        env:
        - name: JUPYTER_ENABLE_LAB
          value: "yes"
        - name: JUPYTER_TOKEN
          value: "ml-scheduler-2025"
        volumeMounts:
        - name: notebooks
          mountPath: /home/jovyan/work
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
      volumes:
      - name: notebooks
        persistentVolumeClaim:
          claimName: jupyter-pvc
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: jupyter-pvc
  namespace: ${NAMESPACE_ML_SCHEDULER}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
---
apiVersion: v1
kind: Service
metadata:
  name: ml-jupyter
  namespace: ${NAMESPACE_ML_SCHEDULER}
spec:
  selector:
    app: ml-jupyter
  ports:
  - port: 8888
    targetPort: 8888
  type: LoadBalancer
EOF
    
    log_success "Jupyter development environment deployed"
}

# Display access information
show_access_info() {
    echo -e "\n${GREEN}🎉 ML-Scheduler Installation Complete! 🎉${NC}\n"
    
    echo -e "${BLUE}📊 Access Information:${NC}"
    echo -e "├── ${YELLOW}Kubeflow Dashboard:${NC} http://kubeflow.mlops.local"
    echo -e "├── ${YELLOW}Grafana Monitoring:${NC} http://$(kubectl get svc grafana-nodeport -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):3000"
    echo -e "├── ${YELLOW}Prometheus:${NC} http://$(kubectl get svc prometheus-k8s-nodeport -n monitoring -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):9090"
    echo -e "└── ${YELLOW}Jupyter Lab:${NC} http://$(kubectl get svc ml-jupyter -n ml-scheduler -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8888"
    
    echo -e "\n${BLUE}🔑 Credentials:${NC}"
    echo -e "├── ${YELLOW}Grafana:${NC} admin / admin"
    echo -e "└── ${YELLOW}Jupyter Token:${NC} ml-scheduler-2025"
    
    echo -e "\n${BLUE}📁 Next Steps:${NC}"
    echo -e "├── 1️⃣  Access Jupyter Lab to start ML development"
    echo -e "├── 2️⃣  Review historical data in Prometheus"
    echo -e "├── 3️⃣  Start EDA notebook development"
    echo -e "└── 4️⃣  Follow the roadmap in ROADMAP.md"
    
    echo -e "\n${GREEN}🚀 Ready to build the world's first AI-powered Kubernetes scheduler! 🧠⚡${NC}"
}

# Main installation flow
main() {
    log_info "Starting ML-Scheduler installation..."
    
    check_prerequisites
    setup_longhorn
    deploy_kubeflow
    setup_prometheus
    deploy_feast
    setup_ml_scheduler
    deploy_jupyter
    
    show_access_info
}

# Handle script interruption
trap 'log_error "Installation interrupted"; exit 1' INT TERM

# Run main installation
main "$@"
