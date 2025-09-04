#!/bin/bash

# 📓 Jupyter ML Environment Deployment Script
# Deploys Jupyter Lab with ML libraries for ML-Scheduler development

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

NAMESPACE="ml-scheduler"
KUBECONFIG_SECRET="ml-jupyter-kubeconfig"

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Banner
echo -e "${GREEN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║              📓 ML JUPYTER ENVIRONMENT SETUP             ║
║                                                           ║
║  Advanced Data Science Environment for ML-Scheduler      ║
║  ├── Jupyter Lab with ML/AI Libraries                    ║
║  ├── Kubernetes API Access                               ║
║  ├── Prometheus Integration                               ║
║  └── MLflow & Kubeflow Integration                        ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Create kubeconfig secret for Jupyter
create_kubeconfig_secret() {
    log_info "🔑 Creating kubeconfig secret for Jupyter..."
    
    # Get current kubeconfig
    local kubeconfig_content
    if [[ -f ~/.kube/config ]]; then
        kubeconfig_content=$(cat ~/.kube/config | base64 -w 0)
    else
        log_error "No kubeconfig found at ~/.kube/config"
        exit 1
    fi
    
    # Create secret
    kubectl create secret generic ${KUBECONFIG_SECRET} \
        --namespace=${NAMESPACE} \
        --from-literal=config="$(echo $kubeconfig_content | base64 -d)" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_success "Kubeconfig secret created"
}

# Deploy Jupyter environment
deploy_jupyter() {
    log_info "🚀 Deploying Jupyter ML environment..."
    
    # Apply manifests
    kubectl apply -f /home/wassim/Desktop/mlops/ml-scheduler/manifests/namespace.yaml
    kubectl apply -f /home/wassim/Desktop/mlops/ml-scheduler/manifests/jupyter.yaml
    
    log_success "Jupyter manifests applied"
}

# Wait for deployment to be ready
wait_for_jupyter() {
    log_info "⏳ Waiting for Jupyter deployment to be ready..."
    
    kubectl wait --for=condition=available deployment/ml-jupyter \
        --namespace=${NAMESPACE} --timeout=600s
    
    kubectl wait --for=condition=ready pod -l app=ml-jupyter \
        --namespace=${NAMESPACE} --timeout=300s
    
    log_success "Jupyter is ready"
}

# Get access information
get_access_info() {
    log_info "📋 Getting Jupyter access information..."
    
    # Get LoadBalancer IP
    local external_ip=""
    local max_attempts=30
    local attempt=1
    
    while [[ -z "$external_ip" && $attempt -le $max_attempts ]]; do
        external_ip=$(kubectl get svc ml-jupyter -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "")
        if [[ -z "$external_ip" || "$external_ip" == "null" ]]; then
            log_info "Waiting for LoadBalancer IP assignment... (${attempt}/${max_attempts})"
            sleep 10
            ((attempt++))
        fi
    done
    
    if [[ -z "$external_ip" ]]; then
        # Fallback to NodePort
        local node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
        local node_port=$(kubectl get svc ml-jupyter -n ${NAMESPACE} -o jsonpath='{.spec.ports[0].nodePort}')
        external_ip="${node_ip}:${node_port}"
        log_warn "LoadBalancer IP not available, using NodePort: ${external_ip}"
    fi
    
    return 0
}

# Display access information
show_access_info() {
    local jupyter_url="$1"
    
    echo -e "\n${GREEN}🎉 Jupyter ML Environment Deployed Successfully! 🎉${NC}\n"
    
    echo -e "${BLUE}📊 Access Information:${NC}"
    echo -e "├── ${YELLOW}Jupyter Lab:${NC} http://${jupyter_url}:8888"
    echo -e "├── ${YELLOW}Token:${NC} ml-scheduler-2025"
    echo -e "├── ${YELLOW}Alternative URL:${NC} http://jupyter.mlops.local (if DNS configured)"
    echo -e "└── ${YELLOW}Namespace:${NC} ${NAMESPACE}"
    
    echo -e "\n${BLUE}📚 Pre-installed Libraries:${NC}"
    echo -e "├── 🧠 ML Frameworks: XGBoost, LightGBM, CatBoost, Scikit-learn"
    echo -e "├── 🔄 RL Libraries: Gym, Stable-Baselines3"
    echo -e "├── 📊 MLOps: MLflow, Kubeflow Pipelines, Feast"
    echo -e "├── 🔍 Explainability: SHAP, LIME"
    echo -e "├── 📈 Monitoring: Evidently, Great Expectations"
    echo -e "├── 🎯 Optimization: Optuna"
    echo -e "├── 📊 Visualization: Plotly, Dash, Streamlit"
    echo -e "└── ⚙️  Infrastructure: Kubernetes client, Prometheus client"
    
    echo -e "\n${BLUE}📂 Workspace Structure:${NC}"
    echo -e "├── 📓 /home/jovyan/work/notebooks/ - Jupyter notebooks"
    echo -e "├── 📊 /home/jovyan/work/data/ - Datasets and features"
    echo -e "├── 🤖 /home/jovyan/work/models/ - Trained models"
    echo -e "├── 🧪 /home/jovyan/work/experiments/ - ML experiments"
    echo -e "└── 🚀 /home/jovyan/work/pipelines/ - ML pipelines"
    
    echo -e "\n${BLUE}🔧 Configuration:${NC}"
    echo -e "├── ${YELLOW}Prometheus:${NC} http://10.110.190.83:9090"
    echo -e "├── ${YELLOW}Grafana:${NC} http://10.110.190.84:3000"
    echo -e "├── ${YELLOW}Storage:${NC} 100Gi Longhorn persistent volume"
    echo -e "├── ${YELLOW}Resources:${NC} 4-16Gi RAM, 2-8 CPU cores"
    echo -e "└── ${YELLOW}Kubernetes Access:${NC} Full cluster access via service account"
    
    echo -e "\n${BLUE}🚀 Next Steps:${NC}"
    echo -e "├── 1️⃣  Access Jupyter Lab using the URL above"
    echo -e "├── 2️⃣  Navigate to /work/notebooks directory"
    echo -e "├── 3️⃣  Start with EDA notebook for historical data analysis"
    echo -e "├── 4️⃣  Explore Prometheus metrics and cluster patterns"
    echo -e "└── 5️⃣  Begin feature engineering for ML algorithms"
    
    echo -e "\n${GREEN}🧠 Ready to revolutionize Kubernetes scheduling with AI! ⚡${NC}"
}

# Main execution
main() {
    log_info "Starting Jupyter ML environment deployment..."
    
    # Check prerequisites
    if ! kubectl get namespace ${NAMESPACE} &>/dev/null; then
        log_error "Namespace ${NAMESPACE} does not exist. Please run namespace setup first."
        exit 1
    fi
    
    # Deploy components
    create_kubeconfig_secret
    deploy_jupyter
    wait_for_jupyter
    
    # Get access info
    get_access_info
    local jupyter_endpoint=$(kubectl get svc ml-jupyter -n ${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' || echo "pending")
    
    if [[ "$jupyter_endpoint" == "pending" ]]; then
        local node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
        jupyter_endpoint="$node_ip"
    fi
    
    show_access_info "$jupyter_endpoint"
}

# Handle interruption
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main
main "$@"
