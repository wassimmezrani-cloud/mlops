#!/bin/bash

# ÉTAPE 10 - LANCEMENT EXPÉRIENCES KATIB HYPERPARAMETER OPTIMIZATION
# Script de déploiement et lancement des expériences d'optimisation ML-Scheduler

set -euo pipefail

echo "🚀 ÉTAPE 10 - OPTIMISATION KATIB HYPERPARAMETERS"
echo "================================================"

# Configuration
KUBECONFIG=${KUBECONFIG:-~/.kube/config}
NAMESPACE="kubeflow"
KATIB_NAMESPACE="katib-system"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Action 1: Setup Katib Environment (45 min)
setup_katib_environment() {
    log_info "Action 1: Setup Katib Environment"
    
    # Create namespaces if they don't exist
    kubectl create namespace ${KATIB_NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    
    # Install Katib
    log_info "Installing Katib components..."
    kubectl apply -f katib-setup.yaml
    
    # Wait for Katib controller to be ready
    log_info "Waiting for Katib controller to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/katib-controller -n ${KATIB_NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/katib-ui -n ${KATIB_NAMESPACE}
    
    # Verify Katib installation
    kubectl get pods -n ${KATIB_NAMESPACE}
    
    log_success "Katib environment setup completed"
}

# Action 2: Launch XGBoost Experiments (60 min)
launch_xgboost_experiments() {
    log_info "Action 2: Launching XGBoost 'Le Prophète' Hyperparameter Experiments"
    
    # Apply XGBoost experiment configuration
    kubectl apply -f xgboost-experiment.yaml
    
    # Monitor experiment launch
    kubectl get experiment xgboost-load-predictor-optimization -n ${NAMESPACE} -w --timeout=60s || true
    
    log_success "XGBoost experiments launched - 150 trials with TPE algorithm"
}

# Action 3: Launch Q-Learning Experiments (60 min)  
launch_qlearning_experiments() {
    log_info "Action 3: Launching Q-Learning 'L'Optimiseur' Hyperparameter Experiments"
    
    # Apply Q-Learning experiment configuration
    kubectl apply -f qlearning-experiment.yaml
    
    # Start TensorBoard for visualization
    kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: qlearning-tensorboard-ingress
  namespace: ${NAMESPACE}
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: tensorboard.ml-scheduler.hydatis.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: qlearning-tensorboard
            port:
              number: 6006
EOF
    
    # Monitor experiment launch
    kubectl get experiment qlearning-placement-optimizer-optimization -n ${NAMESPACE} -w --timeout=60s || true
    
    log_success "Q-Learning experiments launched - 120 trials with Bayesian Optimization"
    log_info "TensorBoard available at: http://tensorboard.ml-scheduler.hydatis.local"
}

# Action 4: Launch Isolation Forest Experiments (45 min)
launch_isolation_experiments() {
    log_info "Action 4: Launching Isolation Forest 'Le Détective' Hyperparameter Experiments"
    
    # First launch data collection job
    kubectl apply -f isolation-forest-experiment.yaml
    
    # Wait for data collection to complete
    log_info "Waiting for data collection to complete..."
    kubectl wait --for=condition=complete --timeout=600s job/isolation-forest-data-collection -n ${NAMESPACE} || log_warning "Data collection taking longer than expected"
    
    # Monitor experiment launch
    kubectl get experiment isolation-anomaly-detector-optimization -n ${NAMESPACE} -w --timeout=60s || true
    
    log_success "Isolation Forest experiments launched - 80 trials with Random Search"
}

# Action 5: Monitor All Experiments
monitor_experiments() {
    log_info "Action 5: Monitoring All Experiments"
    
    echo "📊 EXPERIMENT STATUS DASHBOARD"
    echo "=============================="
    
    # Function to get experiment status
    get_experiment_status() {
        local exp_name=$1
        local status=$(kubectl get experiment ${exp_name} -n ${NAMESPACE} -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo "NotFound")
        local succeeded=$(kubectl get experiment ${exp_name} -n ${NAMESPACE} -o jsonpath='{.status.trialsSucceeded}' 2>/dev/null || echo "0")
        local running=$(kubectl get experiment ${exp_name} -n ${NAMESPACE} -o jsonpath='{.status.trialsRunning}' 2>/dev/null || echo "0")
        local failed=$(kubectl get experiment ${exp_name} -n ${NAMESPACE} -o jsonpath='{.status.trialsFailed}' 2>/dev/null || echo "0")
        
        echo "  ${exp_name}:"
        echo "    Status: ${status}"
        echo "    Succeeded: ${succeeded}"
        echo "    Running: ${running}" 
        echo "    Failed: ${failed}"
        echo ""
    }
    
    while true; do
        clear
        echo "📊 KATIB EXPERIMENTS STATUS - $(date)"
        echo "========================================="
        echo ""
        
        get_experiment_status "xgboost-load-predictor-optimization"
        get_experiment_status "qlearning-placement-optimizer-optimization" 
        get_experiment_status "isolation-anomaly-detector-optimization"
        
        # Check if all experiments are completed
        local all_complete=true
        for exp in "xgboost-load-predictor-optimization" "qlearning-placement-optimizer-optimization" "isolation-anomaly-detector-optimization"; do
            local status=$(kubectl get experiment ${exp} -n ${NAMESPACE} -o jsonpath='{.status.conditions[0].type}' 2>/dev/null || echo "NotFound")
            if [[ "${status}" != "Succeeded" && "${status}" != "Failed" ]]; then
                all_complete=false
            fi
        done
        
        if [[ "${all_complete}" == "true" ]]; then
            log_success "All experiments completed!"
            break
        fi
        
        echo "⏰ Refreshing in 60 seconds... (Ctrl+C to stop monitoring)"
        sleep 60
    done
}

# Generate experiment summary report
generate_summary_report() {
    log_info "Generating Experiment Summary Report"
    
    local report_file="/tmp/katib-experiments-summary-$(date +%Y%m%d-%H%M%S).md"
    
    cat > ${report_file} << EOF
# KATIB HYPERPARAMETER OPTIMIZATION RESULTS
## Generated: $(date)

### Experiment Overview
- **XGBoost "Le Prophète"**: Load Prediction Optimization
- **Q-Learning "L'Optimiseur"**: Placement Strategy Optimization  
- **Isolation Forest "Le Détective"**: Anomaly Detection Optimization

### Results Summary

#### XGBoost Load Predictor
\`\`\`
$(kubectl get experiment xgboost-load-predictor-optimization -n ${NAMESPACE} -o yaml 2>/dev/null | grep -A 10 "bestTrialName" || echo "Experiment not found or incomplete")
\`\`\`

#### Q-Learning Placement Optimizer
\`\`\`
$(kubectl get experiment qlearning-placement-optimizer-optimization -n ${NAMESPACE} -o yaml 2>/dev/null | grep -A 10 "bestTrialName" || echo "Experiment not found or incomplete")
\`\`\`

#### Isolation Forest Anomaly Detector
\`\`\`
$(kubectl get experiment isolation-anomaly-detector-optimization -n ${NAMESPACE} -o yaml 2>/dev/null | grep -A 10 "bestTrialName" || echo "Experiment not found or incomplete")
\`\`\`

### Next Steps
1. Export optimized models to MLflow Registry
2. Update ML-Scheduler plugin configuration
3. Deploy optimized models to KServe
4. Begin integration testing (Étape 11)
EOF
    
    log_success "Summary report generated: ${report_file}"
    cat ${report_file}
}

# Main execution
main() {
    log_info "Starting ÉTAPE 10 - Katib Hyperparameter Optimization"
    log_info "Target: 350+ experiments across 3 ML models"
    echo ""
    
    # Check prerequisites
    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check kubeconfig."
        exit 1
    fi
    
    # Execute actions sequentially
    setup_katib_environment
    echo ""
    
    launch_xgboost_experiments
    echo ""
    
    launch_qlearning_experiments  
    echo ""
    
    launch_isolation_experiments
    echo ""
    
    # Ask user if they want to monitor
    read -p "🔍 Monitor experiments in real-time? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        monitor_experiments
    fi
    
    generate_summary_report
    
    log_success "ÉTAPE 10 completed! Ready for model selection and deployment."
    
    echo ""
    echo "🔗 USEFUL LINKS:"
    echo "  - Katib UI: kubectl port-forward svc/katib-ui 8080:80 -n katib-system"
    echo "  - TensorBoard: http://tensorboard.ml-scheduler.hydatis.local"
    echo "  - MLflow: kubectl port-forward svc/mlflow-server 5000:5000 -n kubeflow"
}

# Trap for cleanup
trap 'log_warning "Script interrupted. Experiments may continue running in background."' INT TERM

# Run main function
main "$@"