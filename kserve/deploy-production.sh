#!/bin/bash

# ML-Scheduler KServe Production Deployment Script
# Deploys the complete production stack with monitoring and security

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="ml-scheduler"
MONITORING_NAMESPACE="monitoring"
ISTIO_NAMESPACE="istio-system"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl not found. Please install kubectl."
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        error "Cannot connect to Kubernetes cluster."
    fi
    
    # Check KServe CRDs
    if ! kubectl get crd inferenceservices.serving.kserve.io &> /dev/null; then
        error "KServe CRDs not found. Please install KServe first."
    fi
    
    # Check Istio
    if ! kubectl get namespace istio-system &> /dev/null; then
        error "Istio not found. Please install Istio service mesh first."
    fi
    
    success "Prerequisites check passed"
}

# Create namespaces
create_namespaces() {
    log "Creating namespaces..."
    
    # Create ml-scheduler namespace with Istio injection
    kubectl create namespace ${NAMESPACE} --dry-run=client -o yaml | kubectl apply -f -
    kubectl label namespace ${NAMESPACE} istio-injection=enabled --overwrite
    kubectl label namespace ${NAMESPACE} serving.kserve.io/inferenceservice=enabled --overwrite
    
    # Create monitoring namespace if it doesn't exist
    kubectl create namespace ${MONITORING_NAMESPACE} --dry-run=client -o yaml | kubectl apply -f - || true
    
    success "Namespaces created and configured"
}

# Deploy RBAC and security
deploy_security() {
    log "Deploying RBAC and security configurations..."
    
    kubectl apply -f ./production/kserve-rbac.yaml
    kubectl apply -f ./security/network-policies.yaml
    kubectl apply -f ./security/security-policies.yaml
    
    success "Security configurations deployed"
}

# Deploy high availability configurations
deploy_ha_config() {
    log "Deploying high availability configurations..."
    
    kubectl apply -f ./high-availability/pod-disruption-budgets.yaml
    kubectl apply -f ./high-availability/horizontal-pod-autoscalers.yaml
    kubectl apply -f ./high-availability/persistent-volumes.yaml
    
    success "High availability configurations deployed"
}

# Deploy KServe InferenceServices
deploy_inference_services() {
    log "Deploying KServe InferenceServices..."
    
    # Deploy XGBoost Predictor
    log "Deploying XGBoost Load Predictor..."
    kubectl apply -f ./production/xgboost-inference-service.yaml
    
    # Deploy Q-Learning Optimizer  
    log "Deploying Q-Learning Placement Optimizer..."
    kubectl apply -f ./production/qlearning-inference-service.yaml
    
    # Deploy Isolation Forest Detector
    log "Deploying Isolation Forest Anomaly Detector..."
    kubectl apply -f ./production/isolation-inference-service.yaml
    
    success "KServe InferenceServices deployed"
}

# Deploy Istio configurations
deploy_istio_config() {
    log "Deploying Istio gateway and traffic management..."
    
    kubectl apply -f ./security/istio-gateway.yaml
    kubectl apply -f ./security/destination-rules.yaml
    
    success "Istio configurations deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    log "Deploying monitoring and observability stack..."
    
    kubectl apply -f ./monitoring/service-monitor.yaml
    kubectl apply -f ./monitoring/prometheus-rules.yaml
    kubectl apply -f ./monitoring/grafana-dashboards.yaml
    kubectl apply -f ./monitoring/alertmanager-config.yaml
    
    success "Monitoring stack deployed"
}

# Deploy performance configurations
deploy_performance_config() {
    log "Deploying performance optimization configurations..."
    
    kubectl apply -f ./performance/performance-tuning.yaml
    
    success "Performance configurations deployed"
}

# Wait for deployments
wait_for_deployments() {
    log "Waiting for InferenceServices to be ready..."
    
    local services=("xgboost-load-predictor" "qlearning-placement-optimizer" "isolation-anomaly-detector")
    
    for service in "${services[@]}"; do
        log "Waiting for ${service} to be ready..."
        kubectl wait --for=condition=Ready inferenceservice/${service} -n ${NAMESPACE} --timeout=300s
        
        if kubectl get inferenceservice/${service} -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' | grep -q "True"; then
            success "${service} is ready"
        else
            warning "${service} may not be fully ready. Check status manually."
        fi
    done
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    echo -e "\n${PURPLE}=== KSERVE INFERENCE SERVICES STATUS ===${NC}"
    kubectl get inferenceservices -n ${NAMESPACE} -o wide
    
    echo -e "\n${PURPLE}=== PODS STATUS ===${NC}"
    kubectl get pods -n ${NAMESPACE} -o wide
    
    echo -e "\n${PURPLE}=== SERVICES STATUS ===${NC}"  
    kubectl get services -n ${NAMESPACE}
    
    echo -e "\n${PURPLE}=== ISTIO GATEWAY STATUS ===${NC}"
    kubectl get gateways -n ${NAMESPACE}
    kubectl get virtualservices -n ${NAMESPACE}
    kubectl get destinationrules -n ${NAMESPACE}
    
    echo -e "\n${PURPLE}=== HORIZONTAL POD AUTOSCALERS ===${NC}"
    kubectl get hpa -n ${NAMESPACE}
    
    echo -e "\n${PURPLE}=== MONITORING RESOURCES ===${NC}"
    kubectl get servicemonitors -n ${NAMESPACE}
    kubectl get prometheusrules -n ${NAMESPACE}
    
    success "Deployment verification completed"
}

# Test endpoints
test_endpoints() {
    log "Testing service endpoints..."
    
    # Get gateway external IP
    local gateway_ip
    gateway_ip=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
    
    if [[ "$gateway_ip" == "localhost" ]]; then
        warning "No external LoadBalancer IP found. Using port-forward for testing."
        
        # Test with port-forward
        log "Testing services via port-forward..."
        
        # Port-forward XGBoost service
        kubectl port-forward -n ${NAMESPACE} svc/xgboost-load-predictor-predictor-default 8081:80 &
        local pf_pid1=$!
        sleep 5
        
        if curl -s -o /dev/null -w "%{http_code}" "http://localhost:8081/v1/models/xgboost-load-predictor" | grep -q "200\|404"; then
            success "XGBoost service responding"
        else
            warning "XGBoost service may not be responding correctly"
        fi
        
        kill $pf_pid1 2>/dev/null || true
        
    else
        log "Testing endpoints via external gateway IP: ${gateway_ip}"
        
        # Test XGBoost endpoint
        if curl -s -k "https://${gateway_ip}/xgboost/v1/models/xgboost-load-predictor" -o /dev/null; then
            success "XGBoost endpoint accessible"
        else
            warning "XGBoost endpoint may not be accessible"
        fi
    fi
}

# Run performance tests
run_performance_tests() {
    log "Running basic performance validation..."
    
    if [[ -f "./performance/load-testing-script.py" ]]; then
        log "Load testing script found. You can run performance tests with:"
        echo "  python3 ./performance/load-testing-script.py"
    else
        warning "Load testing script not found. Skipping performance tests."
    fi
}

# Main deployment function
main() {
    echo -e "${PURPLE}"
    echo "================================================================================"
    echo "                 ML-SCHEDULER KSERVE PRODUCTION DEPLOYMENT"
    echo "================================================================================"
    echo -e "${NC}"
    
    log "Starting ML-Scheduler KServe production deployment..."
    
    # Check current directory
    if [[ ! -f "./production/xgboost-inference-service.yaml" ]]; then
        error "Please run this script from the kserve directory containing the manifests."
    fi
    
    # Execute deployment steps
    check_prerequisites
    create_namespaces
    deploy_security
    deploy_ha_config
    deploy_performance_config
    deploy_inference_services
    deploy_istio_config
    deploy_monitoring
    
    log "Waiting for services to stabilize..."
    sleep 30
    
    wait_for_deployments
    verify_deployment
    test_endpoints
    run_performance_tests
    
    echo -e "\n${GREEN}"
    echo "================================================================================"
    echo "                    🎉 DEPLOYMENT COMPLETED SUCCESSFULLY! 🎉"
    echo "================================================================================"
    echo -e "${NC}"
    
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Verify all services are healthy: kubectl get inferenceservices -n ${NAMESPACE}"
    echo "2. Check monitoring dashboards in Grafana"
    echo "3. Run load tests: python3 ./performance/load-testing-script.py"  
    echo "4. Configure DNS entries for production domains"
    echo "5. Review alerting configurations and test notifications"
    
    echo -e "\n${BLUE}Service Endpoints:${NC}"
    echo "• XGBoost Predictor: https://xgboost.ml-scheduler.hydatis.local/v1/models/xgboost-load-predictor:predict"
    echo "• Q-Learning Optimizer: https://qlearning.ml-scheduler.hydatis.local/v1/models/qlearning-placement-optimizer:predict"
    echo "• Isolation Detector: https://isolation.ml-scheduler.hydatis.local/v1/models/isolation-anomaly-detector:predict"
    
    echo -e "\n${BLUE}Monitoring:${NC}"
    echo "• Prometheus Rules: kubectl get prometheusrules -n ${NAMESPACE}"
    echo "• Service Monitors: kubectl get servicemonitors -n ${NAMESPACE}"
    echo "• Grafana Dashboards: Check monitoring namespace ConfigMaps"
    
    success "ML-Scheduler KServe production deployment completed successfully!"
}

# Run main function
main "$@"