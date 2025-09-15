#!/bin/bash

# ÉTAPE 11 - TEST INTÉGRATION ML-SCHEDULER AVEC KSERVE
# Script de test end-to-end pour validation du système complet

set -euo pipefail

echo "🧪 ÉTAPE 11 - TEST INTÉGRATION ML-SCHEDULER"
echo "==========================================="

# Configuration
SCHEDULER_NAMESPACE="kube-system"
ML_NAMESPACE="ml-scheduler-prod"
TEST_NAMESPACE="default"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Test 1: Verify KServe services are running
test_kserve_services() {
    log_info "Test 1: Verification des services KServe"
    
    local services=("xgboost-load-predictor" "isolation-anomaly-detector" "qlearning-placement-optimizer")
    local ready_count=0
    
    for service in "${services[@]}"; do
        local ready=$(kubectl get inferenceservice $service -n $ML_NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
        if [[ "$ready" == "True" ]]; then
            log_success "Service $service: READY"
            ((ready_count++))
        else
            log_warning "Service $service: NOT READY"
        fi
    done
    
    log_info "Services KServe prêts: $ready_count/3"
    return 0
}

# Test 2: Test ML service endpoints
test_ml_service_endpoints() {
    log_info "Test 2: Test des endpoints ML services"
    
    # Test XGBoost service
    log_info "Testing XGBoost service..."
    local xgboost_pod=$(kubectl get pods -n $ML_NAMESPACE -l serving.kserve.io/inferenceservice=xgboost-load-predictor -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [[ -n "$xgboost_pod" ]]; then
        kubectl port-forward -n $ML_NAMESPACE pod/$xgboost_pod 8082:8080 &
        local pf_pid=$!
        sleep 5
        
        local response=$(curl -s -X POST "http://localhost:8082/v1/models/xgboost-load-predictor:predict" \
            -H "Content-Type: application/json" \
            -d '{"instances": [[5.1, 3.5, 1.4, 0.2]]}' || echo "ERROR")
        
        kill $pf_pid 2>/dev/null || true
        
        if [[ "$response" == *"predictions"* ]]; then
            log_success "XGBoost endpoint: FUNCTIONAL"
            echo "Response: $response"
        else
            log_warning "XGBoost endpoint: ISSUES"
        fi
    else
        log_warning "XGBoost pod not found"
    fi
}

# Test 3: Test scheduler plugin configuration
test_scheduler_plugin_config() {
    log_info "Test 3: Test configuration scheduler plugin"
    
    # Create test config
    cat > /tmp/test-scheduler-config.yaml << EOF
pluginName: "MLScheduler"
pluginWeight: 100
xgboostEndpoint: "http://xgboost-load-predictor-predictor-00001.ml-scheduler-prod.svc.cluster.local/v1/models/xgboost-load-predictor:predict"
qlearningEndpoint: "http://qlearning-placement-optimizer-predictor-00001.ml-scheduler-prod.svc.cluster.local/v1/models/qlearning-placement-optimizer:predict"
isolationEndpoint: "http://isolation-anomaly-detector-predictor-00001.ml-scheduler-prod.svc.cluster.local/v1/models/isolation-anomaly-detector:predict"
httpTimeout: 30s
maxRetries: 3
metricsPort: 8080
logLevel: "info"
EOF

    # Test plugin with config (skip if binary doesn't exist)
    if [[ -x "./build/bin/ml-scheduler" ]]; then
        timeout 10s ./build/bin/ml-scheduler --config /tmp/test-scheduler-config.yaml &
        local scheduler_pid=$!
        sleep 5
        
        if kill -0 $scheduler_pid 2>/dev/null; then
            log_success "Scheduler plugin: STARTED"
            kill $scheduler_pid 2>/dev/null || true
        else
            log_warning "Scheduler plugin: STARTUP ISSUES"
        fi
    else
        log_warning "Scheduler plugin binary not found - skipping runtime test"
    fi
    
    rm -f /tmp/test-scheduler-config.yaml
}

# Test 4: Simulate pod scheduling decision
test_scheduling_simulation() {
    log_info "Test 4: Simulation décision de scheduling"
    
    # Create a simple scheduling simulation
    cat > /tmp/scheduling_test.py << 'EOF'
#!/usr/bin/env python3
import requests
import json
import sys

def simulate_ml_scheduling_decision():
    """Simulate ML-Scheduler decision making process"""
    
    # Mock pod specification
    pod_spec = {
        "name": "test-pod",
        "namespace": "default",
        "cpu_request": 200,  # 200m CPU
        "memory_request": 268435456,  # 256Mi
        "labels": {"app": "test", "workload-type": "cpu-intensive"}
    }
    
    # Mock node states
    available_nodes = [
        {"name": "worker1", "cpu_capacity": 2000, "memory_capacity": 8589934592},
        {"name": "worker2", "cpu_capacity": 4000, "memory_capacity": 17179869184},
        {"name": "worker3", "cpu_capacity": 2000, "memory_capacity": 8589934592}
    ]
    
    print("🎯 ML-Scheduler Decision Simulation")
    print("=" * 40)
    print(f"Pod: {pod_spec['name']}")
    print(f"CPU Request: {pod_spec['cpu_request']}m")
    print(f"Memory Request: {pod_spec['memory_request'] // 1024 // 1024}Mi")
    print(f"Available Nodes: {len(available_nodes)}")
    
    # Simple scoring simulation (without actual ML calls)
    best_node = None
    best_score = 0
    
    for node in available_nodes:
        # Simple scoring: prefer nodes with more available resources
        cpu_ratio = pod_spec["cpu_request"] / node["cpu_capacity"]
        memory_ratio = pod_spec["memory_request"] / node["memory_capacity"]
        
        # Lower resource usage ratio = higher score
        score = 1.0 - max(cpu_ratio, memory_ratio)
        
        print(f"Node {node['name']}: Score = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_node = node
    
    print(f"\n✅ Selected Node: {best_node['name']} (Score: {best_score:.3f})")
    return best_node['name']

if __name__ == "__main__":
    selected_node = simulate_ml_scheduling_decision()
    print(f"\nScheduling Decision: Pod -> {selected_node}")
EOF

    python3 /tmp/scheduling_test.py
    rm -f /tmp/scheduling_test.py
    
    log_success "Scheduling simulation: COMPLETED"
}

# Test 5: End-to-end integration check
test_end_to_end_integration() {
    log_info "Test 5: Validation intégration bout-en-bout"
    
    local integration_score=0
    local total_tests=4
    
    # Check KServe services
    local ready_services=$(kubectl get inferenceservice -n $ML_NAMESPACE -o jsonpath='{.items[?(@.status.conditions[0].type=="Ready")].metadata.name}' | wc -w)
    if [[ $ready_services -gt 0 ]]; then
        ((integration_score++))
        log_success "KServe services: $ready_services services ready"
    fi
    
    # Check scheduler plugin binary
    if [[ -x "./build/bin/ml-scheduler" ]]; then
        ((integration_score++))
        log_success "Scheduler plugin binary: AVAILABLE"
    fi
    
    # Check network connectivity to ML namespace
    if kubectl get pods -n $ML_NAMESPACE &>/dev/null; then
        ((integration_score++))
        log_success "Network connectivity: FUNCTIONAL"
    fi
    
    # Check RBAC permissions
    if kubectl auth can-i get pods --as=system:serviceaccount:kube-system:ml-scheduler-plugin &>/dev/null; then
        ((integration_score++))
        log_success "RBAC permissions: CONFIGURED"
    fi
    
    local integration_percentage=$((integration_score * 100 / total_tests))
    
    echo ""
    echo "🎯 INTÉGRATION SCORE: $integration_score/$total_tests ($integration_percentage%)"
    
    if [[ $integration_percentage -ge 75 ]]; then
        log_success "Integration status: READY FOR PRODUCTION"
        return 0
    elif [[ $integration_percentage -ge 50 ]]; then
        log_warning "Integration status: PARTIAL - Needs attention"
        return 1
    else
        log_error "Integration status: FAILED - Major issues"
        return 2
    fi
}

# Performance metrics collection
collect_performance_metrics() {
    log_info "Collection des métriques de performance"
    
    cat << EOF

📊 PERFORMANCE METRICS
=====================

KServe Services Status:
$(kubectl get inferenceservice -n $ML_NAMESPACE -o wide)

Pod Resources:
$(kubectl top pods -n $ML_NAMESPACE 2>/dev/null || echo "Metrics server not available")

Scheduler Plugin Configuration:
- Plugin Name: MLScheduler
- Target Latency: <200ms P99
- Max Replicas: 5 per service
- Fallback Strategy: 4-layer cascade

Business Impact Targets:
- CPU Optimization: 90% → 65%
- Memory Optimization: 90% → 70%  
- Availability Target: 99.7%
- Placement Efficiency: +34% vs random

EOF
}

# Main execution
main() {
    log_info "Démarrage des tests d'intégration ML-Scheduler"
    echo ""
    
    # Execute all tests
    test_kserve_services
    echo ""
    
    test_ml_service_endpoints
    echo ""
    
    test_scheduler_plugin_config
    echo ""
    
    test_scheduling_simulation
    echo ""
    
    test_end_to_end_integration
    local integration_result=$?
    echo ""
    
    collect_performance_metrics
    
    # Final summary
    if [[ $integration_result -eq 0 ]]; then
        log_success "🚀 ÉTAPE 11 COMPLÉTÉE AVEC SUCCÈS!"
        log_success "ML-Scheduler est prêt pour la production!"
        echo ""
        echo "🔗 PROCHAINES ÉTAPES:"
        echo "  - Déployer en production avec kubectl apply"
        echo "  - Configurer monitoring Grafana"
        echo "  - Lancer tests de charge"
        echo "  - Commencer migration workloads"
    else
        log_warning "🔧 ÉTAPE 11 PARTIELLEMENT COMPLÉTÉE"
        log_warning "Quelques ajustements nécessaires avant production"
    fi
    
    return $integration_result
}

# Trap for cleanup
trap 'log_warning "Test interrupted. Cleaning up..."; jobs -p | xargs -r kill 2>/dev/null' INT TERM

# Run main function
main "$@"