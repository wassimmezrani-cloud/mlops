#!/bin/bash

# 📊 Historical Data Analysis Script for ML-Scheduler
# Validates Prometheus data availability and quality for ML training

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROMETHEUS_SERVICE="prometheus-k8s-external"
PROMETHEUS_NAMESPACE="monitoring" 
PROMETHEUS_PORT="9090"
MIN_DAYS_REQUIRED=7  # Minimum days for initial ML training
OPTIMAL_DAYS=30      # Optimal historical data for production ML

# Get Prometheus endpoint
get_prometheus_endpoint() {
    local external_ip=$(kubectl get svc ${PROMETHEUS_SERVICE} -n ${PROMETHEUS_NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
    
    if [[ -z "$external_ip" || "$external_ip" == "null" ]]; then
        # Fallback to NodePort
        local node_ip=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
        local node_port=$(kubectl get svc ${PROMETHEUS_SERVICE} -n ${PROMETHEUS_NAMESPACE} -o jsonpath='{.spec.ports[0].nodePort}')
        echo "${node_ip}:${node_port}"
    else
        echo "${external_ip}:${PROMETHEUS_PORT}"
    fi
}

# Query Prometheus for data availability
query_prometheus() {
    local endpoint=$1
    local query=$2
    local curl_response
    
    curl_response=$(curl -s -G "http://${endpoint}/api/v1/query" --data-urlencode "query=${query}" 2>/dev/null)
    echo "$curl_response"
}

# Query Prometheus for time range data
query_prometheus_range() {
    local endpoint=$1
    local query=$2
    local start=$3
    local end=$4
    local step="300s"  # 5 minutes
    local curl_response
    
    curl_response=$(curl -s -G "http://${endpoint}/api/v1/query_range" \
        --data-urlencode "query=${query}" \
        --data-urlencode "start=${start}" \
        --data-urlencode "end=${end}" \
        --data-urlencode "step=${step}" 2>/dev/null)
    echo "$curl_response"
}

# Analyze data availability and quality
analyze_historical_data() {
    local endpoint=$1
    
    echo -e "${BLUE}📊 Analyzing Prometheus historical data availability...${NC}\n"
    
    # Current time
    local current_time=$(date +%s)
    local days_30_ago=$((current_time - 30*24*3600))
    local days_7_ago=$((current_time - 7*24*3600))
    
    # Key metrics for ML-Scheduler training
    local metrics=(
        "node_cpu_seconds_total"
        "node_memory_MemAvailable_bytes"
        "node_memory_MemTotal_bytes"
        "node_load1"
        "node_load5"
        "node_load15"
        "node_filesystem_avail_bytes"
        "node_network_receive_bytes_total"
        "node_network_transmit_bytes_total"
        "kube_pod_info"
        "kube_pod_status_phase"
        "kube_node_status_condition"
    )
    
    echo -e "${YELLOW}🔍 Key Metrics Analysis:${NC}"
    echo "================================="
    
    local available_metrics=0
    local total_metrics=${#metrics[@]}
    
    for metric in "${metrics[@]}"; do
        # Test if metric exists and has recent data
        local query_result=$(query_prometheus "$endpoint" "$metric")
        local status=$(echo "$query_result" | jq -r '.status' 2>/dev/null)
        local data_points=$(echo "$query_result" | jq -r '.data.result | length' 2>/dev/null)
        
        if [[ "$status" == "success" && "$data_points" != "0" && "$data_points" != "null" ]]; then
            echo -e "  ✅ ${metric}: ${GREEN}Available${NC} (${data_points} series)"
            ((available_metrics++))
        else
            echo -e "  ❌ ${metric}: ${RED}Not Available${NC}"
        fi
    done
    
    echo -e "\n${BLUE}📈 Data Availability Summary:${NC}"
    echo "================================="
    echo -e "Available Metrics: ${GREEN}${available_metrics}${NC}/${total_metrics}"
    echo -e "Coverage: ${GREEN}$((available_metrics * 100 / total_metrics))%${NC}"
    
    # Check data retention and density
    echo -e "\n${BLUE}📅 Data Retention Analysis:${NC}"
    echo "================================="
    
    # Check 30-day data availability
    local retention_query="up{job=\"node-exporter\"}"
    local range_30d=$(query_prometheus_range "$endpoint" "$retention_query" "$days_30_ago" "$current_time")
    local range_7d=$(query_prometheus_range "$endpoint" "$retention_query" "$days_7_ago" "$current_time")
    
    local data_points_30d=$(echo "$range_30d" | jq -r '.data.result[0].values | length' 2>/dev/null || echo "0")
    local data_points_7d=$(echo "$range_7d" | jq -r '.data.result[0].values | length' 2>/dev/null || echo "0")
    
    echo -e "30-day data points: ${GREEN}${data_points_30d}${NC}"
    echo -e "7-day data points: ${GREEN}${data_points_7d}${NC}"
    
    # Calculate expected data points (every 5 minutes)
    local expected_30d=$((30 * 24 * 12))  # 30 days * 24 hours * 12 (5-min intervals per hour)
    local expected_7d=$((7 * 24 * 12))    # 7 days * 24 hours * 12
    
    local coverage_30d=0
    local coverage_7d=0
    
    if [[ "$data_points_30d" != "null" && "$data_points_30d" != "0" ]]; then
        coverage_30d=$((data_points_30d * 100 / expected_30d))
    fi
    
    if [[ "$data_points_7d" != "null" && "$data_points_7d" != "0" ]]; then
        coverage_7d=$((data_points_7d * 100 / expected_7d))
    fi
    
    echo -e "30-day coverage: ${GREEN}${coverage_30d}%${NC}"
    echo -e "7-day coverage: ${GREEN}${coverage_7d}%${NC}"
    
    # ML-Scheduler recommendations
    echo -e "\n${BLUE}🧠 ML-Scheduler Training Recommendations:${NC}"
    echo "=============================================="
    
    if [[ $data_points_30d -gt $((expected_30d * 70 / 100)) ]]; then
        echo -e "✅ ${GREEN}EXCELLENT${NC}: 30+ days data available - Ready for production ML training"
        echo -e "   🎯 Can proceed with all 3 algorithms (XGBoost + Q-Learning + Isolation Forest)"
        echo -e "   📊 Historical patterns analysis will be highly accurate"
        echo -e "   🔄 Seasonal and trend detection fully supported"
    elif [[ $data_points_7d -gt $((expected_7d * 70 / 100)) ]]; then
        echo -e "⚠️  ${YELLOW}GOOD${NC}: 7+ days data available - Can start initial ML training"
        echo -e "   🎯 Start with XGBoost model development"
        echo -e "   📊 Basic pattern recognition possible"
        echo -e "   ⏰ Recommend collecting more data for optimal performance"
    else
        echo -e "❌ ${RED}INSUFFICIENT${NC}: Less than 7 days of quality data"
        echo -e "   💡 Recommendation: Wait for more data collection"
        echo -e "   🔄 Continue monitoring data accumulation"
        echo -e "   📈 Expected minimum: ${MIN_DAYS_REQUIRED} days"
    fi
    
    # Feature engineering readiness
    echo -e "\n${BLUE}🔧 Feature Engineering Readiness:${NC}"
    echo "===================================="
    
    # Check for key node metrics needed for features
    local node_count=$(query_prometheus "$endpoint" "up{job=\"node-exporter\"}" | jq -r '.data.result | length' 2>/dev/null || echo "0")
    local pod_metrics=$(query_prometheus "$endpoint" "kube_pod_info" | jq -r '.data.result | length' 2>/dev/null || echo "0")
    
    echo -e "Monitored Nodes: ${GREEN}${node_count}${NC}"
    echo -e "Pod Metrics Available: ${GREEN}${pod_metrics}${NC}"
    
    if [[ $node_count -ge 3 && $pod_metrics -gt 0 ]]; then
        echo -e "✅ ${GREEN}Ready for feature engineering${NC}"
        echo -e "   📊 Multi-node patterns can be analyzed"
        echo -e "   🔄 Workload distribution metrics available"
    else
        echo -e "⚠️  ${YELLOW}Limited feature scope${NC}"
        echo -e "   💡 Some advanced features may not be available"
    fi
    
    # Next steps
    echo -e "\n${BLUE}🚀 Immediate Next Steps:${NC}"
    echo "========================="
    echo "1️⃣  Deploy Jupyter ML environment"
    echo "2️⃣  Create initial EDA notebook"
    echo "3️⃣  Start feature engineering pipeline"
    echo "4️⃣  Setup Feast feature store"
    
    if [[ $data_points_30d -gt $((expected_30d * 70 / 100)) ]]; then
        echo "5️⃣  Begin XGBoost model development"
        echo "6️⃣  Start Q-Learning environment setup"
    else
        echo "5️⃣  Continue data collection for optimal training"
        echo "6️⃣  Monitor data quality daily"
    fi
}

# Main execution
main() {
    echo -e "${GREEN}"
    cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║              📊 ML-SCHEDULER DATA ANALYZER                ║
║                                                            ║
║  Validating Prometheus Historical Data for ML Training    ║
╚════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}\n"
    
    # Check if required tools are available
    if ! command -v curl &> /dev/null; then
        echo -e "${RED}❌ curl is required but not installed${NC}"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        echo -e "${YELLOW}⚠️  Installing jq for JSON processing...${NC}"
        sudo apt-get update && sudo apt-get install -y jq
    fi
    
    # Get Prometheus endpoint
    local prometheus_endpoint
    prometheus_endpoint=$(get_prometheus_endpoint)
    
    if [[ -z "$prometheus_endpoint" ]]; then
        echo -e "${RED}❌ Cannot find Prometheus service endpoint${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}🎯 Prometheus Endpoint: ${GREEN}http://${prometheus_endpoint}${NC}\n"
    
    # Test connection
    if ! curl -s "http://${prometheus_endpoint}/-/healthy" > /dev/null; then
        echo -e "${RED}❌ Cannot connect to Prometheus at ${prometheus_endpoint}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Prometheus connection successful${NC}\n"
    
    # Analyze data
    analyze_historical_data "$prometheus_endpoint"
    
    echo -e "\n${GREEN}🎉 Analysis completed! Check recommendations above. 🚀${NC}"
}

# Run main function
main "$@"
