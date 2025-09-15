package pkg

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// MetricsCollector collects and exports scheduler plugin metrics
type MetricsCollector struct {
	port     int
	server   *http.Server
	registry *prometheus.Registry

	// Decision metrics
	decisionLatency  prometheus.HistogramVec
	decisionCounter  prometheus.CounterVec
	placementCounter prometheus.CounterVec

	// ML service metrics
	mlServiceCalls      prometheus.CounterVec
	mlServiceLatency    prometheus.HistogramVec
	mlServiceErrors     prometheus.CounterVec
	circuitBreakerState prometheus.GaugeVec

	// Fallback metrics
	fallbackUsage   prometheus.CounterVec
	fallbackLatency prometheus.HistogramVec

	// Business impact metrics
	resourceOptimization prometheus.GaugeVec
	loadBalancingScore   prometheus.GaugeVec
	anomalyAvoidanceRate prometheus.GaugeVec
	schedulingAccuracy   prometheus.GaugeVec

	// Health and availability metrics
	serviceHealth prometheus.GaugeVec
	pluginHealth  prometheus.GaugeVec

	// Performance tracking
	lastDecisions       []DecisionMetrics
	decisionsMutex      sync.RWMutex
	performanceWindow   time.Duration
	maxDecisionsHistory int
}

// DecisionMetrics tracks individual scheduling decisions for analysis
type DecisionMetrics struct {
	PodName              string
	PodNamespace         string
	SelectedNode         string
	Score                float64
	Confidence           float64
	Strategy             string
	Latency              time.Duration
	ContributingServices []string
	Timestamp            time.Time
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(port int) *MetricsCollector {
	registry := prometheus.NewRegistry()

	mc := &MetricsCollector{
		port:                port,
		registry:            registry,
		performanceWindow:   24 * time.Hour,
		maxDecisionsHistory: 10000,
		lastDecisions:       make([]DecisionMetrics, 0, 10000),
	}

	mc.initializeMetrics()
	mc.registerMetrics()

	return mc
}

// initializeMetrics initializes all Prometheus metrics
func (mc *MetricsCollector) initializeMetrics() {
	// Decision metrics
	mc.decisionLatency = *prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "ml_scheduler_decision_duration_seconds",
			Help:    "Time taken to make scheduling decisions",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 15), // 1ms to 16s
		},
		[]string{"strategy", "pod_namespace", "node_selected"},
	)

	mc.decisionCounter = *prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "ml_scheduler_decisions_total",
			Help: "Total number of scheduling decisions made",
		},
		[]string{"strategy", "outcome", "pod_namespace"},
	)

	mc.placementCounter = *prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "ml_scheduler_placements_total",
			Help: "Total number of pod placements by strategy",
		},
		[]string{"strategy", "node", "pod_namespace"},
	)

	// ML service metrics
	mc.mlServiceCalls = *prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "ml_scheduler_service_calls_total",
			Help: "Total calls to ML services",
		},
		[]string{"service", "status"},
	)

	mc.mlServiceLatency = *prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "ml_scheduler_service_duration_seconds",
			Help:    "Time taken for ML service calls",
			Buckets: prometheus.ExponentialBuckets(0.01, 2, 12), // 10ms to 40s
		},
		[]string{"service"},
	)

	mc.mlServiceErrors = *prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "ml_scheduler_service_errors_total",
			Help: "Total errors from ML services",
		},
		[]string{"service", "error_type"},
	)

	mc.circuitBreakerState = *prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "ml_scheduler_circuit_breaker_state",
			Help: "Circuit breaker state (0=closed, 1=open, 2=half-open)",
		},
		[]string{"service"},
	)

	// Fallback metrics
	mc.fallbackUsage = *prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "ml_scheduler_fallback_usage_total",
			Help: "Total usage of fallback strategies",
		},
		[]string{"fallback_type", "reason"},
	)

	mc.fallbackLatency = *prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "ml_scheduler_fallback_duration_seconds",
			Help:    "Time taken for fallback decisions",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 10), // 1ms to 1s
		},
		[]string{"fallback_type"},
	)

	// Business impact metrics
	mc.resourceOptimization = *prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "ml_scheduler_resource_optimization_rate",
			Help: "Resource optimization rate compared to baseline",
		},
		[]string{"resource_type"},
	)

	mc.loadBalancingScore = *prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "ml_scheduler_load_balancing_score",
			Help: "Load balancing effectiveness score",
		},
		[]string{"time_window"},
	)

	mc.anomalyAvoidanceRate = *prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "ml_scheduler_anomaly_avoidance_rate",
			Help: "Rate of avoiding nodes with anomalies",
		},
		[]string{"time_window"},
	)

	mc.schedulingAccuracy = *prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "ml_scheduler_accuracy_rate",
			Help: "Scheduling accuracy compared to expected outcomes",
		},
		[]string{"metric_type"},
	)

	// Health metrics
	mc.serviceHealth = *prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "ml_scheduler_service_health",
			Help: "Health status of ML services (1=healthy, 0=unhealthy)",
		},
		[]string{"service"},
	)

	mc.pluginHealth = *prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "ml_scheduler_plugin_health",
			Help: "Overall plugin health status",
		},
		[]string{"component"},
	)
}

// registerMetrics registers all metrics with the registry
func (mc *MetricsCollector) registerMetrics() {
	metrics := []prometheus.Collector{
		&mc.decisionLatency,
		&mc.decisionCounter,
		&mc.placementCounter,
		&mc.mlServiceCalls,
		&mc.mlServiceLatency,
		&mc.mlServiceErrors,
		&mc.circuitBreakerState,
		&mc.fallbackUsage,
		&mc.fallbackLatency,
		&mc.resourceOptimization,
		&mc.loadBalancingScore,
		&mc.anomalyAvoidanceRate,
		&mc.schedulingAccuracy,
		&mc.serviceHealth,
		&mc.pluginHealth,
	}

	for _, metric := range metrics {
		mc.registry.MustRegister(metric)
	}
}

// Start starts the metrics HTTP server
func (mc *MetricsCollector) Start() error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(mc.registry, promhttp.HandlerOpts{}))
	mux.HandleFunc("/health", mc.healthHandler)
	mux.HandleFunc("/performance", mc.performanceHandler)

	mc.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", mc.port),
		Handler: mux,
	}

	go func() {
		if err := mc.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			// Log error but don't fail the plugin
			fmt.Printf("Metrics server error: %v\n", err)
		}
	}()

	return nil
}

// Stop gracefully stops the metrics server
func (mc *MetricsCollector) Stop() error {
	if mc.server == nil {
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return mc.server.Shutdown(ctx)
}

// RecordDecisionLatency records the time taken for a scheduling decision
func (mc *MetricsCollector) RecordDecisionLatency(duration time.Duration) {
	mc.decisionLatency.WithLabelValues("unknown", "unknown", "unknown").Observe(duration.Seconds())
}

// RecordDecisionWithDetails records a detailed scheduling decision
func (mc *MetricsCollector) RecordDecisionWithDetails(
	strategy, outcome, namespace, node string,
	latency time.Duration,
	confidence float64,
) {
	mc.decisionLatency.WithLabelValues(strategy, namespace, node).Observe(latency.Seconds())
	mc.decisionCounter.WithLabelValues(strategy, outcome, namespace).Inc()

	if outcome == "success" {
		mc.placementCounter.WithLabelValues(strategy, node, namespace).Inc()
	}
}

// RecordPlacement records a successful pod placement
func (mc *MetricsCollector) RecordPlacement(strategy string) {
	mc.placementCounter.WithLabelValues(strategy, "unknown", "unknown").Inc()
}

// RecordMLServiceCall records calls to ML services
func (mc *MetricsCollector) RecordMLServiceCall(service, status string, duration time.Duration) {
	mc.mlServiceCalls.WithLabelValues(service, status).Inc()
	mc.mlServiceLatency.WithLabelValues(service).Observe(duration.Seconds())
}

// RecordMLServiceError records errors from ML services
func (mc *MetricsCollector) RecordMLServiceError(service, errorType string) {
	mc.mlServiceErrors.WithLabelValues(service, errorType).Inc()
}

// RecordCircuitBreakerState records circuit breaker state changes
func (mc *MetricsCollector) RecordCircuitBreakerState(service string, state string) {
	var stateValue float64
	switch state {
	case "closed":
		stateValue = 0
	case "open":
		stateValue = 1
	case "half-open":
		stateValue = 2
	}
	mc.circuitBreakerState.WithLabelValues(service).Set(stateValue)
}

// RecordFallbackUsage records usage of fallback strategies
func (mc *MetricsCollector) RecordFallbackUsage(fallbackType string) {
	mc.fallbackUsage.WithLabelValues(fallbackType, "ml_service_unavailable").Inc()
}

// RecordFallbackLatency records latency of fallback decisions
func (mc *MetricsCollector) RecordFallbackLatency(fallbackType string, duration time.Duration) {
	mc.fallbackLatency.WithLabelValues(fallbackType).Observe(duration.Seconds())
}

// UpdateServiceHealth updates health status of ML services
func (mc *MetricsCollector) UpdateServiceHealth(service string, healthy bool) {
	healthValue := 0.0
	if healthy {
		healthValue = 1.0
	}
	mc.serviceHealth.WithLabelValues(service).Set(healthValue)
}

// UpdatePluginHealth updates overall plugin health
func (mc *MetricsCollector) UpdatePluginHealth(component string, healthy bool) {
	healthValue := 0.0
	if healthy {
		healthValue = 1.0
	}
	mc.pluginHealth.WithLabelValues(component).Set(healthValue)
}

// UpdateBusinessMetrics updates business impact metrics
func (mc *MetricsCollector) UpdateBusinessMetrics(
	resourceOptimizationCPU, resourceOptimizationMemory float64,
	loadBalancingScore, anomalyAvoidanceRate, schedulingAccuracy float64,
) {
	mc.resourceOptimization.WithLabelValues("cpu").Set(resourceOptimizationCPU)
	mc.resourceOptimization.WithLabelValues("memory").Set(resourceOptimizationMemory)
	mc.loadBalancingScore.WithLabelValues("1h").Set(loadBalancingScore)
	mc.anomalyAvoidanceRate.WithLabelValues("1h").Set(anomalyAvoidanceRate)
	mc.schedulingAccuracy.WithLabelValues("placement_success").Set(schedulingAccuracy)
}

// RecordDecision records a complete scheduling decision for analysis
func (mc *MetricsCollector) RecordDecision(
	podName, podNamespace, selectedNode string,
	score, confidence float64,
	strategy string,
	latency time.Duration,
	contributingServices []string,
) {
	// Record in detailed metrics
	mc.RecordDecisionWithDetails(strategy, "success", podNamespace, selectedNode, latency, confidence)

	// Add to decision history for analysis
	mc.decisionsMutex.Lock()
	defer mc.decisionsMutex.Unlock()

	decision := DecisionMetrics{
		PodName:              podName,
		PodNamespace:         podNamespace,
		SelectedNode:         selectedNode,
		Score:                score,
		Confidence:           confidence,
		Strategy:             strategy,
		Latency:              latency,
		ContributingServices: contributingServices,
		Timestamp:            time.Now(),
	}

	mc.lastDecisions = append(mc.lastDecisions, decision)

	// Maintain history size
	if len(mc.lastDecisions) > mc.maxDecisionsHistory {
		mc.lastDecisions = mc.lastDecisions[1:]
	}

	// Clean old decisions outside the performance window
	cutoff := time.Now().Add(-mc.performanceWindow)
	for i, d := range mc.lastDecisions {
		if d.Timestamp.After(cutoff) {
			mc.lastDecisions = mc.lastDecisions[i:]
			break
		}
	}
}

// GetPerformanceMetrics returns performance metrics for the specified time window
func (mc *MetricsCollector) GetPerformanceMetrics(window time.Duration) map[string]interface{} {
	mc.decisionsMutex.RLock()
	defer mc.decisionsMutex.RUnlock()

	cutoff := time.Now().Add(-window)
	recentDecisions := []DecisionMetrics{}

	for _, decision := range mc.lastDecisions {
		if decision.Timestamp.After(cutoff) {
			recentDecisions = append(recentDecisions, decision)
		}
	}

	if len(recentDecisions) == 0 {
		return map[string]interface{}{
			"total_decisions": 0,
			"window_duration": window.String(),
		}
	}

	// Calculate performance statistics
	totalLatency := time.Duration(0)
	strategyCount := make(map[string]int)
	confidenceSum := 0.0
	nodeDistribution := make(map[string]int)

	for _, decision := range recentDecisions {
		totalLatency += decision.Latency
		strategyCount[decision.Strategy]++
		confidenceSum += decision.Confidence
		nodeDistribution[decision.SelectedNode]++
	}

	avgLatency := totalLatency / time.Duration(len(recentDecisions))
	avgConfidence := confidenceSum / float64(len(recentDecisions))

	// Calculate load balancing score (lower variance = better distribution)
	nodeVariance := mc.calculateNodeDistributionVariance(nodeDistribution)
	loadBalancingScore := math.Max(0, 1.0-nodeVariance)

	return map[string]interface{}{
		"total_decisions":       len(recentDecisions),
		"average_latency_ms":    float64(avgLatency.Nanoseconds()) / 1e6,
		"average_confidence":    avgConfidence,
		"strategy_distribution": strategyCount,
		"node_distribution":     nodeDistribution,
		"load_balancing_score":  loadBalancingScore,
		"window_duration":       window.String(),
	}
}

// healthHandler provides a health check endpoint
func (mc *MetricsCollector) healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	health := map[string]interface{}{
		"status":            "healthy",
		"timestamp":         time.Now().UTC(),
		"metrics_port":      mc.port,
		"decisions_tracked": len(mc.lastDecisions),
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(health)
}

// performanceHandler provides detailed performance metrics
func (mc *MetricsCollector) performanceHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	// Get metrics for different time windows
	performance := map[string]interface{}{
		"last_1h":  mc.GetPerformanceMetrics(1 * time.Hour),
		"last_24h": mc.GetPerformanceMetrics(24 * time.Hour),
		"last_7d":  mc.GetPerformanceMetrics(7 * 24 * time.Hour),
	}

	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(performance)
}

// calculateNodeDistributionVariance calculates variance in node selection distribution
func (mc *MetricsCollector) calculateNodeDistributionVariance(distribution map[string]int) float64 {
	if len(distribution) <= 1 {
		return 0.0
	}

	total := 0
	for _, count := range distribution {
		total += count
	}

	if total == 0 {
		return 0.0
	}

	expectedPerNode := float64(total) / float64(len(distribution))
	variance := 0.0

	for _, count := range distribution {
		diff := float64(count) - expectedPerNode
		variance += diff * diff
	}

	variance /= float64(len(distribution))
	return variance / (expectedPerNode * expectedPerNode) // Normalize by expected value
}

// GetCurrentStats returns current statistics snapshot
func (mc *MetricsCollector) GetCurrentStats() map[string]interface{} {
	mc.decisionsMutex.RLock()
	defer mc.decisionsMutex.RUnlock()

	stats := map[string]interface{}{
		"total_decisions_tracked": len(mc.lastDecisions),
		"metrics_port":            mc.port,
		"uptime_seconds":          time.Since(time.Now().Add(-24 * time.Hour)).Seconds(), // Placeholder
	}

	if len(mc.lastDecisions) > 0 {
		latest := mc.lastDecisions[len(mc.lastDecisions)-1]
		stats["latest_decision"] = map[string]interface{}{
			"timestamp":  latest.Timestamp,
			"strategy":   latest.Strategy,
			"confidence": latest.Confidence,
			"latency_ms": float64(latest.Latency.Nanoseconds()) / 1e6,
		}
	}

	return stats
}
