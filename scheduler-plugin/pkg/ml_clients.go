package pkg

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/sony/gobreaker"
)

// MLClientsManager manages HTTP clients for ML services with resilience patterns
type MLClientsManager struct {
	config            *MLSchedulerConfig
	httpClient        *http.Client
	circuitBreakers   map[string]*gobreaker.CircuitBreaker
	logger            logr.Logger
	healthStatus      map[string]bool
	healthMutex       sync.RWMutex
	lastDecisionCache *DecisionCache
}

// DecisionCache caches recent ML decisions for fallback scenarios
type DecisionCache struct {
	cache         map[string]*CachedDecision
	mutex         sync.RWMutex
	ttl           time.Duration
	cleanupTicker *time.Ticker
}

// CachedDecision represents a cached ML decision
type CachedDecision struct {
	Decision  *MLRecommendations
	Timestamp time.Time
	NodeName  string
	PodHash   string
}

// ServiceHealthChecker periodically checks ML service health
type ServiceHealthChecker struct {
	clients     *MLClientsManager
	interval    time.Duration
	stopChannel chan bool
	logger      logr.Logger
}

// NewMLClientsManager creates a new ML clients manager
func NewMLClientsManager(config *MLSchedulerConfig, logger logr.Logger) (*MLClientsManager, error) {
	// Configure HTTP client with timeouts and connection pooling
	httpClient := &http.Client{
		Timeout: config.HTTPTimeout,
		Transport: &http.Transport{
			MaxIdleConns:        config.MaxIdleConns,
			MaxIdleConnsPerHost: config.MaxIdleConnsHost,
			IdleConnTimeout:     30 * time.Second,
			DisableKeepAlives:   false,
		},
	}

	// Initialize circuit breakers for each service
	circuitBreakers := make(map[string]*gobreaker.CircuitBreaker)

	cbSettings := gobreaker.Settings{
		Name:        "xgboost",
		MaxRequests: config.CircuitBreaker.MaxRequests,
		Interval:    config.CircuitBreaker.Interval,
		Timeout:     config.CircuitBreaker.Timeout,
		ReadyToTrip: func(counts gobreaker.Counts) bool {
			return counts.ConsecutiveFailures >= config.CircuitBreaker.FailureThreshold
		},
	}
	circuitBreakers["xgboost"] = gobreaker.NewCircuitBreaker(cbSettings)

	cbSettings.Name = "qlearning"
	circuitBreakers["qlearning"] = gobreaker.NewCircuitBreaker(cbSettings)

	cbSettings.Name = "isolation"
	circuitBreakers["isolation"] = gobreaker.NewCircuitBreaker(cbSettings)

	// Initialize decision cache
	cache := &DecisionCache{
		cache: make(map[string]*CachedDecision),
		ttl:   config.FallbackPolicy.CacheTTL,
	}

	if config.FallbackPolicy.CacheLastDecisions {
		cache.startCleanup()
	}

	manager := &MLClientsManager{
		config:            config,
		httpClient:        httpClient,
		circuitBreakers:   circuitBreakers,
		logger:            logger,
		healthStatus:      make(map[string]bool),
		lastDecisionCache: cache,
	}

	// Initialize health status
	manager.healthStatus["xgboost"] = true
	manager.healthStatus["qlearning"] = true
	manager.healthStatus["isolation"] = true

	// Start health checker
	healthChecker := &ServiceHealthChecker{
		clients:  manager,
		interval: 30 * time.Second,
		logger:   logger,
	}
	go healthChecker.Start()

	return manager, nil
}

// GetRecommendationWithFallback gets ML recommendations with sophisticated fallback strategy
func (m *MLClientsManager) GetRecommendationWithFallback(
	ctx context.Context,
	podSpec PodSpecification,
	nodeState NodeState,
	clusterState ClusterState,
) (*MLRecommendations, string) {

	// Try to get trio recommendations
	if recommendations, err := m.getTrioRecommendations(ctx, podSpec, nodeState, clusterState); err == nil {
		m.logger.V(4).Info("Successfully got trio ML recommendations")
		return recommendations, "trio-ml"
	}

	// Fallback to duo (XGBoost + Q-Learning)
	if recommendations, err := m.getDuoRecommendations(ctx, podSpec, nodeState, clusterState); err == nil {
		m.logger.V(3).Info("Using duo ML recommendations fallback")
		return recommendations, "duo-ml"
	}

	// Fallback to XGBoost only
	if recommendations, err := m.getXGBoostOnlyRecommendations(ctx, podSpec, nodeState, clusterState); err == nil {
		m.logger.V(3).Info("Using XGBoost-only recommendations fallback")
		return recommendations, "xgboost-only"
	}

	// Fallback to cached decision if available
	if m.config.FallbackPolicy.CacheLastDecisions {
		if cached := m.getCachedDecision(podSpec, nodeState.Name); cached != nil {
			m.logger.V(2).Info("Using cached ML decision fallback")
			return cached, "cached-decision"
		}
	}

	// Final fallback to default scoring
	m.logger.V(1).Info("Using default scoring fallback - all ML services unavailable")
	return m.getDefaultRecommendations(podSpec, nodeState, clusterState), "default-fallback"
}

// getTrioRecommendations gets recommendations from all three ML services
func (m *MLClientsManager) getTrioRecommendations(
	ctx context.Context,
	podSpec PodSpecification,
	nodeState NodeState,
	clusterState ClusterState,
) (*MLRecommendations, error) {

	// Create channels for concurrent requests
	xgboostCh := make(chan XGBoostResponse, 1)
	qlearningCh := make(chan QLearningResponse, 1)
	isolationCh := make(chan IsolationResponse, 1)
	errorCh := make(chan error, 3)

	// Launch concurrent ML service requests
	go func() {
		response, err := m.callXGBoostService(ctx, podSpec, nodeState, clusterState)
		if err != nil {
			errorCh <- err
			return
		}
		xgboostCh <- response
	}()

	go func() {
		response, err := m.callQLearningService(ctx, podSpec, nodeState, clusterState)
		if err != nil {
			errorCh <- err
			return
		}
		qlearningCh <- response
	}()

	go func() {
		response, err := m.callIsolationService(ctx, podSpec, nodeState, clusterState)
		if err != nil {
			errorCh <- err
			return
		}
		isolationCh <- response
	}()

	// Collect responses with timeout
	var xgboost XGBoostResponse
	var qlearning QLearningResponse
	var isolation IsolationResponse
	responses := 0
	errors := 0

	timeout := time.After(m.config.DecisionTimeout)

	for responses+errors < 3 {
		select {
		case xgboost = <-xgboostCh:
			responses++
		case qlearning = <-qlearningCh:
			responses++
		case isolation = <-isolationCh:
			responses++
		case <-errorCh:
			errors++
		case <-timeout:
			return nil, fmt.Errorf("timeout waiting for ML service responses")
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	if responses < 3 {
		return nil, fmt.Errorf("failed to get all ML service responses: %d successful, %d errors", responses, errors)
	}

	recommendations := &MLRecommendations{
		XGBoostPrediction:   xgboost,
		QLearningChoice:     qlearning,
		IsolationAssessment: isolation,
		Timestamp:           time.Now(),
	}

	// Cache successful decision
	if m.config.FallbackPolicy.CacheLastDecisions {
		m.cacheDecision(podSpec, nodeState.Name, recommendations)
	}

	return recommendations, nil
}

// getDuoRecommendations gets recommendations from XGBoost and Q-Learning only
func (m *MLClientsManager) getDuoRecommendations(
	ctx context.Context,
	podSpec PodSpecification,
	nodeState NodeState,
	clusterState ClusterState,
) (*MLRecommendations, error) {

	xgboostCh := make(chan XGBoostResponse, 1)
	qlearningCh := make(chan QLearningResponse, 1)
	errorCh := make(chan error, 2)

	go func() {
		response, err := m.callXGBoostService(ctx, podSpec, nodeState, clusterState)
		if err != nil {
			errorCh <- err
			return
		}
		xgboostCh <- response
	}()

	go func() {
		response, err := m.callQLearningService(ctx, podSpec, nodeState, clusterState)
		if err != nil {
			errorCh <- err
			return
		}
		qlearningCh <- response
	}()

	var xgboost XGBoostResponse
	var qlearning QLearningResponse
	responses := 0
	errors := 0

	timeout := time.After(m.config.DecisionTimeout)

	for responses+errors < 2 {
		select {
		case xgboost = <-xgboostCh:
			responses++
		case qlearning = <-qlearningCh:
			responses++
		case <-errorCh:
			errors++
		case <-timeout:
			return nil, fmt.Errorf("timeout waiting for duo ML service responses")
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	if responses < 2 {
		return nil, fmt.Errorf("failed to get duo ML service responses")
	}

	// Create default isolation response for missing service
	isolation := IsolationResponse{
		Score:      0.5, // Neutral score
		Confidence: 0.0, // Zero confidence indicates missing data
		IsAnomaly:  false,
		Reasoning:  "Isolation service unavailable - assuming normal",
		Timestamp:  time.Now(),
	}

	return &MLRecommendations{
		XGBoostPrediction:   xgboost,
		QLearningChoice:     qlearning,
		IsolationAssessment: isolation,
		Timestamp:           time.Now(),
	}, nil
}

// getXGBoostOnlyRecommendations gets recommendations from XGBoost service only
func (m *MLClientsManager) getXGBoostOnlyRecommendations(
	ctx context.Context,
	podSpec PodSpecification,
	nodeState NodeState,
	clusterState ClusterState,
) (*MLRecommendations, error) {

	xgboost, err := m.callXGBoostService(ctx, podSpec, nodeState, clusterState)
	if err != nil {
		return nil, err
	}

	// Create default responses for missing services
	qlearning := QLearningResponse{
		Action:     "schedule", // Default to schedule
		Score:      0.5,        // Neutral score
		Confidence: 0.0,        // Zero confidence indicates missing data
		Reasoning:  "Q-Learning service unavailable - using default action",
		Timestamp:  time.Now(),
	}

	isolation := IsolationResponse{
		Score:      0.5,
		Confidence: 0.0,
		IsAnomaly:  false,
		Reasoning:  "Isolation service unavailable - assuming normal",
		Timestamp:  time.Now(),
	}

	return &MLRecommendations{
		XGBoostPrediction:   xgboost,
		QLearningChoice:     qlearning,
		IsolationAssessment: isolation,
		Timestamp:           time.Now(),
	}, nil
}

// callXGBoostService calls the XGBoost prediction service
func (m *MLClientsManager) callXGBoostService(
	ctx context.Context,
	podSpec PodSpecification,
	nodeState NodeState,
	clusterState ClusterState,
) (XGBoostResponse, error) {

	// Prepare request payload
	request := XGBoostRequest{
		Instances: []XGBoostInstance{{
			CPUUsageRate:       nodeState.CPUUtilization,
			MemoryUsageRate:    nodeState.MemoryUtilization,
			NetworkIO:          1024000, // Default network IO
			DiskIO:             512000,  // Default disk IO
			HourOfDay:          float64(time.Now().Hour()),
			DayOfWeek:          float64(time.Now().Weekday()),
			BusinessHours:      getBusinessHoursIndicator(),
			CPUPressure:        nodeState.CPUUtilization * 0.3,
			MemoryPressure:     nodeState.MemoryUtilization * 0.4,
			EfficiencyRatio:    calculateEfficiencyRatio(nodeState),
			LoadTrend1H:        0.05, // Default load trend
			ResourceVolatility: 0.08, // Default volatility
		}},
	}

	// Call service through circuit breaker
	result, err := m.circuitBreakers["xgboost"].Execute(func() (interface{}, error) {
		return m.makeHTTPRequest(ctx, m.config.XGBoostEndpoint, request)
	})

	if err != nil {
		m.updateHealthStatus("xgboost", false)
		return XGBoostResponse{}, fmt.Errorf("XGBoost service call failed: %w", err)
	}

	m.updateHealthStatus("xgboost", true)

	// Parse response
	responseData, ok := result.([]byte)
	if !ok {
		return XGBoostResponse{}, fmt.Errorf("invalid XGBoost response type")
	}

	var xgboostResp XGBoostAPIResponse
	if err := json.Unmarshal(responseData, &xgboostResp); err != nil {
		return XGBoostResponse{}, fmt.Errorf("failed to parse XGBoost response: %w", err)
	}

	// Convert to internal format
	return XGBoostResponse{
		PredictedCPU:    xgboostResp.Predictions[0].CPUUsage1H,
		PredictedMemory: xgboostResp.Predictions[0].MemoryUsage1H,
		Score:           calculateXGBoostScore(xgboostResp.Predictions[0]),
		Confidence:      xgboostResp.Predictions[0].Confidence,
		Reasoning:       "XGBoost load prediction based on historical patterns",
		Timestamp:       time.Now(),
	}, nil
}

// callQLearningService calls the Q-Learning optimization service
func (m *MLClientsManager) callQLearningService(
	ctx context.Context,
	podSpec PodSpecification,
	nodeState NodeState,
	clusterState ClusterState,
) (QLearningResponse, error) {

	request := QLearningRequest{
		Instances: []QLearningInstance{{
			NodeCPUAvailable:     1.0 - nodeState.CPUUtilization,
			NodeMemoryAvailable:  1.0 - nodeState.MemoryUtilization,
			NodeLoadScore:        nodeState.CPUUtilization*0.6 + nodeState.MemoryUtilization*0.4,
			PodCPURequest:        podSpec.CPURequest,
			PodMemoryRequest:     podSpec.MemoryRequest,
			PodPriority:          float64(podSpec.Priority),
			ClusterFragmentation: clusterState.ClusterFragmentation,
			NodeAffinityScore:    calculateAffinityScore(podSpec, nodeState),
		}},
	}

	result, err := m.circuitBreakers["qlearning"].Execute(func() (interface{}, error) {
		return m.makeHTTPRequest(ctx, m.config.QLearningEndpoint, request)
	})

	if err != nil {
		m.updateHealthStatus("qlearning", false)
		return QLearningResponse{}, fmt.Errorf("Q-Learning service call failed: %w", err)
	}

	m.updateHealthStatus("qlearning", true)

	responseData, ok := result.([]byte)
	if !ok {
		return QLearningResponse{}, fmt.Errorf("invalid Q-Learning response type")
	}

	var qlearningResp QLearningAPIResponse
	if err := json.Unmarshal(responseData, &qlearningResp); err != nil {
		return QLearningResponse{}, fmt.Errorf("failed to parse Q-Learning response: %w", err)
	}

	return QLearningResponse{
		Action:     qlearningResp.Predictions[0].Action,
		Score:      qlearningResp.Predictions[0].QValue,
		Confidence: qlearningResp.Predictions[0].Confidence,
		Reasoning:  "Q-Learning optimization based on reinforcement learning",
		Timestamp:  time.Now(),
	}, nil
}

// callIsolationService calls the Isolation Forest anomaly detection service
func (m *MLClientsManager) callIsolationService(
	ctx context.Context,
	podSpec PodSpecification,
	nodeState NodeState,
	clusterState ClusterState,
) (IsolationResponse, error) {

	request := IsolationRequest{
		Instances: []IsolationInstance{{
			CPUUsageRate:    nodeState.CPUUtilization,
			MemoryUsageRate: nodeState.MemoryUtilization,
			NetworkIO:       2048000, // Default network IO
			DiskIO:          1024000, // Default disk IO
			CPUPressure:     nodeState.CPUUtilization * 0.15,
			MemoryPressure:  nodeState.MemoryUtilization * 0.25,
			LoadAverage:     2.5,   // Default load average
			ContextSwitches: 15000, // Default context switches
			InterruptRate:   8000,  // Default interrupt rate
			SwapUsage:       0.05,  // Default swap usage
			DiskUtilization: 0.75,  // Default disk utilization
			NetworkErrors:   10,    // Default network errors
		}},
	}

	result, err := m.circuitBreakers["isolation"].Execute(func() (interface{}, error) {
		return m.makeHTTPRequest(ctx, m.config.IsolationEndpoint, request)
	})

	if err != nil {
		m.updateHealthStatus("isolation", false)
		return IsolationResponse{}, fmt.Errorf("Isolation service call failed: %w", err)
	}

	m.updateHealthStatus("isolation", true)

	responseData, ok := result.([]byte)
	if !ok {
		return IsolationResponse{}, fmt.Errorf("invalid Isolation response type")
	}

	var isolationResp IsolationAPIResponse
	if err := json.Unmarshal(responseData, &isolationResp); err != nil {
		return IsolationResponse{}, fmt.Errorf("failed to parse Isolation response: %w", err)
	}

	return IsolationResponse{
		Score:      isolationResp.Predictions[0].AnomalyScore,
		Confidence: isolationResp.Predictions[0].Confidence,
		IsAnomaly:  isolationResp.Predictions[0].IsAnomaly,
		Reasoning:  "Isolation Forest anomaly detection based on statistical patterns",
		Timestamp:  time.Now(),
	}, nil
}

// makeHTTPRequest makes HTTP request with retry logic
func (m *MLClientsManager) makeHTTPRequest(ctx context.Context, endpoint string, payload interface{}) ([]byte, error) {
	requestBody, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt <= m.config.MaxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-time.After(m.config.RetryDelay * time.Duration(attempt)):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		req, err := http.NewRequestWithContext(ctx, "POST", endpoint, bytes.NewBuffer(requestBody))
		if err != nil {
			lastErr = err
			continue
		}

		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("User-Agent", "ML-Scheduler-Plugin/1.0")

		resp, err := m.httpClient.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		responseBody, err := io.ReadAll(resp.Body)
		resp.Body.Close()

		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return responseBody, nil
		}

		lastErr = fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(responseBody))
	}

	return nil, fmt.Errorf("failed after %d attempts: %w", m.config.MaxRetries+1, lastErr)
}

// Helper functions

func getBusinessHoursIndicator() float64 {
	hour := time.Now().Hour()
	if hour >= 9 && hour <= 17 {
		return 1.0
	}
	return 0.0
}

func calculateEfficiencyRatio(nodeState NodeState) float64 {
	utilizationSum := nodeState.CPUUtilization + nodeState.MemoryUtilization
	if utilizationSum > 0 {
		return 1.0 - (utilizationSum / 2.0)
	}
	return 1.0
}

func calculateXGBoostScore(prediction XGBoostPrediction) float64 {
	// Higher predicted load = lower score (avoid overloaded nodes)
	return 1.0 - ((prediction.CPUUsage1H + prediction.MemoryUsage1H) / 2.0)
}

func calculateAffinityScore(podSpec PodSpecification, nodeState NodeState) float64 {
	// Simple affinity calculation based on labels
	score := 0.5 // Default neutral score

	// This would implement more sophisticated affinity logic
	// based on pod and node labels/annotations

	return score
}

func (m *MLClientsManager) updateHealthStatus(service string, healthy bool) {
	m.healthMutex.Lock()
	defer m.healthMutex.Unlock()
	m.healthStatus[service] = healthy
}

// Additional methods for decision caching and health checking
func (m *MLClientsManager) cacheDecision(podSpec PodSpecification, nodeName string, decision *MLRecommendations) {
	// Implementation of decision caching
}

func (m *MLClientsManager) getCachedDecision(podSpec PodSpecification, nodeName string) *MLRecommendations {
	// Implementation of cached decision retrieval
	return nil
}

func (m *MLClientsManager) getDefaultRecommendations(podSpec PodSpecification, nodeState NodeState, clusterState ClusterState) *MLRecommendations {
	// Default scoring based on resource availability
	score := 1.0 - ((nodeState.CPUUtilization + nodeState.MemoryUtilization) / 2.0)

	return &MLRecommendations{
		XGBoostPrediction: XGBoostResponse{
			Score:      score,
			Confidence: 0.0,
			Reasoning:  "Default resource-based scoring",
			Timestamp:  time.Now(),
		},
		QLearningChoice: QLearningResponse{
			Action:     "schedule",
			Score:      score,
			Confidence: 0.0,
			Reasoning:  "Default scheduling decision",
			Timestamp:  time.Now(),
		},
		IsolationAssessment: IsolationResponse{
			Score:      0.5,
			Confidence: 0.0,
			IsAnomaly:  false,
			Reasoning:  "Default normal assessment",
			Timestamp:  time.Now(),
		},
		Timestamp: time.Now(),
	}
}

// Health checker implementation
func (hc *ServiceHealthChecker) Start() {
	hc.stopChannel = make(chan bool, 1)
	ticker := time.NewTicker(hc.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			hc.checkServicesHealth()
		case <-hc.stopChannel:
			return
		}
	}
}

func (hc *ServiceHealthChecker) checkServicesHealth() {
	// Implementation of periodic health checks
}

func (c *DecisionCache) startCleanup() {
	c.cleanupTicker = time.NewTicker(time.Minute)
	go func() {
		for range c.cleanupTicker.C {
			c.cleanup()
		}
	}()
}

func (c *DecisionCache) cleanup() {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	cutoff := time.Now().Add(-c.ttl)
	for key, decision := range c.cache {
		if decision.Timestamp.Before(cutoff) {
			delete(c.cache, key)
		}
	}
}
