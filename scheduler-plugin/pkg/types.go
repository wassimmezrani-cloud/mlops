package pkg

import (
	"fmt"
	"time"
)

// PodSpecification represents pod information for ML prediction
type PodSpecification struct {
	Name          string            `json:"name"`
	Namespace     string            `json:"namespace"`
	CPURequest    float64           `json:"cpu_request"`
	MemoryRequest float64           `json:"memory_request"`
	Priority      int32             `json:"priority"`
	Labels        map[string]string `json:"labels"`
	Annotations   map[string]string `json:"annotations"`
}

// NodeState represents current node state for ML prediction
type NodeState struct {
	Name              string            `json:"name"`
	CPUAllocatable    float64           `json:"cpu_allocatable"`
	MemoryAllocatable float64           `json:"memory_allocatable"`
	CPUUtilization    float64           `json:"cpu_utilization"`
	MemoryUtilization float64           `json:"memory_utilization"`
	PodCount          int               `json:"pod_count"`
	Labels            map[string]string `json:"labels"`
	Annotations       map[string]string `json:"annotations"`
}

// ClusterState represents current cluster state
type ClusterState struct {
	Timestamp              time.Time `json:"timestamp"`
	TotalNodes             int       `json:"total_nodes"`
	TotalPods              int       `json:"total_pods"`
	AverageNodeUtilization float64   `json:"average_node_utilization"`
	ClusterFragmentation   float64   `json:"cluster_fragmentation"`
}

// MLRecommendations combines all ML service responses
type MLRecommendations struct {
	XGBoostPrediction   XGBoostResponse   `json:"xgboost_prediction"`
	QLearningChoice     QLearningResponse `json:"qlearning_choice"`
	IsolationAssessment IsolationResponse `json:"isolation_assessment"`
	Timestamp           time.Time         `json:"timestamp"`
}

// XGBoost Service Types
type XGBoostRequest struct {
	Instances []XGBoostInstance `json:"instances"`
}

type XGBoostInstance struct {
	CPUUsageRate       float64 `json:"cpu_usage_rate"`
	MemoryUsageRate    float64 `json:"memory_usage_rate"`
	NetworkIO          float64 `json:"network_io"`
	DiskIO             float64 `json:"disk_io"`
	HourOfDay          float64 `json:"hour_of_day"`
	DayOfWeek          float64 `json:"day_of_week"`
	BusinessHours      float64 `json:"business_hours"`
	CPUPressure        float64 `json:"cpu_pressure"`
	MemoryPressure     float64 `json:"memory_pressure"`
	EfficiencyRatio    float64 `json:"efficiency_ratio"`
	LoadTrend1H        float64 `json:"load_trend_1h"`
	ResourceVolatility float64 `json:"resource_volatility"`
}

type XGBoostAPIResponse struct {
	Predictions []XGBoostPrediction `json:"predictions"`
}

type XGBoostPrediction struct {
	CPUUsage1H    float64 `json:"cpu_usage_1h"`
	MemoryUsage1H float64 `json:"memory_usage_1h"`
	Confidence    float64 `json:"confidence"`
}

type XGBoostResponse struct {
	PredictedCPU    float64   `json:"predicted_cpu"`
	PredictedMemory float64   `json:"predicted_memory"`
	Score           float64   `json:"score"`
	Confidence      float64   `json:"confidence"`
	Reasoning       string    `json:"reasoning"`
	Timestamp       time.Time `json:"timestamp"`
}

// Q-Learning Service Types
type QLearningRequest struct {
	Instances []QLearningInstance `json:"instances"`
}

type QLearningInstance struct {
	NodeCPUAvailable     float64 `json:"node_cpu_available"`
	NodeMemoryAvailable  float64 `json:"node_memory_available"`
	NodeLoadScore        float64 `json:"node_load_score"`
	PodCPURequest        float64 `json:"pod_cpu_request"`
	PodMemoryRequest     float64 `json:"pod_memory_request"`
	PodPriority          float64 `json:"pod_priority"`
	ClusterFragmentation float64 `json:"cluster_fragmentation"`
	NodeAffinityScore    float64 `json:"node_affinity_score"`
}

type QLearningAPIResponse struct {
	Predictions []QLearningPrediction `json:"predictions"`
}

type QLearningPrediction struct {
	Action     string  `json:"action"`
	QValue     float64 `json:"q_value"`
	Confidence float64 `json:"confidence"`
}

type QLearningResponse struct {
	Action     string    `json:"action"`
	Score      float64   `json:"score"`
	Confidence float64   `json:"confidence"`
	Reasoning  string    `json:"reasoning"`
	Timestamp  time.Time `json:"timestamp"`
}

// Isolation Forest Service Types
type IsolationRequest struct {
	Instances []IsolationInstance `json:"instances"`
}

type IsolationInstance struct {
	CPUUsageRate    float64 `json:"cpu_usage_rate"`
	MemoryUsageRate float64 `json:"memory_usage_rate"`
	NetworkIO       float64 `json:"network_io"`
	DiskIO          float64 `json:"disk_io"`
	CPUPressure     float64 `json:"cpu_pressure"`
	MemoryPressure  float64 `json:"memory_pressure"`
	LoadAverage     float64 `json:"load_average"`
	ContextSwitches float64 `json:"context_switches"`
	InterruptRate   float64 `json:"interrupt_rate"`
	SwapUsage       float64 `json:"swap_usage"`
	DiskUtilization float64 `json:"disk_utilization"`
	NetworkErrors   float64 `json:"network_errors"`
}

type IsolationAPIResponse struct {
	Predictions []IsolationPrediction `json:"predictions"`
}

type IsolationPrediction struct {
	AnomalyScore float64 `json:"anomaly_score"`
	IsAnomaly    bool    `json:"is_anomaly"`
	Confidence   float64 `json:"confidence"`
}

type IsolationResponse struct {
	Score      float64   `json:"score"`
	Confidence float64   `json:"confidence"`
	IsAnomaly  bool      `json:"is_anomaly"`
	Reasoning  string    `json:"reasoning"`
	Timestamp  time.Time `json:"timestamp"`
}

// Decision Fusion Types
type FusionResult struct {
	SelectedNode         string    `json:"selected_node"`
	FinalScore           float64   `json:"final_score"`
	Confidence           float64   `json:"confidence"`
	Strategy             string    `json:"strategy"`
	Reasoning            string    `json:"reasoning"`
	ContributingServices []string  `json:"contributing_services"`
	Timestamp            time.Time `json:"timestamp"`
}

// Metrics Types
type SchedulingDecision struct {
	PodName      string    `json:"pod_name"`
	PodNamespace string    `json:"pod_namespace"`
	SelectedNode string    `json:"selected_node"`
	Score        int64     `json:"score"`
	Strategy     string    `json:"strategy"`
	Latency      float64   `json:"latency_ms"`
	Timestamp    time.Time `json:"timestamp"`
}

type ServiceHealthStatus struct {
	ServiceName  string    `json:"service_name"`
	IsHealthy    bool      `json:"is_healthy"`
	LastCheck    time.Time `json:"last_check"`
	ResponseTime float64   `json:"response_time_ms"`
	ErrorCount   int       `json:"error_count"`
	SuccessRate  float64   `json:"success_rate"`
}

// Error Types
type MLServiceError struct {
	Service   string    `json:"service"`
	Message   string    `json:"message"`
	Code      string    `json:"code"`
	Timestamp time.Time `json:"timestamp"`
}

func (e *MLServiceError) Error() string {
	return fmt.Sprintf("ML service error (%s): %s", e.Service, e.Message)
}

// Business Logic Types
type BusinessMetrics struct {
	ResourceOptimizationRate   float64   `json:"resource_optimization_rate"`
	LoadBalancingEffectiveness float64   `json:"load_balancing_effectiveness"`
	AnomalyAvoidanceRate       float64   `json:"anomaly_avoidance_rate"`
	OverallSchedulingScore     float64   `json:"overall_scheduling_score"`
	Timestamp                  time.Time `json:"timestamp"`
}

type SchedulingContext struct {
	PodSpec        PodSpecification `json:"pod_spec"`
	AvailableNodes []NodeState      `json:"available_nodes"`
	ClusterState   ClusterState     `json:"cluster_state"`
	PolicyWeights  WeightConfig     `json:"policy_weights"`
}

// Configuration validation types
type ValidationResult struct {
	IsValid  bool     `json:"is_valid"`
	Errors   []string `json:"errors"`
	Warnings []string `json:"warnings"`
}

// Health check types
type HealthCheckResult struct {
	Service      string        `json:"service"`
	Status       string        `json:"status"` // "healthy", "unhealthy", "unknown"
	ResponseTime time.Duration `json:"response_time"`
	LastCheck    time.Time     `json:"last_check"`
	ErrorMessage string        `json:"error_message,omitempty"`
}

type OverallHealthStatus struct {
	Status          string              `json:"status"` // "healthy", "degraded", "unhealthy"
	ServiceStatuses []HealthCheckResult `json:"service_statuses"`
	HealthyServices int                 `json:"healthy_services"`
	TotalServices   int                 `json:"total_services"`
	LastUpdate      time.Time           `json:"last_update"`
}

// Performance monitoring types
type PerformanceMetrics struct {
	DecisionLatencyP50    float64 `json:"decision_latency_p50_ms"`
	DecisionLatencyP95    float64 `json:"decision_latency_p95_ms"`
	DecisionLatencyP99    float64 `json:"decision_latency_p99_ms"`
	ThroughputRPS         float64 `json:"throughput_rps"`
	ErrorRate             float64 `json:"error_rate"`
	FallbackUsageRate     float64 `json:"fallback_usage_rate"`
	MLServiceAvailability float64 `json:"ml_service_availability"`
}

// Cache types
type CacheKey string

type CacheEntry struct {
	Key       CacheKey    `json:"key"`
	Data      interface{} `json:"data"`
	CreatedAt time.Time   `json:"created_at"`
	ExpiresAt time.Time   `json:"expires_at"`
	HitCount  int         `json:"hit_count"`
}

// Tracing and debugging types
type DecisionTrace struct {
	TraceID       string                 `json:"trace_id"`
	SpanID        string                 `json:"span_id"`
	PodName       string                 `json:"pod_name"`
	PodNamespace  string                 `json:"pod_namespace"`
	StartTime     time.Time              `json:"start_time"`
	EndTime       time.Time              `json:"end_time"`
	Duration      time.Duration          `json:"duration"`
	Steps         []DecisionStep         `json:"steps"`
	FinalDecision SchedulingDecision     `json:"final_decision"`
	Metadata      map[string]interface{} `json:"metadata"`
}

type DecisionStep struct {
	StepName  string                 `json:"step_name"`
	StartTime time.Time              `json:"start_time"`
	EndTime   time.Time              `json:"end_time"`
	Duration  time.Duration          `json:"duration"`
	Input     map[string]interface{} `json:"input"`
	Output    map[string]interface{} `json:"output"`
	Error     string                 `json:"error,omitempty"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// Circuit breaker types
type CircuitBreakerState string

const (
	CircuitBreakerClosed   CircuitBreakerState = "closed"
	CircuitBreakerOpen     CircuitBreakerState = "open"
	CircuitBreakerHalfOpen CircuitBreakerState = "half-open"
)

type CircuitBreakerStatus struct {
	Service       string              `json:"service"`
	State         CircuitBreakerState `json:"state"`
	FailureCount  int                 `json:"failure_count"`
	SuccessCount  int                 `json:"success_count"`
	LastFailure   time.Time           `json:"last_failure,omitempty"`
	NextRetryTime time.Time           `json:"next_retry_time,omitempty"`
}
