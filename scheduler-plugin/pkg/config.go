package pkg

import (
	"time"

	"gopkg.in/yaml.v2"
)

// MLSchedulerConfig represents the configuration for the ML scheduler plugin
type MLSchedulerConfig struct {
	// Plugin configuration
	PluginName    string `yaml:"pluginName"`
	PluginWeight  int    `yaml:"pluginWeight"`
	EnableProfile bool   `yaml:"enableProfile"`

	// ML Service endpoints
	XGBoostEndpoint   string `yaml:"xgboostEndpoint"`
	QLearningEndpoint string `yaml:"qlearningEndpoint"`
	IsolationEndpoint string `yaml:"isolationEndpoint"`

	// HTTP client configuration
	HTTPTimeout       time.Duration `yaml:"httpTimeout"`
	MaxRetries        int           `yaml:"maxRetries"`
	RetryDelay        time.Duration `yaml:"retryDelay"`
	ConnectionTimeout time.Duration `yaml:"connectionTimeout"`
	MaxIdleConns      int           `yaml:"maxIdleConns"`
	MaxIdleConnsHost  int           `yaml:"maxIdleConnsHost"`

	// Circuit breaker configuration
	CircuitBreaker CircuitBreakerConfig `yaml:"circuitBreaker"`

	// Decision fusion configuration
	FusionWeights        WeightConfig         `yaml:"fusionWeights"`
	ConfidenceThresholds map[string]float64   `yaml:"confidenceThresholds"`
	FallbackPolicy       FallbackPolicyConfig `yaml:"fallbackPolicy"`

	// Performance configuration
	MetricsEnabled     bool          `yaml:"metricsEnabled"`
	MetricsPort        int           `yaml:"metricsPort"`
	HealthCheckPort    int           `yaml:"healthCheckPort"`
	DecisionTimeout    time.Duration `yaml:"decisionTimeout"`
	ConcurrentRequests int           `yaml:"concurrentRequests"`

	// Business logic configuration
	ResourceOptimizationWeight float64 `yaml:"resourceOptimizationWeight"`
	LoadBalancingWeight        float64 `yaml:"loadBalancingWeight"`
	AnomalyAvoidanceWeight     float64 `yaml:"anomalyAvoidanceWeight"`
	PerformancePenaltyFactor   float64 `yaml:"performancePenaltyFactor"`

	// Logging and debugging
	LogLevel        string `yaml:"logLevel"`
	EnableTracing   bool   `yaml:"enableTracing"`
	TracingEndpoint string `yaml:"tracingEndpoint"`
}

// CircuitBreakerConfig defines circuit breaker parameters
type CircuitBreakerConfig struct {
	MaxRequests      uint32        `yaml:"maxRequests"`
	Interval         time.Duration `yaml:"interval"`
	Timeout          time.Duration `yaml:"timeout"`
	FailureThreshold uint32        `yaml:"failureThreshold"`
	SuccessThreshold uint32        `yaml:"successThreshold"`
	OnStateChange    bool          `yaml:"onStateChange"`
}

// WeightConfig defines weights for decision fusion
type WeightConfig struct {
	XGBoostWeight    float64 `yaml:"xgboostWeight"`
	QLearningWeight  float64 `yaml:"qlearningWeight"`
	IsolationWeight  float64 `yaml:"isolationWeight"`
	NormalizeWeights bool    `yaml:"normalizeWeights"`
}

// FallbackPolicyConfig defines fallback behavior
type FallbackPolicyConfig struct {
	EnableFallback       bool          `yaml:"enableFallback"`
	FallbackTimeout      time.Duration `yaml:"fallbackTimeout"`
	MinServicesRequired  int           `yaml:"minServicesRequired"`
	UseKubernetesDefault bool          `yaml:"useKubernetesDefault"`
	CacheLastDecisions   bool          `yaml:"cacheLastDecisions"`
	CacheTTL             time.Duration `yaml:"cacheTTL"`
}

// DefaultConfig returns default configuration values
func DefaultConfig() *MLSchedulerConfig {
	return &MLSchedulerConfig{
		// Plugin configuration
		PluginName:    "MLScheduler",
		PluginWeight:  100,
		EnableProfile: false,

		// ML Service endpoints (production KServe services)
		XGBoostEndpoint:   "https://xgboost.ml-scheduler.hydatis.local/v1/models/xgboost-load-predictor:predict",
		QLearningEndpoint: "https://qlearning.ml-scheduler.hydatis.local/v1/models/qlearning-placement-optimizer:predict",
		IsolationEndpoint: "https://isolation.ml-scheduler.hydatis.local/v1/models/isolation-anomaly-detector:predict",

		// HTTP client configuration
		HTTPTimeout:       30 * time.Second,
		MaxRetries:        3,
		RetryDelay:        100 * time.Millisecond,
		ConnectionTimeout: 10 * time.Second,
		MaxIdleConns:      100,
		MaxIdleConnsHost:  10,

		// Circuit breaker configuration
		CircuitBreaker: CircuitBreakerConfig{
			MaxRequests:      10,
			Interval:         30 * time.Second,
			Timeout:          60 * time.Second,
			FailureThreshold: 5,
			SuccessThreshold: 3,
			OnStateChange:    true,
		},

		// Decision fusion configuration
		FusionWeights: WeightConfig{
			XGBoostWeight:    0.4,
			QLearningWeight:  0.4,
			IsolationWeight:  0.2,
			NormalizeWeights: true,
		},
		ConfidenceThresholds: map[string]float64{
			"xgboost":   0.75,
			"qlearning": 0.6,
			"isolation": 0.8,
		},
		FallbackPolicy: FallbackPolicyConfig{
			EnableFallback:       true,
			FallbackTimeout:      5 * time.Second,
			MinServicesRequired:  1,
			UseKubernetesDefault: true,
			CacheLastDecisions:   true,
			CacheTTL:             5 * time.Minute,
		},

		// Performance configuration
		MetricsEnabled:     true,
		MetricsPort:        8080,
		HealthCheckPort:    8081,
		DecisionTimeout:    200 * time.Millisecond,
		ConcurrentRequests: 100,

		// Business logic configuration
		ResourceOptimizationWeight: 0.3,
		LoadBalancingWeight:        0.3,
		AnomalyAvoidanceWeight:     0.4,
		PerformancePenaltyFactor:   0.1,

		// Logging and debugging
		LogLevel:        "info",
		EnableTracing:   true,
		TracingEndpoint: "http://jaeger-collector.monitoring.svc.cluster.local:14268/api/traces",
	}
}

// LoadConfig loads configuration from YAML data
func LoadConfig(data []byte) (*MLSchedulerConfig, error) {
	config := DefaultConfig()
	if err := yaml.Unmarshal(data, config); err != nil {
		return nil, err
	}
	return config, nil
}

// Validate validates the configuration
func (c *MLSchedulerConfig) Validate() error {
	if c.PluginWeight <= 0 {
		return NewValidationError("pluginWeight must be positive")
	}

	if c.HTTPTimeout <= 0 {
		return NewValidationError("httpTimeout must be positive")
	}

	if c.DecisionTimeout <= 0 {
		return NewValidationError("decisionTimeout must be positive")
	}

	if c.XGBoostEndpoint == "" {
		return NewValidationError("xgboostEndpoint is required")
	}

	if c.QLearningEndpoint == "" {
		return NewValidationError("qlearningEndpoint is required")
	}

	if c.IsolationEndpoint == "" {
		return NewValidationError("isolationEndpoint is required")
	}

	// Validate weights sum to reasonable value when normalized
	totalWeight := c.FusionWeights.XGBoostWeight + c.FusionWeights.QLearningWeight + c.FusionWeights.IsolationWeight
	if totalWeight <= 0 {
		return NewValidationError("fusion weights must sum to positive value")
	}

	return nil
}

// ValidationError represents a configuration validation error
type ValidationError struct {
	Message string
}

func (e *ValidationError) Error() string {
	return "configuration validation error: " + e.Message
}

func NewValidationError(message string) *ValidationError {
	return &ValidationError{Message: message}
}

// ToYAML converts configuration to YAML format
func (c *MLSchedulerConfig) ToYAML() ([]byte, error) {
	return yaml.Marshal(c)
}
