package utils

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"gopkg.in/yaml.v2"
)

// Config représente la configuration complète du ML-Scheduler
type Config struct {
	Scheduler       SchedulerConfig       `yaml:"scheduler"`
	Redis          RedisConfig           `yaml:"redis"`
	MLServices     MLServicesConfig      `yaml:"ml_services"`
	HistoricalData HistoricalDataConfig  `yaml:"historical_data"`
	Monitoring     MonitoringConfig      `yaml:"monitoring"`
	Kubeflow       KubeflowConfig        `yaml:"kubeflow"`
	Longhorn       LonghornConfig        `yaml:"longhorn"`
	Security       SecurityConfig        `yaml:"security"`
	Performance    PerformanceConfig     `yaml:"performance"`
	Development    DevelopmentConfig     `yaml:"development"`
	BusinessImpact BusinessImpactConfig  `yaml:"business_impact"`
}

// SchedulerConfig configuration du scheduler principal
type SchedulerConfig struct {
	Name      string `yaml:"name"`
	Version   string `yaml:"version"`
	Namespace string `yaml:"namespace"`
	
	Scoring    ScoringConfig    `yaml:"scoring"`
	Thresholds ThresholdsConfig `yaml:"thresholds"`
}

// ScoringConfig configuration du scoring ML
type ScoringConfig struct {
	Weights  WeightsConfig  `yaml:"weights"`
	Fallback FallbackConfig `yaml:"fallback"`
}

// WeightsConfig poids des algorithmes ML
type WeightsConfig struct {
	XGBoost     float64 `yaml:"xgboost"`
	QLearning   float64 `yaml:"qlearning"`
	Isolation   float64 `yaml:"isolation"`
}

// FallbackConfig configuration fallback
type FallbackConfig struct {
	Enabled       bool    `yaml:"enabled"`
	DefaultWeight float64 `yaml:"default_weight"`
}

// ThresholdsConfig seuils de décision
type ThresholdsConfig struct {
	MinScore      float64 `yaml:"min_score"`
	MaxLatencyMs  int     `yaml:"max_latency_ms"`
	ConfidenceMin float64 `yaml:"confidence_min"`
}

// RedisConfig configuration Redis cache
type RedisConfig struct {
	Host     string          `yaml:"host"`
	Port     int             `yaml:"port"`
	Password string          `yaml:"password"`
	DB       int             `yaml:"db"`
	TTL      TTLConfig       `yaml:"ttl"`
	Connection ConnectionConfig `yaml:"connection"`
}

// TTLConfig configuration TTL cache
type TTLConfig struct {
	NodeMetrics      int `yaml:"node_metrics"`
	Predictions      int `yaml:"predictions"`
	PlacementHistory int `yaml:"placement_history"`
}

// ConnectionConfig configuration connexion
type ConnectionConfig struct {
	PoolSize       int `yaml:"pool_size"`
	TimeoutSeconds int `yaml:"timeout_seconds"`
	RetryAttempts  int `yaml:"retry_attempts"`
}

// MLServicesConfig configuration services ML
type MLServicesConfig struct {
	Enabled        bool               `yaml:"enabled"`
	TimeoutSeconds int                `yaml:"timeout_seconds"`
	RetryAttempts  int                `yaml:"retry_attempts"`
	XGBoost        XGBoostConfig      `yaml:"xgboost"`
	QLearning      QLearningConfig    `yaml:"qlearning"`
	IsolationForest IsolationConfig   `yaml:"isolation_forest"`
}

// XGBoostConfig configuration XGBoost service
type XGBoostConfig struct {
	URL       string            `yaml:"url"`
	Endpoints EndpointsConfig   `yaml:"endpoints"`
	Model     ModelConfig       `yaml:"model"`
}

// QLearningConfig configuration Q-Learning service
type QLearningConfig struct {
	URL       string            `yaml:"url"`
	Endpoints EndpointsConfig   `yaml:"endpoints"`
	Agent     AgentConfig       `yaml:"agent"`
}

// IsolationConfig configuration Isolation Forest service
type IsolationConfig struct {
	URL       string            `yaml:"url"`
	Endpoints EndpointsConfig   `yaml:"endpoints"`
	Detector  DetectorConfig    `yaml:"detector"`
}

// EndpointsConfig endpoints des services
type EndpointsConfig struct {
	Predict  string `yaml:"predict,omitempty"`
	Optimize string `yaml:"optimize,omitempty"`
	Detect   string `yaml:"detect,omitempty"`
	Health   string `yaml:"health"`
	Metrics  string `yaml:"metrics"`
}

// ModelConfig configuration modèle
type ModelConfig struct {
	Version         string  `yaml:"version"`
	AccuracyTarget  float64 `yaml:"accuracy_target"`
	MemoryAccuracy  float64 `yaml:"memory_accuracy"`
}

// AgentConfig configuration agent Q-Learning
type AgentConfig struct {
	Version           string  `yaml:"version"`
	PerformanceTarget float64 `yaml:"performance_target"`
	LearningRate      float64 `yaml:"learning_rate"`
}

// DetectorConfig configuration détecteur anomalies
type DetectorConfig struct {
	Version           string  `yaml:"version"`
	PrecisionTarget   float64 `yaml:"precision_target"`
	FalsePositiveMax  float64 `yaml:"false_positive_max"`
}

// HistoricalDataConfig configuration données historiques
type HistoricalDataConfig struct {
	Storage    StorageConfig    `yaml:"storage"`
	Retention  RetentionConfig  `yaml:"retention"`
	Collection CollectionConfig `yaml:"collection"`
}

// StorageConfig configuration stockage
type StorageConfig struct {
	BasePath       string `yaml:"base_path"`
	ClusterMetrics string `yaml:"cluster_metrics"`
	NodeMetrics    string `yaml:"node_metrics"`
	PodMetrics     string `yaml:"pod_metrics"`
	Events         string `yaml:"events"`
}

// RetentionConfig configuration rétention
type RetentionConfig struct {
	Metrics     string `yaml:"metrics"`
	Events      string `yaml:"events"`
	Predictions string `yaml:"predictions"`
}

// CollectionConfig configuration collection
type CollectionConfig struct {
	MetricsInterval     string `yaml:"metrics_interval"`
	EventsInterval      string `yaml:"events_interval"`
	AggregationInterval string `yaml:"aggregation_interval"`
}

// MonitoringConfig configuration monitoring
type MonitoringConfig struct {
	PrometheusURL string        `yaml:"prometheus_url"`
	Metrics       MetricsConfig `yaml:"metrics"`
}

// MetricsConfig configuration métriques
type MetricsConfig struct {
	Enabled      bool     `yaml:"enabled"`
	Port         int      `yaml:"port"`
	Path         string   `yaml:"path"`
	Business     []string `yaml:"business"`
	ML           []string `yaml:"ml"`
	Operational  []string `yaml:"operational"`
}

// KubeflowConfig configuration Kubeflow
type KubeflowConfig struct {
	Namespace string        `yaml:"namespace"`
	MLflow    MLflowConfig  `yaml:"mlflow"`
	KServe    KServeConfig  `yaml:"kserve"`
	Feast     FeastConfig   `yaml:"feast"`
}

// MLflowConfig configuration MLflow
type MLflowConfig struct {
	TrackingURI    string `yaml:"tracking_uri"`
	ExperimentName string `yaml:"experiment_name"`
}

// KServeConfig configuration KServe
type KServeConfig struct {
	Namespace    string              `yaml:"namespace"`
	ModelServing ModelServingConfig  `yaml:"model_serving"`
}

// ModelServingConfig configuration serving
type ModelServingConfig struct {
	Timeout  int             `yaml:"timeout"`
	Replicas ReplicasConfig  `yaml:"replicas"`
}

// ReplicasConfig configuration replicas
type ReplicasConfig struct {
	Min int `yaml:"min"`
	Max int `yaml:"max"`
}

// FeastConfig configuration Feast
type FeastConfig struct {
	RegistryPath  string `yaml:"registry_path"`
	OnlineStore   string `yaml:"online_store"`
}

// LonghornConfig configuration Longhorn
type LonghornConfig struct {
	Namespace string        `yaml:"namespace"`
	Volumes   VolumesConfig `yaml:"volumes"`
	Backup    BackupConfig  `yaml:"backup"`
}

// VolumesConfig configuration volumes
type VolumesConfig struct {
	HistoricalData VolumeConfig `yaml:"historical_data"`
	MLModels       VolumeConfig `yaml:"ml_models"`
	CacheData      VolumeConfig `yaml:"cache_data"`
}

// VolumeConfig configuration volume
type VolumeConfig struct {
	Size         string `yaml:"size"`
	Replicas     int    `yaml:"replicas"`
	StorageClass string `yaml:"storage_class"`
}

// BackupConfig configuration backup
type BackupConfig struct {
	Enabled   bool   `yaml:"enabled"`
	Schedule  string `yaml:"schedule"`
	Retention string `yaml:"retention"`
}

// SecurityConfig configuration sécurité
type SecurityConfig struct {
	RBAC           RBACConfig           `yaml:"rbac"`
	NetworkPolicies NetworkPoliciesConfig `yaml:"network_policies"`
	PodSecurity    PodSecurityConfig    `yaml:"pod_security"`
}

// RBACConfig configuration RBAC
type RBACConfig struct {
	Enabled        bool   `yaml:"enabled"`
	ServiceAccount string `yaml:"service_account"`
}

// NetworkPoliciesConfig configuration network policies
type NetworkPoliciesConfig struct {
	Enabled bool                    `yaml:"enabled"`
	Ingress []map[string]string     `yaml:"ingress"`
	Egress  []map[string]string     `yaml:"egress"`
}

// PodSecurityConfig configuration pod security
type PodSecurityConfig struct {
	RunAsNonRoot     bool     `yaml:"run_as_non_root"`
	ReadOnlyRootFS   bool     `yaml:"read_only_root_fs"`
	DropCapabilities []string `yaml:"drop_capabilities"`
}

// PerformanceConfig configuration performance
type PerformanceConfig struct {
	Scheduling   SchedulingPerformanceConfig `yaml:"scheduling"`
	Cache        CachePerformanceConfig      `yaml:"cache"`
	MLInference  MLInferenceConfig           `yaml:"ml_inference"`
}

// SchedulingPerformanceConfig configuration performance scheduling
type SchedulingPerformanceConfig struct {
	MaxConcurrentDecisions int `yaml:"max_concurrent_decisions"`
	DecisionTimeoutMs      int `yaml:"decision_timeout_ms"`
	BatchSize              int `yaml:"batch_size"`
}

// CachePerformanceConfig configuration performance cache
type CachePerformanceConfig struct {
	PreloadEnabled      bool   `yaml:"preload_enabled"`
	PreloadInterval     string `yaml:"preload_interval"`
	CompressionEnabled  bool   `yaml:"compression_enabled"`
}

// MLInferenceConfig configuration ML inference
type MLInferenceConfig struct {
	BatchPredictions  bool `yaml:"batch_predictions"`
	BatchSize         int  `yaml:"batch_size"`
	AsyncProcessing   bool `yaml:"async_processing"`
}

// DevelopmentConfig configuration développement
type DevelopmentConfig struct {
	Debug   DebugConfig   `yaml:"debug"`
	Testing TestingConfig `yaml:"testing"`
}

// DebugConfig configuration debug
type DebugConfig struct {
	Enabled         bool   `yaml:"enabled"`
	LogLevel        string `yaml:"log_level"`
	DetailedMetrics bool   `yaml:"detailed_metrics"`
}

// TestingConfig configuration testing
type TestingConfig struct {
	MockMLServices   bool `yaml:"mock_ml_services"`
	SimulationMode   bool `yaml:"simulation_mode"`
	LoadTestEnabled  bool `yaml:"load_test_enabled"`
}

// BusinessImpactConfig configuration impact business
type BusinessImpactConfig struct {
	KPIs      KPIsConfig      `yaml:"kpis"`
	ROI       ROIConfig       `yaml:"roi"`
	Reporting ReportingConfig `yaml:"reporting"`
}

// KPIsConfig configuration KPIs
type KPIsConfig struct {
	AvailabilityTarget        float64 `yaml:"availability_target"`
	CPUUtilizationTarget      float64 `yaml:"cpu_utilization_target"`
	MemoryUtilizationTarget   float64 `yaml:"memory_utilization_target"`
	SimultaneousProjects      int     `yaml:"simultaneous_projects"`
}

// ROIConfig configuration ROI
type ROIConfig struct {
	TargetPercentage float64 `yaml:"target_percentage"`
	TargetMonths     int     `yaml:"target_months"`
	BaselineCosts    float64 `yaml:"baseline_costs"`
}

// ReportingConfig configuration reporting
type ReportingConfig struct {
	Enabled      bool   `yaml:"enabled"`
	Interval     string `yaml:"interval"`
	DashboardURL string `yaml:"dashboard_url"`
}

// LoadConfig charge la configuration depuis un fichier YAML
func LoadConfig(configFile string) (*Config, error) {
	if configFile == "" {
		configFile = "/etc/ml-scheduler/config.yaml"
	}

	data, err := ioutil.ReadFile(configFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file %s: %w", configFile, err)
	}

	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file %s: %w", configFile, err)
	}

	// Validation de la configuration
	if err := validateConfig(&config); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return &config, nil
}

// validateConfig valide la cohérence de la configuration
func validateConfig(config *Config) error {
	// Validation poids algorithmes ML
	totalWeight := config.Scheduler.Scoring.Weights.XGBoost + 
		config.Scheduler.Scoring.Weights.QLearning + 
		config.Scheduler.Scoring.Weights.Isolation
		
	if totalWeight < 0.99 || totalWeight > 1.01 {
		return fmt.Errorf("weights sum must equal 1.0, got %.2f", totalWeight)
	}

	// Validation seuils
	if config.Scheduler.Thresholds.MinScore < 0 || config.Scheduler.Thresholds.MinScore > 1 {
		return fmt.Errorf("min_score must be between 0 and 1")
	}

	if config.Scheduler.Thresholds.MaxLatencyMs <= 0 {
		return fmt.Errorf("max_latency_ms must be positive")
	}

	// Validation Redis
	if config.Redis.Port <= 0 || config.Redis.Port > 65535 {
		return fmt.Errorf("invalid Redis port: %d", config.Redis.Port)
	}

	// Validation URLs services ML
	if config.MLServices.Enabled {
		if config.MLServices.XGBoost.URL == "" {
			return fmt.Errorf("XGBoost URL is required when ML services enabled")
		}
		if config.MLServices.QLearning.URL == "" {
			return fmt.Errorf("Q-Learning URL is required when ML services enabled")
		}
		if config.MLServices.IsolationForest.URL == "" {
			return fmt.Errorf("Isolation Forest URL is required when ML services enabled")
		}
	}

	return nil
}

// CheckRedisConnection vérifie la connectivité Redis
func CheckRedisConnection(config RedisConfig) error {
	// Implémentation simple pour vérifier Redis
	// Dans la vraie implémentation, utiliser go-redis client
	url := fmt.Sprintf("http://%s:%d", config.Host, config.Port)
	
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("Redis connection failed: %w", err)
	}
	defer resp.Body.Close()
	
	return nil
}

// CheckMLServiceHealth vérifie la santé d'un service ML
func CheckMLServiceHealth(serviceURL string) error {
	if serviceURL == "" {
		return fmt.Errorf("service URL is empty")
	}

	url := serviceURL + "/health"
	client := &http.Client{Timeout: 2 * time.Second}
	
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status: %d", resp.StatusCode)
	}

	return nil
}

// CheckLonghornHealth vérifie la santé de Longhorn
func CheckLonghornHealth() error {
	// Implémentation simplifiée
	// Dans la vraie implémentation, vérifier API Longhorn
	return nil
}

// CheckPrometheusHealth vérifie la santé de Prometheus
func CheckPrometheusHealth(prometheusURL string) error {
	if prometheusURL == "" {
		return fmt.Errorf("Prometheus URL is empty")
	}

	url := prometheusURL + "/-/healthy"
	client := &http.Client{Timeout: 5 * time.Second}
	
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("Prometheus health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Prometheus health check failed with status: %d", resp.StatusCode)
	}

	return nil
}
