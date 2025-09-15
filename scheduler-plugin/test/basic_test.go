package test

import (
	"testing"
	"time"

	"ml-scheduler-plugin/pkg"
)

func TestMLSchedulerConfig(t *testing.T) {
	config := &pkg.MLSchedulerConfig{
		PluginName:        "MLScheduler",
		PluginWeight:      100,
		XGBoostEndpoint:   "http://xgboost-predictor.kubeflow.svc.cluster.local:80/v1/models/predictor:predict",
		QLearningEndpoint: "http://qlearning-optimizer.kubeflow.svc.cluster.local:80/v1/models/optimizer:predict",
		IsolationEndpoint: "http://isolation-detector.kubeflow.svc.cluster.local:80/v1/models/detector:predict",
		HTTPTimeout:       5000 * time.Millisecond,
		MaxRetries:        3,
		MetricsPort:       8080,
		LogLevel:          "info",
	}

	if config.PluginName != "MLScheduler" {
		t.Errorf("Expected PluginName to be 'MLScheduler', got '%s'", config.PluginName)
	}

	if config.PluginWeight != 100 {
		t.Errorf("Expected PluginWeight to be 100, got %d", config.PluginWeight)
	}

	if config.MaxRetries != 3 {
		t.Errorf("Expected MaxRetries to be 3, got %d", config.MaxRetries)
	}

	if config.MetricsPort != 8080 {
		t.Errorf("Expected MetricsPort to be 8080, got %d", config.MetricsPort)
	}

	if config.HTTPTimeout != 5000*time.Millisecond {
		t.Errorf("Expected HTTPTimeout to be 5s, got %v", config.HTTPTimeout)
	}
}

func TestMetricsCollector(t *testing.T) {
	collector := pkg.NewMetricsCollector(8080)
	if collector == nil {
		t.Error("Expected MetricsCollector to be created, got nil")
	}
}

func TestPluginConstants(t *testing.T) {
	if pkg.PluginName == "" {
		t.Error("Expected PluginName constant to be defined")
	}
}

func TestMLServiceEndpoints(t *testing.T) {
	config := &pkg.MLSchedulerConfig{
		XGBoostEndpoint:   "http://test-xgboost:80/predict",
		QLearningEndpoint: "http://test-qlearning:80/predict",
		IsolationEndpoint: "http://test-isolation:80/predict",
	}

	endpoints := []string{
		config.XGBoostEndpoint,
		config.QLearningEndpoint,
		config.IsolationEndpoint,
	}

	for _, endpoint := range endpoints {
		if endpoint == "" {
			t.Error("Expected ML service endpoint to be non-empty")
		}
	}
}
