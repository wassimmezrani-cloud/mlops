package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"k8s.io/klog/v2"

	"ml-scheduler-plugin/pkg"
)

func main() {
	// Parse command line flags
	var configFile string
	flag.StringVar(&configFile, "config", "", "Path to scheduler configuration file")
	flag.Parse()

	klog.InfoS("Starting ML-Scheduler Plugin", "configFile", configFile)

	// Simple validation for scheduler plugin
	if configFile != "" {
		if _, err := os.Stat(configFile); os.IsNotExist(err) {
			klog.Fatalf("Configuration file does not exist: %s", configFile)
		}
		klog.InfoS("Using configuration file", "file", configFile)
	} else {
		klog.Info("No configuration file specified, will use default settings")
	}

	// Test basic plugin creation
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

	fmt.Printf("ML-Scheduler Plugin configuration created successfully\n")
	fmt.Printf("XGBoost Endpoint: %s\n", config.XGBoostEndpoint)
	fmt.Printf("Q-Learning Endpoint: %s\n", config.QLearningEndpoint)
	fmt.Printf("Isolation Forest Endpoint: %s\n", config.IsolationEndpoint)
	fmt.Printf("HTTP Timeout: %v\n", config.HTTPTimeout)
	fmt.Printf("Max Retries: %d\n", config.MaxRetries)
	fmt.Printf("Metrics Port: %d\n", config.MetricsPort)

	// Test metrics collector creation
	metrics := pkg.NewMetricsCollector(config.MetricsPort)
	if metrics != nil {
		fmt.Printf("Metrics collector created successfully on port %d\n", config.MetricsPort)
	}

	fmt.Println("ML-Scheduler Plugin test completed successfully!")
}
