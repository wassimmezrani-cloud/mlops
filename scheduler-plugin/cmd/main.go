package main

import (
	"flag"
	"fmt"
	"os"

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

	// Test plugin instantiation
	config := &pkg.MLSchedulerConfig{
		XGBoostURL:       "http://xgboost-predictor.kubeflow.svc.cluster.local:80/v1/models/predictor:predict",
		QLearningURL:     "http://qlearning-optimizer.kubeflow.svc.cluster.local:80/v1/models/optimizer:predict",
		IsolationURL:     "http://isolation-detector.kubeflow.svc.cluster.local:80/v1/models/detector:predict",
		Timeout:          5000,
		MaxRetries:       3,
		MetricsPort:      8080,
		HealthPort:       8081,
		EnablePrometheus: true,
		LogLevel:         "info",
	}

	// Validate plugin can be created
	plugin, err := pkg.New(config)
	if err != nil {
		klog.ErrorS(err, "Failed to create ML scheduler plugin")
		os.Exit(1)
	}

	klog.InfoS("ML-Scheduler Plugin created successfully", "pluginName", pkg.PluginName)

	// For testing, just print success
	fmt.Printf("ML-Scheduler Plugin %s initialized successfully\n", pkg.PluginName)
	fmt.Printf("Plugin configuration: %+v\n", plugin)
}
