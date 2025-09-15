package main

import (
	"context"
	"fmt"
	"os"

	"k8s.io/component-base/cli"
	"k8s.io/kubernetes/cmd/kube-scheduler/app"

	// Import ML Scheduler plugin
	_ "ml-scheduler-plugin/pkg"
)

func main() {
	ctx := context.Background()
	
	// ML-Scheduler Plugin version info
	fmt.Printf("Starting ML-Scheduler Plugin v1.0.0\n")
	fmt.Printf("Kubernetes Scheduler with AI-powered pod placement\n")
	fmt.Printf("Trio d'experts: XGBoost + Q-Learning + Isolation Forest\n\n")
	
	// Run scheduler with ML plugin
	command := app.NewSchedulerCommand()
	code := cli.Run(command)
	
	if code != 0 {
		fmt.Printf("ML-Scheduler exited with code: %d\n", code)
		os.Exit(code)
	}
}