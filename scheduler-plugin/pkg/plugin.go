package pkg

import (
	"context"
	"fmt"
	"net/http"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// MLSchedulerPlugin implements the ML-powered scheduler plugin
type MLSchedulerPlugin struct {
	handle       framework.Handle
	mlClients    *MLClientsManager
	fusionEngine *DecisionFusionEngine
	metrics      *MetricsCollector
}

// Name returns the name of the plugin
func (pl *MLSchedulerPlugin) Name() string {
	return "ml-scheduler-plugin"
}

// New initializes a new ML Scheduler plugin
func New(obj runtime.Object, h framework.Handle) (framework.Plugin, error) {
	klog.InfoS("Initializing ML-Scheduler Plugin", "version", "v1.0.0")
	
	// Initialize ML clients manager
	mlClients := &MLClientsManager{
		xgboostURL:   "http://xgboost-predictor.ml-scheduler.svc.cluster.local:8080/v1/models/xgboost-predictor:predict",
		qlearningURL: "http://qlearning-optimizer.ml-scheduler.svc.cluster.local:8080/v1/models/qlearning-optimizer:predict", 
		isolationURL: "http://isolation-detector.ml-scheduler.svc.cluster.local:8080/v1/models/isolation-detector:predict",
		httpClient: &http.Client{
			Timeout: 200 * time.Millisecond,
		},
	}
	
	// Initialize decision fusion engine
	fusionEngine := &DecisionFusionEngine{
		strategy: WeightedVotingFusion,
		weights: map[string]float64{
			"xgboost":   0.35,
			"qlearning": 0.35,
			"isolation": 0.30,
		},
	}
	
	// Initialize metrics collector
	metrics := NewMetricsCollector()
	
	plugin := &MLSchedulerPlugin{
		handle:       h,
		mlClients:    mlClients,
		fusionEngine: fusionEngine,
		metrics:      metrics,
	}
	
	klog.InfoS("ML-Scheduler Plugin initialized successfully",
		"xgboost_url", mlClients.xgboostURL,
		"qlearning_url", mlClients.qlearningURL,
		"isolation_url", mlClients.isolationURL)
	
	return plugin, nil
}

// Score performs ML-powered node scoring
func (pl *MLSchedulerPlugin) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	start := time.Now()
	defer func() {
		pl.metrics.recordDecisionLatency(time.Since(start))
	}()
	
	// Get ML recommendations for this node
	recommendation, err := pl.mlClients.getNodeRecommendation(ctx, pod, nodeName)
	if err != nil {
		klog.ErrorS(err, "Failed to get ML recommendation", "pod", pod.Name, "node", nodeName)
		// Fallback to default scoring
		pl.metrics.recordFallback("ml_service_error")
		return 50, framework.NewStatus(framework.Success, "ML fallback - default score")
	}
	
	// Fuse decisions from trio of experts
	score, confidence, reasoning := pl.fusionEngine.fuseDecisions(recommendation, nodeName)
	
	// Record metrics
	pl.metrics.recordDecision("trio_ml", confidence)
	pl.metrics.recordPlacement(nodeName, score)
	
	klog.V(4).InfoS("ML-Scheduler decision",
		"pod", pod.Name,
		"node", nodeName, 
		"score", score,
		"confidence", confidence,
		"reasoning", reasoning)
	
	// Convert to Kubernetes scheduler score (0-100)
	schedulerScore := int64(score * 100)
	if schedulerScore < 0 {
		schedulerScore = 0
	} else if schedulerScore > 100 {
		schedulerScore = 100
	}
	
	return schedulerScore, framework.NewStatus(framework.Success, reasoning)
}

// Filter performs basic filtering with anomaly exclusion
func (pl *MLSchedulerPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	nodeName := nodeInfo.Node().Name
	
	// Quick anomaly check via Isolation Forest
	anomalyScore, err := pl.mlClients.checkNodeAnomaly(ctx, nodeName)
	if err != nil {
		klog.V(4).InfoS("Anomaly check failed, allowing node", "node", nodeName, "error", err)
		return framework.NewStatus(framework.Success, "Anomaly check failed - allowing")
	}
	
	// If node has high anomaly score, filter it out
	if anomalyScore > 0.8 {
		pl.metrics.recordAnomalyAvoidance(nodeName)
		return framework.NewStatus(framework.Unschedulable, 
			fmt.Sprintf("Node %s has high anomaly score: %.2f", nodeName, anomalyScore))
	}
	
	return framework.NewStatus(framework.Success, "Node passed anomaly filter")
}

// PreFilter initializes pod state for ML processing
func (pl *MLSchedulerPlugin) PreFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod) (*framework.PreFilterResult, *framework.Status) {
	// Store pod metadata in cycle state for later use
	podState := &PodMLState{
		PodName:      pod.Name,
		Namespace:    pod.Namespace,
		CPURequest:   getPodCPURequest(pod),
		MemoryRequest: getPodMemoryRequest(pod),
		Priority:     getPodPriority(pod),
	}
	
	state.Write(podStateKey, podState)
	
	klog.V(4).InfoS("ML-Scheduler PreFilter", 
		"pod", pod.Name,
		"cpu_request", podState.CPURequest,
		"memory_request", podState.MemoryRequest,
		"priority", podState.Priority)
	
	return nil, framework.NewStatus(framework.Success, "Pod state initialized")
}

// Helper functions
func getPodCPURequest(pod *v1.Pod) int64 {
	var totalCPU int64
	for _, container := range pod.Spec.Containers {
		if cpu := container.Resources.Requests.Cpu(); cpu != nil {
			totalCPU += cpu.MilliValue()
		}
	}
	return totalCPU
}

func getPodMemoryRequest(pod *v1.Pod) int64 {
	var totalMemory int64
	for _, container := range pod.Spec.Containers {
		if memory := container.Resources.Requests.Memory(); memory != nil {
			totalMemory += memory.Value()
		}
	}
	return totalMemory
}

func getPodPriority(pod *v1.Pod) int32 {
	if pod.Spec.Priority != nil {
		return *pod.Spec.Priority
	}
	return 0
}

// PodMLState holds ML-specific pod state
type PodMLState struct {
	PodName       string
	Namespace     string
	CPURequest    int64
	MemoryRequest int64
	Priority      int32
}

const podStateKey = "ml-scheduler-pod-state"