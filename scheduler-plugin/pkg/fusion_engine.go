package pkg

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/go-logr/logr"
)

// DecisionFusionEngine combines ML recommendations using sophisticated fusion algorithms
type DecisionFusionEngine struct {
	config             *MLSchedulerConfig
	logger             logr.Logger
	decisionHistory    []FusionResult
	performanceTracker *PerformanceTracker
}

// PerformanceTracker tracks fusion engine performance over time
type PerformanceTracker struct {
	recentDecisions   []FusionResult
	accuracyHistory   []float64
	confidenceHistory []float64
	maxHistorySize    int
}

// FusionStrategy defines different fusion approaches
type FusionStrategy int

const (
	WeightedVoting FusionStrategy = iota
	ConsensusBasedFusion
	VetoSystemFusion
	ConfidenceWeightedFusion
	AdaptiveFusion
)

// NodeCandidate represents a node with its ML assessments
type NodeCandidate struct {
	NodeName            string
	XGBoostScore        float64
	XGBoostConfidence   float64
	QLearningScore      float64
	QLearningConfidence float64
	IsolationScore      float64
	IsolationConfidence float64
	AggregatedScore     float64
	Confidence          float64
	VetoReasons         []string
	SupportingReasons   []string
}

// NewDecisionFusionEngine creates a new decision fusion engine
func NewDecisionFusionEngine(config *MLSchedulerConfig, logger logr.Logger) *DecisionFusionEngine {
	return &DecisionFusionEngine{
		config:          config,
		logger:          logger,
		decisionHistory: make([]FusionResult, 0, 1000),
		performanceTracker: &PerformanceTracker{
			recentDecisions:   make([]FusionResult, 0, 100),
			accuracyHistory:   make([]float64, 0, 100),
			confidenceHistory: make([]float64, 0, 100),
			maxHistorySize:    100,
		},
	}
}

// FuseDecisions combines ML recommendations to select the best node
func (fe *DecisionFusionEngine) FuseDecisions(
	recommendations *MLRecommendations,
	availableNodes []string,
) (selectedNode string, confidence float64, reasoning string) {

	startTime := time.Now()
	defer func() {
		fe.logger.V(4).Info("Decision fusion completed",
			"duration", time.Since(startTime),
			"selectedNode", selectedNode,
			"confidence", confidence)
	}()

	// Convert recommendations to node candidates
	candidates := fe.createNodeCandidates(recommendations, availableNodes)

	// Determine fusion strategy based on available data
	strategy := fe.selectFusionStrategy(recommendations)

	// Apply fusion algorithm
	var result *FusionResult
	switch strategy {
	case WeightedVoting:
		result = fe.weightedVotingFusion(candidates, recommendations)
	case ConsensusBasedFusion:
		result = fe.consensusBasedFusion(candidates, recommendations)
	case VetoSystemFusion:
		result = fe.vetoSystemFusion(candidates, recommendations)
	case ConfidenceWeightedFusion:
		result = fe.confidenceWeightedFusion(candidates, recommendations)
	case AdaptiveFusion:
		result = fe.adaptiveFusion(candidates, recommendations)
	default:
		result = fe.weightedVotingFusion(candidates, recommendations)
	}

	// Record decision in history
	fe.recordDecision(*result)

	return result.SelectedNode, result.Confidence, result.Reasoning
}

// selectFusionStrategy chooses the best fusion strategy based on available data
func (fe *DecisionFusionEngine) selectFusionStrategy(recommendations *MLRecommendations) FusionStrategy {
	// Count available services with good confidence
	availableServices := 0
	totalConfidence := 0.0

	if recommendations.XGBoostPrediction.Confidence > 0.5 {
		availableServices++
		totalConfidence += recommendations.XGBoostPrediction.Confidence
	}

	if recommendations.QLearningChoice.Confidence > 0.5 {
		availableServices++
		totalConfidence += recommendations.QLearningChoice.Confidence
	}

	if recommendations.IsolationAssessment.Confidence > 0.5 {
		availableServices++
		totalConfidence += recommendations.IsolationAssessment.Confidence
	}

	averageConfidence := totalConfidence / float64(availableServices)

	// Strategy selection logic
	if availableServices >= 3 && averageConfidence > 0.8 {
		return AdaptiveFusion // Best case - all services with high confidence
	} else if availableServices >= 3 && averageConfidence > 0.6 {
		return ConsensusBasedFusion // All services available with moderate confidence
	} else if availableServices >= 2 && recommendations.IsolationAssessment.Confidence > 0.7 {
		return VetoSystemFusion // Use isolation as veto system
	} else if availableServices >= 2 {
		return ConfidenceWeightedFusion // Weight by confidence
	} else {
		return WeightedVoting // Fallback to simple weighted voting
	}
}

// weightedVotingFusion implements weighted voting based on configuration
func (fe *DecisionFusionEngine) weightedVotingFusion(
	candidates []NodeCandidate,
	recommendations *MLRecommendations,
) *FusionResult {

	var bestCandidate *NodeCandidate
	var bestScore float64 = -1.0

	for i := range candidates {
		candidate := &candidates[i]

		// Calculate weighted score
		weightedScore := 0.0
		totalWeight := 0.0

		if recommendations.XGBoostPrediction.Confidence > 0 {
			weight := fe.config.FusionWeights.XGBoostWeight
			weightedScore += candidate.XGBoostScore * weight
			totalWeight += weight
		}

		if recommendations.QLearningChoice.Confidence > 0 {
			weight := fe.config.FusionWeights.QLearningWeight
			weightedScore += candidate.QLearningScore * weight
			totalWeight += weight
		}

		if recommendations.IsolationAssessment.Confidence > 0 {
			weight := fe.config.FusionWeights.IsolationWeight
			weightedScore += candidate.IsolationScore * weight
			totalWeight += weight
		}

		// Normalize by total weight
		if totalWeight > 0 {
			candidate.AggregatedScore = weightedScore / totalWeight
		}

		// Calculate confidence as average of contributing service confidences
		candidate.Confidence = fe.calculateAggregatedConfidence(candidate, recommendations)

		// Select best candidate
		if candidate.AggregatedScore > bestScore {
			bestScore = candidate.AggregatedScore
			bestCandidate = candidate
		}
	}

	if bestCandidate == nil {
		return fe.createDefaultFusionResult(candidates, "weighted-voting-fallback")
	}

	reasoning := fmt.Sprintf("Weighted voting fusion: XGBoost=%.2f (weight %.1f), Q-Learning=%.2f (weight %.1f), Isolation=%.2f (weight %.1f)",
		bestCandidate.XGBoostScore, fe.config.FusionWeights.XGBoostWeight,
		bestCandidate.QLearningScore, fe.config.FusionWeights.QLearningWeight,
		bestCandidate.IsolationScore, fe.config.FusionWeights.IsolationWeight)

	return &FusionResult{
		SelectedNode:         bestCandidate.NodeName,
		FinalScore:           bestCandidate.AggregatedScore,
		Confidence:           bestCandidate.Confidence,
		Strategy:             "weighted-voting",
		Reasoning:            reasoning,
		ContributingServices: fe.getContributingServices(recommendations),
		Timestamp:            time.Now(),
	}
}

// consensusBasedFusion requires agreement between services
func (fe *DecisionFusionEngine) consensusBasedFusion(
	candidates []NodeCandidate,
	recommendations *MLRecommendations,
) *FusionResult {

	// Calculate consensus score for each candidate
	for i := range candidates {
		candidate := &candidates[i]

		scores := []float64{}
		if recommendations.XGBoostPrediction.Confidence > 0.5 {
			scores = append(scores, candidate.XGBoostScore)
		}
		if recommendations.QLearningChoice.Confidence > 0.5 {
			scores = append(scores, candidate.QLearningScore)
		}
		if recommendations.IsolationAssessment.Confidence > 0.5 {
			scores = append(scores, candidate.IsolationScore)
		}

		// Calculate consensus using harmonic mean (penalizes disagreement)
		if len(scores) >= 2 {
			candidate.AggregatedScore = fe.harmonicMean(scores)
			candidate.Confidence = fe.calculateConsensusConfidence(scores, recommendations)
		} else {
			// Fallback to single service score
			if len(scores) == 1 {
				candidate.AggregatedScore = scores[0]
				candidate.Confidence = 0.3 // Lower confidence for single service
			}
		}
	}

	// Select best consensus candidate
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].AggregatedScore > candidates[j].AggregatedScore
	})

	if len(candidates) == 0 {
		return fe.createDefaultFusionResult(candidates, "consensus-fallback")
	}

	bestCandidate := &candidates[0]

	reasoning := fmt.Sprintf("Consensus-based fusion: Agreement score %.2f with %d contributing services",
		bestCandidate.AggregatedScore, len(fe.getContributingServices(recommendations)))

	return &FusionResult{
		SelectedNode:         bestCandidate.NodeName,
		FinalScore:           bestCandidate.AggregatedScore,
		Confidence:           bestCandidate.Confidence,
		Strategy:             "consensus-based",
		Reasoning:            reasoning,
		ContributingServices: fe.getContributingServices(recommendations),
		Timestamp:            time.Now(),
	}
}

// vetoSystemFusion uses Isolation Forest as a veto system
func (fe *DecisionFusionEngine) vetoSystemFusion(
	candidates []NodeCandidate,
	recommendations *MLRecommendations,
) *FusionResult {

	// First, apply veto system based on anomaly detection
	vetoedNodes := make(map[string]string)

	if recommendations.IsolationAssessment.Confidence > 0.7 && recommendations.IsolationAssessment.IsAnomaly {
		for i := range candidates {
			candidate := &candidates[i]
			if candidate.IsolationScore < 0.3 { // Low isolation score indicates anomaly
				vetoedNodes[candidate.NodeName] = "Anomaly detected by Isolation Forest"
				candidate.VetoReasons = append(candidate.VetoReasons, "Anomaly detected")
			}
		}
	}

	// Filter out vetoed candidates
	validCandidates := []NodeCandidate{}
	for _, candidate := range candidates {
		if _, isVetoed := vetoedNodes[candidate.NodeName]; !isVetoed {
			validCandidates = append(validCandidates, candidate)
		}
	}

	// If all candidates are vetoed, use the least problematic one
	if len(validCandidates) == 0 {
		fe.logger.V(2).Info("All candidates vetoed, selecting least problematic")
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].IsolationScore > candidates[j].IsolationScore
		})
		validCandidates = candidates[:1]
	}

	// Apply weighted voting on remaining candidates
	fusionResult := fe.weightedVotingFusion(validCandidates, recommendations)
	fusionResult.Strategy = "veto-system"

	if len(vetoedNodes) > 0 {
		vetoInfo := fmt.Sprintf(" (%d nodes vetoed due to anomalies)", len(vetoedNodes))
		fusionResult.Reasoning += vetoInfo
	}

	return fusionResult
}

// confidenceWeightedFusion weights decisions by their confidence levels
func (fe *DecisionFusionEngine) confidenceWeightedFusion(
	candidates []NodeCandidate,
	recommendations *MLRecommendations,
) *FusionResult {

	for i := range candidates {
		candidate := &candidates[i]

		weightedScore := 0.0
		totalConfidenceWeight := 0.0

		// Weight scores by both configuration weights and confidence
		if recommendations.XGBoostPrediction.Confidence > 0 {
			configWeight := fe.config.FusionWeights.XGBoostWeight
			confidenceWeight := recommendations.XGBoostPrediction.Confidence
			combinedWeight := configWeight * confidenceWeight

			weightedScore += candidate.XGBoostScore * combinedWeight
			totalConfidenceWeight += combinedWeight
		}

		if recommendations.QLearningChoice.Confidence > 0 {
			configWeight := fe.config.FusionWeights.QLearningWeight
			confidenceWeight := recommendations.QLearningChoice.Confidence
			combinedWeight := configWeight * confidenceWeight

			weightedScore += candidate.QLearningScore * combinedWeight
			totalConfidenceWeight += combinedWeight
		}

		if recommendations.IsolationAssessment.Confidence > 0 {
			configWeight := fe.config.FusionWeights.IsolationWeight
			confidenceWeight := recommendations.IsolationAssessment.Confidence
			combinedWeight := configWeight * confidenceWeight

			weightedScore += candidate.IsolationScore * combinedWeight
			totalConfidenceWeight += combinedWeight
		}

		// Normalize
		if totalConfidenceWeight > 0 {
			candidate.AggregatedScore = weightedScore / totalConfidenceWeight
		}

		candidate.Confidence = totalConfidenceWeight / (fe.config.FusionWeights.XGBoostWeight +
			fe.config.FusionWeights.QLearningWeight + fe.config.FusionWeights.IsolationWeight)
	}

	// Select best candidate
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].AggregatedScore > candidates[j].AggregatedScore
	})

	if len(candidates) == 0 {
		return fe.createDefaultFusionResult(candidates, "confidence-weighted-fallback")
	}

	bestCandidate := &candidates[0]

	reasoning := fmt.Sprintf("Confidence-weighted fusion: Score %.2f with confidence-adjusted weights",
		bestCandidate.AggregatedScore)

	return &FusionResult{
		SelectedNode:         bestCandidate.NodeName,
		FinalScore:           bestCandidate.AggregatedScore,
		Confidence:           bestCandidate.Confidence,
		Strategy:             "confidence-weighted",
		Reasoning:            reasoning,
		ContributingServices: fe.getContributingServices(recommendations),
		Timestamp:            time.Now(),
	}
}

// adaptiveFusion adjusts strategy based on historical performance
func (fe *DecisionFusionEngine) adaptiveFusion(
	candidates []NodeCandidate,
	recommendations *MLRecommendations,
) *FusionResult {

	// Analyze recent performance to adjust weights
	adjustedWeights := fe.calculateAdaptiveWeights(recommendations)

	// Apply adaptive weighted scoring
	for i := range candidates {
		candidate := &candidates[i]

		weightedScore := 0.0
		totalWeight := 0.0

		if recommendations.XGBoostPrediction.Confidence > 0 {
			weight := adjustedWeights.XGBoostWeight
			weightedScore += candidate.XGBoostScore * weight
			totalWeight += weight
		}

		if recommendations.QLearningChoice.Confidence > 0 {
			weight := adjustedWeights.QLearningWeight
			weightedScore += candidate.QLearningScore * weight
			totalWeight += weight
		}

		if recommendations.IsolationAssessment.Confidence > 0 {
			weight := adjustedWeights.IsolationWeight
			weightedScore += candidate.IsolationScore * weight
			totalWeight += weight
		}

		if totalWeight > 0 {
			candidate.AggregatedScore = weightedScore / totalWeight
		}

		candidate.Confidence = fe.calculateAggregatedConfidence(candidate, recommendations)
	}

	// Select best candidate
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].AggregatedScore > candidates[j].AggregatedScore
	})

	if len(candidates) == 0 {
		return fe.createDefaultFusionResult(candidates, "adaptive-fallback")
	}

	bestCandidate := &candidates[0]

	reasoning := fmt.Sprintf("Adaptive fusion: Score %.2f with performance-adjusted weights (XGB:%.2f, QL:%.2f, ISO:%.2f)",
		bestCandidate.AggregatedScore,
		adjustedWeights.XGBoostWeight,
		adjustedWeights.QLearningWeight,
		adjustedWeights.IsolationWeight)

	return &FusionResult{
		SelectedNode:         bestCandidate.NodeName,
		FinalScore:           bestCandidate.AggregatedScore,
		Confidence:           bestCandidate.Confidence,
		Strategy:             "adaptive",
		Reasoning:            reasoning,
		ContributingServices: fe.getContributingServices(recommendations),
		Timestamp:            time.Now(),
	}
}

// Helper methods

// createNodeCandidates converts recommendations into node candidates for evaluation
func (fe *DecisionFusionEngine) createNodeCandidates(
	recommendations *MLRecommendations,
	availableNodes []string,
) []NodeCandidate {

	candidates := make([]NodeCandidate, len(availableNodes))

	for i, nodeName := range availableNodes {
		candidates[i] = NodeCandidate{
			NodeName:            nodeName,
			XGBoostScore:        recommendations.XGBoostPrediction.Score,
			XGBoostConfidence:   recommendations.XGBoostPrediction.Confidence,
			QLearningScore:      recommendations.QLearningChoice.Score,
			QLearningConfidence: recommendations.QLearningChoice.Confidence,
			IsolationScore:      recommendations.IsolationAssessment.Score,
			IsolationConfidence: recommendations.IsolationAssessment.Confidence,
			VetoReasons:         make([]string, 0),
			SupportingReasons:   make([]string, 0),
		}
	}

	return candidates
}

// calculateAggregatedConfidence calculates overall confidence from individual service confidences
func (fe *DecisionFusionEngine) calculateAggregatedConfidence(
	candidate *NodeCandidate,
	recommendations *MLRecommendations,
) float64 {

	confidences := []float64{}

	if recommendations.XGBoostPrediction.Confidence > 0 {
		confidences = append(confidences, candidate.XGBoostConfidence)
	}
	if recommendations.QLearningChoice.Confidence > 0 {
		confidences = append(confidences, candidate.QLearningConfidence)
	}
	if recommendations.IsolationAssessment.Confidence > 0 {
		confidences = append(confidences, candidate.IsolationConfidence)
	}

	if len(confidences) == 0 {
		return 0.0
	}

	// Use geometric mean for confidence (conservative approach)
	product := 1.0
	for _, conf := range confidences {
		product *= conf
	}

	return math.Pow(product, 1.0/float64(len(confidences)))
}

// calculateConsensusConfidence calculates confidence based on score agreement
func (fe *DecisionFusionEngine) calculateConsensusConfidence(
	scores []float64,
	recommendations *MLRecommendations,
) float64 {

	if len(scores) < 2 {
		return 0.3 // Low confidence for single service
	}

	// Calculate variance in scores (lower variance = higher consensus)
	mean := fe.arithmeticMean(scores)
	variance := 0.0
	for _, score := range scores {
		variance += math.Pow(score-mean, 2)
	}
	variance /= float64(len(scores))

	// Convert variance to confidence (lower variance = higher confidence)
	consensusConfidence := math.Max(0.0, 1.0-variance*2.0)

	// Weight by individual service confidences
	avgServiceConfidence := fe.getAverageServiceConfidence(recommendations)

	return (consensusConfidence + avgServiceConfidence) / 2.0
}

// harmonicMean calculates harmonic mean (penalizes outliers)
func (fe *DecisionFusionEngine) harmonicMean(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, val := range values {
		if val > 0 {
			sum += 1.0 / val
		}
	}

	if sum == 0 {
		return 0.0
	}

	return float64(len(values)) / sum
}

// arithmeticMean calculates arithmetic mean
func (fe *DecisionFusionEngine) arithmeticMean(values []float64) float64 {
	if len(values) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, val := range values {
		sum += val
	}

	return sum / float64(len(values))
}

// calculateAdaptiveWeights adjusts weights based on recent performance
func (fe *DecisionFusionEngine) calculateAdaptiveWeights(
	recommendations *MLRecommendations,
) WeightConfig {

	// Start with configured weights
	adaptedWeights := fe.config.FusionWeights

	// Analyze recent performance (simplified version)
	if len(fe.performanceTracker.recentDecisions) > 10 {
		// This would implement sophisticated weight adaptation based on:
		// - Historical accuracy of each service
		// - Correlation with successful placements
		// - Dynamic performance trends

		// For now, use a simple confidence-based adjustment
		totalConfidence := recommendations.XGBoostPrediction.Confidence +
			recommendations.QLearningChoice.Confidence +
			recommendations.IsolationAssessment.Confidence

		if totalConfidence > 0 {
			adaptedWeights.XGBoostWeight *= (1.0 + recommendations.XGBoostPrediction.Confidence/totalConfidence)
			adaptedWeights.QLearningWeight *= (1.0 + recommendations.QLearningChoice.Confidence/totalConfidence)
			adaptedWeights.IsolationWeight *= (1.0 + recommendations.IsolationAssessment.Confidence/totalConfidence)

			// Normalize weights
			if fe.config.FusionWeights.NormalizeWeights {
				total := adaptedWeights.XGBoostWeight + adaptedWeights.QLearningWeight + adaptedWeights.IsolationWeight
				if total > 0 {
					adaptedWeights.XGBoostWeight /= total
					adaptedWeights.QLearningWeight /= total
					adaptedWeights.IsolationWeight /= total
				}
			}
		}
	}

	return adaptedWeights
}

// getContributingServices returns list of services that contributed to the decision
func (fe *DecisionFusionEngine) getContributingServices(recommendations *MLRecommendations) []string {
	services := []string{}

	if recommendations.XGBoostPrediction.Confidence > 0 {
		services = append(services, "xgboost")
	}
	if recommendations.QLearningChoice.Confidence > 0 {
		services = append(services, "qlearning")
	}
	if recommendations.IsolationAssessment.Confidence > 0 {
		services = append(services, "isolation")
	}

	return services
}

// getAverageServiceConfidence calculates average confidence across all services
func (fe *DecisionFusionEngine) getAverageServiceConfidence(recommendations *MLRecommendations) float64 {
	confidences := []float64{
		recommendations.XGBoostPrediction.Confidence,
		recommendations.QLearningChoice.Confidence,
		recommendations.IsolationAssessment.Confidence,
	}

	validConfidences := []float64{}
	for _, conf := range confidences {
		if conf > 0 {
			validConfidences = append(validConfidences, conf)
		}
	}

	return fe.arithmeticMean(validConfidences)
}

// createDefaultFusionResult creates a fallback result when fusion fails
func (fe *DecisionFusionEngine) createDefaultFusionResult(candidates []NodeCandidate, strategy string) *FusionResult {
	// Select first available candidate or create a default
	selectedNode := "unknown"
	if len(candidates) > 0 {
		selectedNode = candidates[0].NodeName
	}

	return &FusionResult{
		SelectedNode:         selectedNode,
		FinalScore:           0.5, // Neutral score
		Confidence:           0.1, // Very low confidence
		Strategy:             strategy,
		Reasoning:            "Default fallback due to fusion failure",
		ContributingServices: []string{},
		Timestamp:            time.Now(),
	}
}

// recordDecision stores decision in history for learning and analysis
func (fe *DecisionFusionEngine) recordDecision(result FusionResult) {
	// Add to performance tracker
	fe.performanceTracker.recentDecisions = append(fe.performanceTracker.recentDecisions, result)

	// Maintain history size limit
	if len(fe.performanceTracker.recentDecisions) > fe.performanceTracker.maxHistorySize {
		fe.performanceTracker.recentDecisions = fe.performanceTracker.recentDecisions[1:]
	}

	// Add to main history
	fe.decisionHistory = append(fe.decisionHistory, result)

	// Maintain main history size limit
	if len(fe.decisionHistory) > 1000 {
		fe.decisionHistory = fe.decisionHistory[1:]
	}

	fe.logger.V(4).Info("Decision recorded in fusion engine history",
		"strategy", result.Strategy,
		"confidence", result.Confidence,
		"historySize", len(fe.decisionHistory))
}

// GetPerformanceMetrics returns performance metrics for monitoring
func (fe *DecisionFusionEngine) GetPerformanceMetrics() map[string]interface{} {
	metrics := make(map[string]interface{})

	if len(fe.performanceTracker.recentDecisions) > 0 {
		// Calculate strategy usage distribution
		strategyCount := make(map[string]int)
		totalConfidence := 0.0

		for _, decision := range fe.performanceTracker.recentDecisions {
			strategyCount[decision.Strategy]++
			totalConfidence += decision.Confidence
		}

		metrics["strategy_distribution"] = strategyCount
		metrics["average_confidence"] = totalConfidence / float64(len(fe.performanceTracker.recentDecisions))
		metrics["total_decisions"] = len(fe.performanceTracker.recentDecisions)
	}

	return metrics
}

// ValidateFusionConfiguration validates the fusion engine configuration
func (fe *DecisionFusionEngine) ValidateFusionConfiguration() []string {
	warnings := []string{}

	// Check weight configuration
	totalWeight := fe.config.FusionWeights.XGBoostWeight +
		fe.config.FusionWeights.QLearningWeight +
		fe.config.FusionWeights.IsolationWeight

	if totalWeight <= 0 {
		warnings = append(warnings, "Total fusion weights sum to zero or negative value")
	}

	if fe.config.FusionWeights.XGBoostWeight < 0 ||
		fe.config.FusionWeights.QLearningWeight < 0 ||
		fe.config.FusionWeights.IsolationWeight < 0 {
		warnings = append(warnings, "Negative weights detected in fusion configuration")
	}

	// Check confidence thresholds
	for service, threshold := range fe.config.ConfidenceThresholds {
		if threshold < 0 || threshold > 1 {
			warnings = append(warnings, fmt.Sprintf("Invalid confidence threshold for %s: %f (should be 0-1)", service, threshold))
		}
	}

	return warnings
}
