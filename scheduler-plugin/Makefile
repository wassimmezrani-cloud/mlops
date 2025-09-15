# ML-Scheduler Plugin Makefile

# Variables
PLUGIN_NAME = ml-scheduler
VERSION ?= v1.0.0
REGISTRY ?= hydatis.local
IMAGE_NAME = $(REGISTRY)/$(PLUGIN_NAME)
IMAGE_TAG = $(VERSION)
FULL_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)

# Go variables
GOOS ?= linux
GOARCH ?= amd64
CGO_ENABLED ?= 0

# Build directory
BUILD_DIR = ./build
BIN_DIR = $(BUILD_DIR)/bin

# Kubernetes variables
KUBECONFIG ?= ~/.kube/config
NAMESPACE ?= kube-system

# Test variables
TEST_TIMEOUT = 10m
COVERAGE_FILE = coverage.out

.PHONY: help
help: ## Display help information
	@echo "ML-Scheduler Plugin Build System"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*##"; printf "  %-20s %s\n", "Target", "Description"} /^[a-zA-Z_-]+:.*?##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

.PHONY: clean
clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -f $(COVERAGE_FILE)
	docker rmi $(FULL_IMAGE) 2>/dev/null || true

.PHONY: deps
deps: ## Download dependencies
	@echo "Downloading Go dependencies..."
	go mod download
	go mod verify

.PHONY: fmt
fmt: ## Format Go code
	@echo "Formatting Go code..."
	go fmt ./...
	goimports -w -local ml-scheduler-plugin ./pkg ./cmd

.PHONY: lint
lint: ## Lint Go code
	@echo "Linting Go code..."
	golangci-lint run ./...

.PHONY: vet
vet: ## Vet Go code
	@echo "Vetting Go code..."
	go vet ./...

.PHONY: test-unit
test-unit: ## Run unit tests
	@echo "Running unit tests..."
	mkdir -p $(BUILD_DIR)
	go test -v -race -timeout $(TEST_TIMEOUT) \
		-coverprofile=$(COVERAGE_FILE) \
		-covermode=atomic \
		./pkg/...

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "Running integration tests..."
	mkdir -p $(BUILD_DIR)
	go test -v -race -timeout $(TEST_TIMEOUT) \
		-tags=integration \
		./test/...

.PHONY: test
test: test-unit test-integration ## Run all tests

.PHONY: coverage
coverage: test-unit ## Generate test coverage report
	@echo "Generating coverage report..."
	go tool cover -html=$(COVERAGE_FILE) -o $(BUILD_DIR)/coverage.html
	go tool cover -func=$(COVERAGE_FILE)

.PHONY: build
build: ## Build the scheduler plugin binary
	@echo "Building ML-Scheduler plugin..."
	mkdir -p $(BIN_DIR)
	CGO_ENABLED=$(CGO_ENABLED) GOOS=$(GOOS) GOARCH=$(GOARCH) go build \
		-a -installsuffix cgo \
		-ldflags="-w -s -X main.version=$(VERSION) -X main.buildDate=$(shell date -u +'%Y-%m-%dT%H:%M:%SZ')" \
		-o $(BIN_DIR)/$(PLUGIN_NAME) \
		./cmd/main.go

.PHONY: build-image
build-image: ## Build Docker image
	@echo "Building Docker image $(FULL_IMAGE)..."
	docker build -t $(FULL_IMAGE) .
	docker tag $(FULL_IMAGE) $(IMAGE_NAME):latest

.PHONY: push-image
push-image: build-image ## Push Docker image to registry
	@echo "Pushing Docker image $(FULL_IMAGE)..."
	docker push $(FULL_IMAGE)
	docker push $(IMAGE_NAME):latest

.PHONY: load-image
load-image: build-image ## Load Docker image into local cluster
	@echo "Loading Docker image into local cluster..."
	kind load docker-image $(FULL_IMAGE) || echo "Kind not available, skipping..."

.PHONY: deploy-config
deploy-config: ## Deploy scheduler configuration
	@echo "Deploying scheduler configuration..."
	kubectl apply -f deployments/scheduler-config.yaml

.PHONY: deploy
deploy: build-image ## Deploy scheduler to Kubernetes
	@echo "Deploying ML-Scheduler to Kubernetes..."
	kubectl apply -f deployments/scheduler-deployment.yaml
	kubectl rollout status deployment/ml-scheduler -n $(NAMESPACE) --timeout=300s

.PHONY: undeploy
undeploy: ## Remove scheduler from Kubernetes
	@echo "Removing ML-Scheduler from Kubernetes..."
	kubectl delete -f deployments/scheduler-deployment.yaml --ignore-not-found=true

.PHONY: logs
logs: ## View scheduler logs
	@echo "Viewing ML-Scheduler logs..."
	kubectl logs -f deployment/ml-scheduler -n $(NAMESPACE)

.PHONY: status
status: ## Check scheduler status
	@echo "Checking ML-Scheduler status..."
	kubectl get deployment ml-scheduler -n $(NAMESPACE)
	kubectl get pods -l app=ml-scheduler -n $(NAMESPACE)
	kubectl get svc ml-scheduler-metrics -n $(NAMESPACE)

.PHONY: metrics
metrics: ## Port-forward metrics endpoint
	@echo "Port-forwarding metrics endpoint..."
	kubectl port-forward svc/ml-scheduler-metrics 8080:8080 -n $(NAMESPACE)

.PHONY: health
health: ## Check scheduler health
	@echo "Checking ML-Scheduler health..."
	kubectl port-forward svc/ml-scheduler-metrics 8081:8081 -n $(NAMESPACE) &
	sleep 2
	curl -s http://localhost:8081/health | jq .
	pkill -f "kubectl port-forward" || true

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	go test -v -bench=. -benchmem -timeout $(TEST_TIMEOUT) ./pkg/...

.PHONY: security-scan
security-scan: ## Run security scans
	@echo "Running security scans..."
	gosec ./...
	trivy image $(FULL_IMAGE) || echo "Trivy not available, skipping image scan..."

.PHONY: generate-docs
generate-docs: ## Generate API documentation
	@echo "Generating API documentation..."
	mkdir -p $(BUILD_DIR)/docs
	godoc -http=:6060 &
	sleep 2
	curl -s http://localhost:6060/pkg/ml-scheduler-plugin/pkg/ > $(BUILD_DIR)/docs/api.html || true
	pkill -f "godoc" || true

.PHONY: validate-config
validate-config: ## Validate Kubernetes configurations
	@echo "Validating Kubernetes configurations..."
	kubectl apply --dry-run=client -f deployments/
	kubeval deployments/*.yaml || echo "Kubeval not available, skipping validation..."

.PHONY: create-test-pod
create-test-pod: ## Create a test pod to verify scheduler
	@echo "Creating test pod with ML-Scheduler..."
	@kubectl apply -f - <<EOF || true
apiVersion: v1
kind: Pod
metadata:
  name: ml-scheduler-test-pod
  namespace: default
  annotations:
    scheduler: ml-scheduler
spec:
  schedulerName: ml-scheduler
  containers:
  - name: test-container
    image: nginx:latest
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 200m
        memory: 256Mi
  restartPolicy: Never
EOF
	@echo "Waiting for pod to be scheduled..."
	@kubectl wait --for=condition=PodScheduled pod/ml-scheduler-test-pod --timeout=60s || true
	@kubectl get pod ml-scheduler-test-pod -o wide || true

.PHONY: delete-test-pod
delete-test-pod: ## Delete test pod
	@echo "Deleting test pod..."
	kubectl delete pod ml-scheduler-test-pod --ignore-not-found=true

.PHONY: install-deps
install-deps: ## Install development dependencies
	@echo "Installing development dependencies..."
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
	go install golang.org/x/tools/cmd/goimports@latest
	go install github.com/securecodewarrior/gosec/v2/cmd/gosec@latest

.PHONY: verify
verify: fmt vet lint test ## Run all verification steps
	@echo "All verification steps completed successfully!"

.PHONY: release-prep
release-prep: clean verify build-image security-scan ## Prepare for release
	@echo "Release preparation completed!"
	@echo "Built image: $(FULL_IMAGE)"
	@echo "Ready for deployment!"

.PHONY: local-dev
local-dev: ## Start local development environment
	@echo "Starting local development environment..."
	kind create cluster --name ml-scheduler-dev || echo "Cluster already exists"
	$(MAKE) load-image
	$(MAKE) deploy
	@echo "Local development environment ready!"

.PHONY: local-cleanup
local-cleanup: ## Clean up local development environment
	@echo "Cleaning up local development environment..."
	kind delete cluster --name ml-scheduler-dev || true

# Default target
.DEFAULT_GOAL := help