# ML-Kubernetes-Scheduler Makefile
# Automation pour build, test, deploy et operations

.PHONY: help build test deploy clean setup install-deps verify-deployment

# Variables
PROJECT_NAME := ml-kubernetes-scheduler
NAMESPACE := ml-scheduler
VERSION ?= v1.0.0
REGISTRY ?= registry.hydatis.tn
GO_VERSION := 1.21
PYTHON_VERSION := 3.9

# Colors pour output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

##@ Help
help: ## Afficher cette aide
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Installation
install-deps: ## Installer toutes les dépendances (Go, Python, K8s tools)
	@echo "$(GREEN)📦 Installation des dépendances...$(NC)"
	@echo "$(BLUE)🔧 Vérification Go $(GO_VERSION)...$(NC)"
	@go version || (echo "$(RED)❌ Go non installé$(NC)" && exit 1)
	@echo "$(BLUE)🐍 Vérification Python $(PYTHON_VERSION)...$(NC)"
	@python3 --version || (echo "$(RED)❌ Python3 non installé$(NC)" && exit 1)
	@echo "$(BLUE)☸️ Vérification kubectl...$(NC)"
	@kubectl version --client || (echo "$(RED)❌ kubectl non installé$(NC)" && exit 1)
	@echo "$(BLUE)📥 Installation dépendances Go...$(NC)"
	@go mod download
	@go mod tidy
	@echo "$(BLUE)📥 Installation dépendances Python ML Services...$(NC)"
	@pip3 install -r ml-services/xgboost-predictor/requirements.txt
	@pip3 install -r ml-services/qlearning-optimizer/requirements.txt
	@pip3 install -r ml-services/isolation-detector/requirements.txt
	@echo "$(GREEN)✅ Toutes les dépendances installées avec succès !$(NC)"

setup-env: ## Configurer l'environnement de développement
	@echo "$(GREEN)🛠️ Configuration environnement développement...$(NC)"
	@echo "$(BLUE)📁 Création namespace Kubernetes...$(NC)"
	@kubectl create namespace $(NAMESPACE) || echo "Namespace $(NAMESPACE) déjà existant"
	@echo "$(BLUE)🏷️ Labeling namespace...$(NC)"
	@kubectl label namespace $(NAMESPACE) name=$(NAMESPACE) --overwrite
	@echo "$(BLUE)⚙️ Application configurations de base...$(NC)"
	@kubectl apply -f deployments/kubernetes/namespace.yaml
	@kubectl apply -f config/environments/development.yaml
	@echo "$(GREEN)✅ Environnement configuré avec succès !$(NC)"

##@ Build & Development
build: ## Builder tous les composants
	@echo "$(GREEN)🔨 Build de tous les composants...$(NC)"
	@$(MAKE) build-scheduler
	@$(MAKE) build-ml-services
	@echo "$(GREEN)✅ Build terminé avec succès !$(NC)"

build-scheduler: ## Builder le scheduler Go
	@echo "$(BLUE)🔨 Build du ML-Scheduler (Go)...$(NC)"
	@go build -o bin/ml-scheduler ./cmd/scheduler/
	@go build -o bin/data-collector ./cmd/data-collector/
	@go build -o bin/metrics-exporter ./cmd/metrics-exporter/
	@echo "$(GREEN)✅ Scheduler Go buildé avec succès !$(NC)"

build-ml-services: ## Builder les services ML Python
	@echo "$(BLUE)🧠 Build des services ML (Python)...$(NC)"
	@docker build -t $(REGISTRY)/xgboost-predictor:$(VERSION) ./ml-services/xgboost-predictor/
	@docker build -t $(REGISTRY)/qlearning-optimizer:$(VERSION) ./ml-services/qlearning-optimizer/
	@docker build -t $(REGISTRY)/isolation-detector:$(VERSION) ./ml-services/isolation-detector/
	@echo "$(GREEN)✅ Services ML buildés avec succès !$(NC)"

build-docker: ## Builder toutes les images Docker
	@echo "$(BLUE)🐳 Build des images Docker...$(NC)"
	@docker build -t $(REGISTRY)/$(PROJECT_NAME):$(VERSION) ./cmd/scheduler/
	@$(MAKE) build-ml-services
	@echo "$(GREEN)✅ Images Docker buildées avec succès !$(NC)"

##@ Testing
test: ## Exécuter tous les tests
	@echo "$(GREEN)🧪 Exécution de tous les tests...$(NC)"
	@$(MAKE) test-unit
	@$(MAKE) test-integration
	@$(MAKE) test-e2e
	@echo "$(GREEN)✅ Tous les tests passés avec succès !$(NC)"

test-unit: ## Tests unitaires Go et Python
	@echo "$(BLUE)🔬 Tests unitaires...$(NC)"
	@echo "$(YELLOW)Go Tests:$(NC)"
	@go test -v -cover ./pkg/... ./cmd/...
	@echo "$(YELLOW)Python Tests:$(NC)"
	@cd ml-services/xgboost-predictor && python -m pytest tests/ -v
	@cd ml-services/qlearning-optimizer && python -m pytest tests/ -v
	@cd ml-services/isolation-detector && python -m pytest tests/ -v

test-integration: ## Tests d'intégration
	@echo "$(BLUE)🔗 Tests d'intégration...$(NC)"
	@go test -v -tags=integration ./tests/integration/...

test-e2e: ## Tests end-to-end
	@echo "$(BLUE)🎯 Tests end-to-end...$(NC)"
	@kubectl apply -f tests/e2e/test-pods.yaml
	@go test -v -tags=e2e ./tests/e2e/...
	@kubectl delete -f tests/e2e/test-pods.yaml

test-load: ## Tests de charge
	@echo "$(BLUE)⚡ Tests de charge...$(NC)"
	@go test -v -tags=load ./tests/load/...

test-chaos: ## Tests chaos engineering
	@echo "$(BLUE)💥 Tests chaos engineering...$(NC)"
	@kubectl apply -f tests/chaos/
	@sleep 30
	@kubectl delete -f tests/chaos/

##@ Deployment
deploy: ## Déployer le ML-Scheduler complet
	@echo "$(GREEN)🚀 Déploiement ML-Scheduler...$(NC)"
	@$(MAKE) deploy-storage
	@$(MAKE) deploy-scheduler
	@$(MAKE) deploy-ml-services
	@$(MAKE) deploy-monitoring
	@echo "$(GREEN)✅ Déploiement terminé avec succès !$(NC)"

deploy-storage: ## Déployer le stockage Longhorn
	@echo "$(BLUE)💾 Déploiement stockage Longhorn...$(NC)"
	@kubectl apply -f storage/longhorn/
	@kubectl wait --for=condition=Ready pod -l app=longhorn-manager -n longhorn-system --timeout=300s

deploy-scheduler: ## Déployer le scheduler principal
	@echo "$(BLUE)⚙️ Déploiement ML-Scheduler...$(NC)"
	@kubectl apply -f deployments/kubernetes/scheduler/
	@kubectl rollout status deployment/ml-scheduler -n $(NAMESPACE)

deploy-ml-services: ## Déployer les services ML
	@echo "$(BLUE)🧠 Déploiement services ML...$(NC)"
	@kubectl apply -f deployments/kubernetes/ml-services/
	@kubectl apply -f deployments/kserve/
	@kubectl wait --for=condition=Ready pod -l app=ml-services -n $(NAMESPACE) --timeout=300s

deploy-monitoring: ## Déployer le monitoring
	@echo "$(BLUE)📊 Déploiement monitoring...$(NC)"
	@kubectl apply -f deployments/kubernetes/monitoring/
	@kubectl wait --for=condition=Ready pod -l app=prometheus -n monitoring --timeout=300s

##@ Operations
verify-deployment: ## Vérifier le déploiement
	@echo "$(GREEN)✅ Vérification du déploiement...$(NC)"
	@echo "$(BLUE)📊 État des pods:$(NC)"
	@kubectl get pods -n $(NAMESPACE)
	@echo "$(BLUE)🌐 Services:$(NC)"
	@kubectl get services -n $(NAMESPACE)
	@echo "$(BLUE)💾 Volumes persistants:$(NC)"
	@kubectl get pv | grep $(NAMESPACE)
	@echo "$(BLUE)📈 Métriques Prometheus:$(NC)"
	@kubectl get servicemonitor -n $(NAMESPACE)
	@echo "$(BLUE)🧠 KServe InferenceServices:$(NC)"
	@kubectl get inferenceservice -n $(NAMESPACE)

logs: ## Afficher les logs du scheduler
	@echo "$(BLUE)📋 Logs ML-Scheduler:$(NC)"
	@kubectl logs -f deployment/ml-scheduler -n $(NAMESPACE)

logs-ml: ## Afficher les logs des services ML
	@echo "$(BLUE)📋 Logs Services ML:$(NC)"
	@kubectl logs -f deployment/xgboost-predictor -n $(NAMESPACE) &
	@kubectl logs -f deployment/qlearning-optimizer -n $(NAMESPACE) &
	@kubectl logs -f deployment/isolation-detector -n $(NAMESPACE) &

status: ## Afficher le statut complet
	@echo "$(GREEN)📊 Statut ML-Scheduler HYDATIS$(NC)"
	@echo "$(BLUE)Cluster Info:$(NC)"
	@kubectl cluster-info
	@echo "$(BLUE)Nodes:$(NC)"
	@kubectl get nodes
	@echo "$(BLUE)ML-Scheduler Namespace:$(NC)"
	@kubectl get all -n $(NAMESPACE)
	@echo "$(BLUE)Kubeflow Status:$(NC)"
	@kubectl get pods -n kubeflow | head -10
	@echo "$(BLUE)Longhorn Status:$(NC)"
	@kubectl get pods -n longhorn-system | head -5
	@echo "$(BLUE)Monitoring Status:$(NC)"
	@kubectl get pods -n monitoring | head -5

##@ Monitoring & Debugging
monitor: ## Ouvrir les dashboards de monitoring
	@echo "$(GREEN)📊 Ouverture dashboards monitoring...$(NC)"
	@echo "$(BLUE)Grafana Dashboard:$(NC) http://localhost:3000"
	@kubectl port-forward -n monitoring svc/grafana 3000:3000 &
	@echo "$(BLUE)Prometheus:$(NC) http://localhost:9090"
	@kubectl port-forward -n monitoring svc/prometheus 9090:9090 &
	@echo "$(BLUE)Kubeflow Dashboard:$(NC) http://localhost:8080"
	@kubectl port-forward -n kubeflow svc/kubeflow-dashboard 8080:8080 &
	@echo "$(YELLOW)Dashboards ouverts ! Ctrl+C pour fermer$(NC)"

debug-scheduler: ## Debug du scheduler
	@echo "$(BLUE)🐛 Debug ML-Scheduler...$(NC)"
	@kubectl describe pod -l app=ml-scheduler -n $(NAMESPACE)
	@kubectl logs --tail=100 -l app=ml-scheduler -n $(NAMESPACE)

debug-ml-services: ## Debug des services ML
	@echo "$(BLUE)🧠 Debug Services ML...$(NC)"
	@kubectl describe pod -l app=ml-services -n $(NAMESPACE)
	@kubectl logs --tail=50 -l app=xgboost-predictor -n $(NAMESPACE)
	@kubectl logs --tail=50 -l app=qlearning-optimizer -n $(NAMESPACE)
	@kubectl logs --tail=50 -l app=isolation-detector -n $(NAMESPACE)

##@ Data Management
collect-data: ## Collecter les données historiques
	@echo "$(BLUE)📊 Collecte données historiques...$(NC)"
	@./scripts/data/collect-historical-data.sh

process-features: ## Traiter les features pour ML
	@echo "$(BLUE)⚙️ Traitement features ML...$(NC)"
	@./scripts/data/process-features.sh

backup: ## Backup des données et configuration
	@echo "$(BLUE)💾 Backup données et config...$(NC)"
	@./scripts/data/backup-data.sh
	@./scripts/maintenance/backup-config.sh

restore: ## Restore des données
	@echo "$(BLUE)♻️ Restore données...$(NC)"
	@./scripts/data/restore-data.sh

##@ Kubeflow Operations
kubeflow-pipelines: ## Exécuter les pipelines Kubeflow
	@echo "$(BLUE)🔄 Exécution pipelines Kubeflow...$(NC)"
	@python kubeflow/pipelines/training-pipeline.py

katib-tuning: ## Lancer hyperparameter tuning avec Katib
	@echo "$(BLUE)🎯 Hyperparameter tuning Katib...$(NC)"
	@kubectl apply -f kubeflow/katib/xgboost-tuning.yaml
	@kubectl apply -f kubeflow/katib/qlearning-tuning.yaml
	@kubectl apply -f kubeflow/katib/isolation-tuning.yaml

jupyter: ## Ouvrir Jupyter Notebooks
	@echo "$(BLUE)📓 Ouverture Jupyter Notebooks...$(NC)"
	@kubectl port-forward -n kubeflow svc/jupyter-web-app-service 8888:80

mlflow: ## Ouvrir MLflow UI
	@echo "$(BLUE)📈 Ouverture MLflow UI...$(NC)"
	@kubectl port-forward -n kubeflow svc/mlflow-server 5000:5000

##@ Maintenance
clean: ## Nettoyer les ressources
	@echo "$(YELLOW)🧹 Nettoyage des ressources...$(NC)"
	@docker system prune -f
	@go clean -cache
	@kubectl delete pods --field-selector=status.phase=Succeeded -n $(NAMESPACE)

uninstall: ## Désinstaller complètement
	@echo "$(RED)🗑️ Désinstallation ML-Scheduler...$(NC)"
	@kubectl delete namespace $(NAMESPACE)
	@kubectl delete -f deployments/kubernetes/monitoring/
	@docker rmi $(REGISTRY)/$(PROJECT_NAME):$(VERSION) || true

update: ## Mettre à jour le déploiement
	@echo "$(BLUE)🔄 Mise à jour déploiement...$(NC)"
	@$(MAKE) build
	@kubectl rollout restart deployment/ml-scheduler -n $(NAMESPACE)
	@kubectl rollout restart deployment/xgboost-predictor -n $(NAMESPACE)
	@kubectl rollout restart deployment/qlearning-optimizer -n $(NAMESPACE)
	@kubectl rollout restart deployment/isolation-detector -n $(NAMESPACE)

##@ Performance Testing
performance-test: ## Tests de performance complets
	@echo "$(BLUE)⚡ Tests de performance...$(NC)"
	@./scripts/monitoring/performance-test.sh

stress-test: ## Tests de stress du cluster
	@echo "$(BLUE)💪 Tests de stress...$(NC)"
	@./scripts/monitoring/stress-test.sh

benchmark: ## Benchmark du scheduler vs default
	@echo "$(BLUE)🏆 Benchmark ML-Scheduler vs Default...$(NC)"
	@go test -v -bench=. ./tests/e2e/

##@ Quick Commands
quick-status: ## Statut rapide
	@kubectl get pods -n $(NAMESPACE) -o wide

quick-logs: ## Logs rapides scheduler
	@kubectl logs --tail=20 -l app=ml-scheduler -n $(NAMESPACE)

quick-restart: ## Restart rapide scheduler
	@kubectl rollout restart deployment/ml-scheduler -n $(NAMESPACE)

##@ Development Helpers
dev-setup: ## Setup développement complet
	@$(MAKE) install-deps
	@$(MAKE) setup-env
	@$(MAKE) build
	@echo "$(GREEN)🎉 Environnement développement prêt !$(NC)"

dev-deploy: ## Déploiement développement rapide
	@$(MAKE) build-scheduler
	@kubectl apply -f deployments/kubernetes/scheduler/
	@kubectl rollout status deployment/ml-scheduler -n $(NAMESPACE)

dev-test: ## Tests développement rapides
	@$(MAKE) test-unit
	@$(MAKE) test-integration

##@ Information
version: ## Afficher la version
	@echo "$(GREEN)ML-Scheduler Version: $(VERSION)$(NC)"
	@echo "$(BLUE)Go Version: $(shell go version)$(NC)"
	@echo "$(BLUE)Kubectl Version: $(shell kubectl version --client --short)$(NC)"
	@echo "$(BLUE)Docker Version: $(shell docker --version)$(NC)"

info: ## Informations système
	@echo "$(GREEN)🏗️ ML-Kubernetes-Scheduler - HYDATIS$(NC)"
	@echo "$(BLUE)📋 Informations Système:$(NC)"
	@echo "  Project: $(PROJECT_NAME)"
	@echo "  Version: $(VERSION)"
	@echo "  Namespace: $(NAMESPACE)"
	@echo "  Registry: $(REGISTRY)"
	@echo "  Go Version: $(GO_VERSION)"
	@echo "  Python Version: $(PYTHON_VERSION)"
	@echo ""
	@echo "$(GREEN)🎯 Mission: Transformer HYDATIS de SPOF vers HA Cluster$(NC)"
	@echo "$(GREEN)🧠 Algorithmes: XGBoost + Q-Learning + Isolation Forest$(NC)"
	@echo "$(GREEN)⚡ Impact: 99.7% availability, 15x capacity, 1,428% ROI$(NC)"

# Default target
.DEFAULT_GOAL := help
