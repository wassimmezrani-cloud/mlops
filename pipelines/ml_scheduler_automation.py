#!/usr/bin/env python3
"""
ML-Scheduler Pipeline Automation - Étape 7 Action 4
Configuration exécution automatique et monitoring pipeline
Triggers, scheduling, monitoring et alerting complets
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLSchedulerAutomation:
    """
    Automation complète du pipeline ML-Scheduler
    Scheduling, triggers, monitoring et alerting
    """
    
    def __init__(self):
        """Initialize automation configuration"""
        self.automation_config = {
            'pipeline_name': 'ml-scheduler-training-pipeline',
            'version': '1.0.0',
            'automation_date': datetime.now().isoformat(),
            'kubeflow_namespace': 'kubeflow',
            'monitoring_namespace': 'ml-scheduler-monitoring'
        }
        
        logger.info("ML-Scheduler Pipeline Automation initialized")
    
    def create_scheduling_configuration(self) -> Dict[str, Any]:
        """
        Create comprehensive scheduling configuration
        
        Returns:
            Scheduling configuration
        """
        scheduling_config = {
            # Weekly re-training schedule
            'weekly_retraining': {
                'enabled': True,
                'schedule': '0 2 * * 0',  # Every Sunday at 2 AM
                'timezone': 'UTC',
                'description': 'Weekly complete trio re-training',
                'max_duration_hours': 4,
                'retry_policy': {
                    'max_retries': 2,
                    'retry_delay_minutes': 30
                }
            },
            
            # Event-driven triggers
            'event_triggers': {
                'data_threshold_trigger': {
                    'enabled': True,
                    'condition': 'new_data_samples > 10000',
                    'description': 'Trigger when significant new data available',
                    'cooldown_hours': 24
                },
                'performance_degradation_trigger': {
                    'enabled': True,
                    'condition': 'model_accuracy < 0.8',
                    'description': 'Trigger on model performance degradation',
                    'cooldown_hours': 6
                },
                'incident_trigger': {
                    'enabled': True,
                    'condition': 'cluster_incidents > 5',
                    'description': 'Trigger after cluster incidents spike',
                    'cooldown_hours': 12
                }
            },
            
            # Manual triggers
            'manual_triggers': {
                'admin_trigger': {
                    'enabled': True,
                    'roles': ['ml-engineer', 'platform-admin'],
                    'approval_required': True,
                    'description': 'Manual admin-triggered training'
                },
                'emergency_retrain': {
                    'enabled': True,
                    'roles': ['platform-admin', 'sre-oncall'],
                    'approval_required': False,
                    'description': 'Emergency retraining for incidents'
                }
            },
            
            # Pipeline resource scheduling
            'resource_scheduling': {
                'preferred_node_selector': {
                    'workload-type': 'ml-training',
                    'instance-type': 'compute-optimized'
                },
                'resource_limits': {
                    'max_parallel_pipelines': 2,
                    'max_cpu_cores': 32,
                    'max_memory_gb': 128
                },
                'priority_class': 'ml-scheduler-training',
                'preemption_policy': 'PreemptLowerPriority'
            }
        }
        
        return scheduling_config
    
    def create_monitoring_configuration(self) -> Dict[str, Any]:
        """
        Create pipeline monitoring and alerting configuration
        
        Returns:
            Monitoring configuration
        """
        monitoring_config = {
            # Pipeline execution monitoring
            'pipeline_metrics': {
                'execution_duration': {
                    'metric_name': 'ml_scheduler_pipeline_duration_seconds',
                    'description': 'Pipeline execution duration',
                    'labels': ['pipeline_name', 'status'],
                    'alert_threshold': 7200  # 2 hours
                },
                'component_duration': {
                    'metric_name': 'ml_scheduler_component_duration_seconds',
                    'description': 'Individual component duration',
                    'labels': ['component_name', 'status'],
                    'alert_thresholds': {
                        'data_collection': 1800,  # 30 min
                        'preprocessing': 1200,    # 20 min
                        'trio_training': 2700,    # 45 min
                        'validation': 900,        # 15 min
                        'deployment': 600         # 10 min
                    }
                },
                'pipeline_success_rate': {
                    'metric_name': 'ml_scheduler_pipeline_success_rate',
                    'description': 'Pipeline success rate percentage',
                    'alert_threshold': 0.8  # 80% success rate
                },
                'model_performance': {
                    'metric_name': 'ml_scheduler_model_performance',
                    'description': 'Model performance scores',
                    'labels': ['algorithm', 'metric_type'],
                    'alert_thresholds': {
                        'xgboost_r2': 0.85,
                        'qlearning_improvement': 0.15,
                        'isolation_f1': 0.85,
                        'trio_integration_score': 75.0
                    }
                }
            },
            
            # Resource utilization monitoring
            'resource_metrics': {
                'cpu_utilization': {
                    'metric_name': 'ml_scheduler_pipeline_cpu_usage',
                    'description': 'Pipeline CPU utilization',
                    'alert_threshold': 0.9
                },
                'memory_utilization': {
                    'metric_name': 'ml_scheduler_pipeline_memory_usage',
                    'description': 'Pipeline memory utilization',
                    'alert_threshold': 0.85
                },
                'storage_usage': {
                    'metric_name': 'ml_scheduler_pipeline_storage_usage',
                    'description': 'Pipeline storage usage',
                    'alert_threshold': 0.8
                }
            },
            
            # Data quality monitoring
            'data_quality_metrics': {
                'data_completeness': {
                    'metric_name': 'ml_scheduler_data_completeness',
                    'description': 'Data completeness percentage',
                    'alert_threshold': 0.95
                },
                'data_freshness': {
                    'metric_name': 'ml_scheduler_data_freshness_hours',
                    'description': 'Data freshness in hours',
                    'alert_threshold': 25  # Data older than 25 hours
                },
                'feature_drift': {
                    'metric_name': 'ml_scheduler_feature_drift_score',
                    'description': 'Feature drift detection score',
                    'alert_threshold': 0.7
                }
            }
        }
        
        return monitoring_config
    
    def create_alerting_configuration(self) -> Dict[str, Any]:
        """
        Create comprehensive alerting configuration
        
        Returns:
            Alerting configuration
        """
        alerting_config = {
            # Alert channels
            'alert_channels': {
                'slack': {
                    'enabled': True,
                    'webhook_url': 'https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX',
                    'channel': '#ml-scheduler-alerts',
                    'severity_levels': ['high', 'critical']
                },
                'email': {
                    'enabled': True,
                    'smtp_server': 'smtp.company.com',
                    'recipients': [
                        'ml-team@company.com',
                        'platform-team@company.com'
                    ],
                    'severity_levels': ['medium', 'high', 'critical']
                },
                'pagerduty': {
                    'enabled': True,
                    'integration_key': 'your-pagerduty-integration-key',
                    'severity_levels': ['critical']
                }
            },
            
            # Alert rules
            'alert_rules': [
                {
                    'name': 'ML-Scheduler Pipeline Failed',
                    'severity': 'high',
                    'condition': 'ml_scheduler_pipeline_success_rate < 0.8',
                    'duration': '5m',
                    'description': 'ML-Scheduler pipeline success rate below 80%',
                    'runbook_url': 'https://wiki.company.com/ml-scheduler-troubleshooting',
                    'channels': ['slack', 'email']
                },
                {
                    'name': 'Pipeline Execution Too Long',
                    'severity': 'medium',
                    'condition': 'ml_scheduler_pipeline_duration_seconds > 7200',
                    'duration': '1m',
                    'description': 'Pipeline execution taking longer than 2 hours',
                    'channels': ['slack']
                },
                {
                    'name': 'Model Performance Degradation',
                    'severity': 'high',
                    'condition': 'ml_scheduler_model_performance{metric_type="business_score"} < 75',
                    'duration': '1m',
                    'description': 'Model performance below business threshold',
                    'channels': ['slack', 'email']
                },
                {
                    'name': 'Data Quality Issues',
                    'severity': 'medium',
                    'condition': 'ml_scheduler_data_completeness < 0.95',
                    'duration': '10m',
                    'description': 'Data quality issues detected',
                    'channels': ['slack']
                },
                {
                    'name': 'Pipeline Resource Exhaustion',
                    'severity': 'critical',
                    'condition': 'ml_scheduler_pipeline_memory_usage > 0.95',
                    'duration': '2m',
                    'description': 'Pipeline consuming too much memory',
                    'channels': ['slack', 'email', 'pagerduty']
                }
            ],
            
            # Escalation policies
            'escalation_policies': {
                'ml_team_escalation': {
                    'levels': [
                        {'duration': '5m', 'contacts': ['ml-team@company.com']},
                        {'duration': '15m', 'contacts': ['ml-lead@company.com']},
                        {'duration': '30m', 'contacts': ['platform-team@company.com']}
                    ]
                },
                'critical_escalation': {
                    'levels': [
                        {'duration': '2m', 'contacts': ['oncall-sre@company.com']},
                        {'duration': '10m', 'contacts': ['platform-lead@company.com']},
                        {'duration': '20m', 'contacts': ['cto@company.com']}
                    ]
                }
            }
        }
        
        return alerting_config
    
    def create_kubernetes_manifests(self) -> Dict[str, Dict]:
        """
        Create Kubernetes manifests for automation
        
        Returns:
            Dictionary of Kubernetes manifests
        """
        manifests = {}
        
        # 1. CronJob for weekly scheduling
        manifests['weekly_cronjob'] = {
            'apiVersion': 'batch/v1',
            'kind': 'CronJob',
            'metadata': {
                'name': 'ml-scheduler-weekly-training',
                'namespace': self.automation_config['kubeflow_namespace'],
                'labels': {
                    'app': 'ml-scheduler',
                    'component': 'automation',
                    'schedule-type': 'weekly'
                }
            },
            'spec': {
                'schedule': '0 2 * * 0',  # Every Sunday at 2 AM
                'timeZone': 'UTC',
                'concurrencyPolicy': 'Forbid',
                'successfulJobsHistoryLimit': 3,
                'failedJobsHistoryLimit': 3,
                'jobTemplate': {
                    'spec': {
                        'template': {
                            'spec': {
                                'serviceAccountName': 'ml-scheduler-pipeline-runner',
                                'containers': [{
                                    'name': 'pipeline-trigger',
                                    'image': 'ml-scheduler/pipeline-trigger:v1.0.0',
                                    'command': ['python', '/app/trigger_pipeline.py'],
                                    'args': ['--pipeline-name', 'ml-scheduler-training-pipeline'],
                                    'env': [
                                        {
                                            'name': 'KUBEFLOW_ENDPOINT',
                                            'value': 'http://ml-pipeline.kubeflow.svc.cluster.local:8888'
                                        },
                                        {
                                            'name': 'PIPELINE_VERSION',
                                            'value': '1.0.0'
                                        }
                                    ],
                                    'resources': {
                                        'requests': {'cpu': '100m', 'memory': '128Mi'},
                                        'limits': {'cpu': '200m', 'memory': '256Mi'}
                                    }
                                }],
                                'restartPolicy': 'OnFailure'
                            }
                        }
                    }
                }
            }
        }
        
        # 2. ServiceAccount and RBAC
        manifests['service_account'] = {
            'apiVersion': 'v1',
            'kind': 'ServiceAccount',
            'metadata': {
                'name': 'ml-scheduler-pipeline-runner',
                'namespace': self.automation_config['kubeflow_namespace']
            }
        }
        
        manifests['cluster_role'] = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'ClusterRole',
            'metadata': {
                'name': 'ml-scheduler-pipeline-runner'
            },
            'rules': [
                {
                    'apiGroups': ['argoproj.io'],
                    'resources': ['workflows'],
                    'verbs': ['create', 'get', 'list', 'watch']
                },
                {
                    'apiGroups': [''],
                    'resources': ['pods', 'configmaps', 'secrets'],
                    'verbs': ['create', 'get', 'list', 'watch']
                }
            ]
        }
        
        manifests['cluster_role_binding'] = {
            'apiVersion': 'rbac.authorization.k8s.io/v1',
            'kind': 'ClusterRoleBinding',
            'metadata': {
                'name': 'ml-scheduler-pipeline-runner'
            },
            'subjects': [{
                'kind': 'ServiceAccount',
                'name': 'ml-scheduler-pipeline-runner',
                'namespace': self.automation_config['kubeflow_namespace']
            }],
            'roleRef': {
                'kind': 'ClusterRole',
                'name': 'ml-scheduler-pipeline-runner',
                'apiGroup': 'rbac.authorization.k8s.io'
            }
        }
        
        # 3. ConfigMap for automation configuration
        manifests['automation_config'] = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': 'ml-scheduler-automation-config',
                'namespace': self.automation_config['kubeflow_namespace']
            },
            'data': {
                'scheduling_config.yaml': yaml.dump(self.create_scheduling_configuration()),
                'monitoring_config.yaml': yaml.dump(self.create_monitoring_configuration()),
                'alerting_config.yaml': yaml.dump(self.create_alerting_configuration())
            }
        }
        
        # 4. Monitoring ServiceMonitor for Prometheus
        manifests['service_monitor'] = {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'ServiceMonitor',
            'metadata': {
                'name': 'ml-scheduler-pipeline-metrics',
                'namespace': self.automation_config['monitoring_namespace'],
                'labels': {
                    'app': 'ml-scheduler',
                    'component': 'pipeline-monitoring'
                }
            },
            'spec': {
                'selector': {
                    'matchLabels': {
                        'app': 'ml-scheduler-pipeline'
                    }
                },
                'endpoints': [{
                    'port': 'metrics',
                    'interval': '30s',
                    'path': '/metrics'
                }]
            }
        }
        
        return manifests
    
    def create_prometheus_rules(self) -> Dict[str, Any]:
        """
        Create Prometheus alerting rules
        
        Returns:
            PrometheusRule manifest
        """
        prometheus_rule = {
            'apiVersion': 'monitoring.coreos.com/v1',
            'kind': 'PrometheusRule',
            'metadata': {
                'name': 'ml-scheduler-pipeline-alerts',
                'namespace': self.automation_config['monitoring_namespace'],
                'labels': {
                    'app': 'ml-scheduler',
                    'component': 'alerting'
                }
            },
            'spec': {
                'groups': [{
                    'name': 'ml-scheduler.pipeline',
                    'rules': [
                        {
                            'alert': 'MLSchedulerPipelineFailed',
                            'expr': 'ml_scheduler_pipeline_success_rate < 0.8',
                            'for': '5m',
                            'labels': {
                                'severity': 'high',
                                'component': 'ml-scheduler-pipeline'
                            },
                            'annotations': {
                                'summary': 'ML-Scheduler pipeline success rate is below 80%',
                                'description': 'Pipeline success rate has been below 80% for more than 5 minutes',
                                'runbook_url': 'https://wiki.company.com/ml-scheduler-troubleshooting'
                            }
                        },
                        {
                            'alert': 'MLSchedulerPipelineDurationHigh',
                            'expr': 'ml_scheduler_pipeline_duration_seconds > 7200',
                            'for': '1m',
                            'labels': {
                                'severity': 'medium',
                                'component': 'ml-scheduler-pipeline'
                            },
                            'annotations': {
                                'summary': 'ML-Scheduler pipeline execution time is too high',
                                'description': 'Pipeline execution is taking longer than 2 hours'
                            }
                        },
                        {
                            'alert': 'MLSchedulerModelPerformanceLow',
                            'expr': 'ml_scheduler_model_performance{metric_type="business_score"} < 75',
                            'for': '1m',
                            'labels': {
                                'severity': 'high',
                                'component': 'ml-scheduler-models'
                            },
                            'annotations': {
                                'summary': 'ML-Scheduler model performance is below threshold',
                                'description': 'Model business score is below 75 points'
                            }
                        }
                    ]
                }]
            }
        }
        
        return prometheus_rule
    
    def create_pipeline_trigger_script(self) -> str:
        """
        Create pipeline trigger script for automation
        
        Returns:
            Python script content
        """
        trigger_script = '''#!/usr/bin/env python3
"""
ML-Scheduler Pipeline Trigger Script
Automated pipeline execution for scheduled and event-driven runs
"""

import os
import sys
import json
import requests
import argparse
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineTrigger:
    """Kubeflow Pipeline trigger for ML-Scheduler"""
    
    def __init__(self, kubeflow_endpoint: str):
        self.kubeflow_endpoint = kubeflow_endpoint
        self.headers = {'Content-Type': 'application/json'}
    
    def trigger_pipeline(self, pipeline_name: str, parameters: dict = None):
        """Trigger pipeline execution"""
        
        # Default parameters if not provided
        if parameters is None:
            parameters = {
                'prometheus_endpoint': 'http://prometheus.monitoring.svc.cluster.local:9090',
                'collection_period_days': 30,
                'validation_split': 0.2,
                'kserve_namespace': 'ml-scheduler',
                'auto_deployment': True
            }
        
        # Create run request
        run_request = {
            'display_name': f'{pipeline_name}-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
            'description': f'Automated execution of {pipeline_name}',
            'pipeline_spec': {
                'pipeline_id': pipeline_name
            },
            'runtime_config': {
                'parameters': parameters
            }
        }
        
        try:
            # Submit pipeline run
            response = requests.post(
                f'{self.kubeflow_endpoint}/pipeline/v1/runs',
                json=run_request,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                run_info = response.json()
                logger.info(f"Pipeline triggered successfully: {run_info.get('run_id')}")
                return True
            else:
                logger.error(f"Failed to trigger pipeline: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering pipeline: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Trigger ML-Scheduler pipeline')
    parser.add_argument('--pipeline-name', required=True, help='Pipeline name to trigger')
    parser.add_argument('--kubeflow-endpoint', 
                       default=os.getenv('KUBEFLOW_ENDPOINT', 'http://localhost:8080'),
                       help='Kubeflow endpoint URL')
    
    args = parser.parse_args()
    
    # Initialize trigger
    trigger = PipelineTrigger(args.kubeflow_endpoint)
    
    # Trigger pipeline
    success = trigger.trigger_pipeline(args.pipeline_name)
    
    if success:
        logger.info("Pipeline triggered successfully")
        sys.exit(0)
    else:
        logger.error("Failed to trigger pipeline")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        return trigger_script
    
    def save_automation_files(self):
        """Save all automation configuration files"""
        
        # Create automation directory
        os.makedirs('./pipelines/automation', exist_ok=True)
        
        # 1. Save scheduling configuration
        scheduling_config = self.create_scheduling_configuration()
        with open('./pipelines/automation/scheduling_config.yaml', 'w') as f:
            yaml.dump(scheduling_config, f, default_flow_style=False)
        
        # 2. Save monitoring configuration
        monitoring_config = self.create_monitoring_configuration()
        with open('./pipelines/automation/monitoring_config.yaml', 'w') as f:
            yaml.dump(monitoring_config, f, default_flow_style=False)
        
        # 3. Save alerting configuration
        alerting_config = self.create_alerting_configuration()
        with open('./pipelines/automation/alerting_config.yaml', 'w') as f:
            yaml.dump(alerting_config, f, default_flow_style=False)
        
        # 4. Save Kubernetes manifests
        manifests = self.create_kubernetes_manifests()
        for manifest_name, manifest_content in manifests.items():
            with open(f'./pipelines/automation/{manifest_name}.yaml', 'w') as f:
                yaml.dump(manifest_content, f, default_flow_style=False)
        
        # 5. Save Prometheus rules
        prometheus_rule = self.create_prometheus_rules()
        with open('./pipelines/automation/prometheus_rules.yaml', 'w') as f:
            yaml.dump(prometheus_rule, f, default_flow_style=False)
        
        # 6. Save pipeline trigger script
        trigger_script = self.create_pipeline_trigger_script()
        with open('./pipelines/automation/trigger_pipeline.py', 'w') as f:
            f.write(trigger_script)
        
        # Make trigger script executable
        os.chmod('./pipelines/automation/trigger_pipeline.py', 0o755)
        
        return True


def main():
    """
    Create complete automation configuration
    """
    print("="*70)
    print("ML-SCHEDULER PIPELINE AUTOMATION - ÉTAPE 7 ACTION 4")
    print("Configuration exécution automatique et monitoring pipeline")
    print("="*70)
    
    # Initialize automation
    automation = MLSchedulerAutomation()
    
    print("\n🤖 AUTOMATION FEATURES:")
    print("  • Weekly scheduled re-training (Sundays 2 AM UTC)")
    print("  • Event-driven triggers (data threshold, performance, incidents)")
    print("  • Manual admin triggers with approval workflow")
    print("  • Resource-aware scheduling with priority classes")
    
    print("\n📊 MONITORING CAPABILITIES:")
    print("  • Pipeline execution duration and success rate")
    print("  • Individual component performance tracking") 
    print("  • Resource utilization monitoring (CPU, Memory, Storage)")
    print("  • Data quality metrics and drift detection")
    print("  • Model performance degradation alerts")
    
    print("\n🚨 ALERTING CHANNELS:")
    print("  • Slack notifications (#ml-scheduler-alerts)")
    print("  • Email alerts to ML and Platform teams")
    print("  • PagerDuty integration for critical issues")
    print("  • Multi-level escalation policies")
    
    print("\n⚙️  KUBERNETES INTEGRATION:")
    print("  • CronJob for weekly scheduling")
    print("  • ServiceAccount with proper RBAC permissions")
    print("  • ConfigMaps for configuration management")
    print("  • PrometheusRule for custom alerting")
    print("  • ServiceMonitor for metrics collection")
    
    # Save automation files
    automation.save_automation_files()
    
    print("\n📁 AUTOMATION FILES CREATED:")
    automation_files = [
        'scheduling_config.yaml - Scheduling and trigger configuration',
        'monitoring_config.yaml - Pipeline monitoring metrics',
        'alerting_config.yaml - Alert rules and channels',
        'weekly_cronjob.yaml - Kubernetes CronJob manifest',
        'service_account.yaml - RBAC configuration',
        'automation_config.yaml - ConfigMap with settings',
        'service_monitor.yaml - Prometheus monitoring',
        'prometheus_rules.yaml - Custom alerting rules',
        'trigger_pipeline.py - Pipeline trigger script'
    ]
    
    for file_desc in automation_files:
        print(f"  • {file_desc}")
    
    print("\n🗓️  SCHEDULING CONFIGURATION:")
    print("  • Weekly: Every Sunday at 2:00 AM UTC")
    print("  • Data trigger: >10,000 new samples")
    print("  • Performance trigger: Accuracy <80%")
    print("  • Incident trigger: >5 cluster incidents")
    print("  • Manual triggers: Admin and emergency")
    
    print("\n📈 MONITORING THRESHOLDS:")
    print("  • Pipeline duration: <2 hours")
    print("  • Success rate: ≥80%")
    print("  • CPU utilization: <90%")
    print("  • Memory usage: <85%")
    print("  • Data completeness: ≥95%")
    
    print("\n🔄 DEPLOYMENT COMMANDS:")
    print("  # Deploy automation infrastructure")
    print("  kubectl apply -f ./pipelines/automation/")
    print("  ")
    print("  # Manual pipeline trigger")
    print("  python ./pipelines/automation/trigger_pipeline.py \\")
    print("    --pipeline-name ml-scheduler-training-pipeline")
    
    print("\n" + "="*70)
    print("✅ ACTION 4 COMPLETED - Pipeline Automation Complete")
    print("Automation, scheduling, monitoring et alerting opérationnels")
    print("="*70)
    
    logger.info("ML-Scheduler Pipeline Automation configuration completed")
    return True


if __name__ == "__main__":
    main()