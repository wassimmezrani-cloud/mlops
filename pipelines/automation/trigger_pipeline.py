#!/usr/bin/env python3
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
