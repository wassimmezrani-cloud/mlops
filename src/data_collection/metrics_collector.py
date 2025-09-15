#!/usr/bin/env python3
"""
ML-Scheduler Metrics Collection Module
Collecte complete metriques historiques pour algorithmes ML
Structure MLOps Lifecycle
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import yaml
import logging
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collecteur complet metriques Prometheus pour ML-Scheduler"""
    
    def __init__(self, config_path="configs/data_collection/collection_config.yaml"):
        """Initialisation avec configuration MLOps"""
        self.config = self._load_config(config_path)
        self.prometheus_url = self.config['prometheus']['url']
        self.api_url = f"{self.prometheus_url}/api/v1"
        self.session = requests.Session()
        self.session.timeout = self.config['prometheus']['timeout']
        
    def _load_config(self, config_path: str) -> Dict:
        """Charger configuration YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration chargee depuis {config_path}")
            return config
        except Exception as e:
            logger.error(f"Erreur chargement config: {e}")
            # Config par defaut si fichier inaccessible
            return {
                'prometheus': {'url': 'http://10.110.190.83:9090', 'timeout': 30},
                'collection': {
                    'default_period_days': 7, 
                    'step_interval': '1m',
                    'node_metrics': [
                        {'name': 'node_cpu_seconds_total', 'type': 'rate', 'critical': True},
                        {'name': 'node_memory_MemAvailable_bytes', 'type': 'instant', 'critical': True},
                        {'name': 'node_memory_MemTotal_bytes', 'type': 'instant', 'critical': True},
                        {'name': 'node_load1', 'type': 'instant', 'critical': True},
                        {'name': 'node_load5', 'type': 'instant', 'critical': True}
                    ],
                    'container_metrics': [
                        {'name': 'container_cpu_usage_seconds_total', 'type': 'rate', 'critical': True},
                        {'name': 'container_memory_usage_bytes', 'type': 'instant', 'critical': True}
                    ]
                },
                'storage': {'base_path': './data/historical', 'nodes_path': './data/historical/nodes', 'pods_path': './data/historical/pods'}
            }
    
    def collect_node_metrics(self, days_back: int = None) -> Dict[str, pd.DataFrame]:
        """Collecte metriques nodes systeme"""
        if days_back is None:
            days_back = self.config['collection'].get('default_period_days', 7)
            
        logger.info(f"Debut collecte metriques nodes pour {days_back} jours")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        node_metrics_data = {}
        node_metrics_config = self.config['collection']['node_metrics']
        
        for metric_config in node_metrics_config:
            metric_name = metric_config['name']
            metric_type = metric_config['type']
            is_critical = metric_config['critical']
            
            logger.info(f"Collecte {metric_name} (type: {metric_type})")
            
            # Construction requete selon type metrique
            if metric_type == 'rate':
                query = f"rate({metric_name}[5m])"
            else:
                query = metric_name
            
            params = {
                'query': query,
                'start': start_time.timestamp(),
                'end': end_time.timestamp(),
                'step': self.config['collection']['step_interval']
            }
            
            try:
                response = self.session.get(f"{self.api_url}/query_range", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data['status'] == 'success' and data['data']['result']:
                        metric_df = self._prometheus_to_dataframe(data, metric_name)
                        node_metrics_data[metric_name] = metric_df
                        logger.info(f"SUCCESS: {metric_name} - {len(metric_df)} points collectes")
                    else:
                        logger.warning(f"EMPTY: {metric_name} - Aucune donnee")
                        if is_critical:
                            logger.error(f"CRITICAL: {metric_name} requis pour ML")
                else:
                    logger.error(f"HTTP ERROR: {metric_name} - Status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"EXCEPTION: {metric_name} - {str(e)}")
        
        logger.info(f"Collecte nodes terminee: {len(node_metrics_data)} metriques")
        return node_metrics_data
    
    def collect_container_metrics(self, days_back: int = None) -> Dict[str, pd.DataFrame]:
        """Collecte metriques containers/pods"""
        if days_back is None:
            days_back = self.config['collection'].get('default_period_days', 7)
            
        logger.info(f"Debut collecte metriques containers pour {days_back} jours")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        container_metrics_data = {}
        container_metrics_config = self.config['collection']['container_metrics']
        
        for metric_config in container_metrics_config:
            metric_name = metric_config['name']
            metric_type = metric_config['type']
            
            logger.info(f"Collecte {metric_name}")
            
            # Construction requete selon type metrique
            if metric_type == 'rate':
                query = f"rate({metric_name}[5m])"
            else:
                query = metric_name
            
            params = {
                'query': query,
                'start': start_time.timestamp(), 
                'end': end_time.timestamp(),
                'step': self.config['collection']['step_interval']
            }
            
            try:
                response = self.session.get(f"{self.api_url}/query_range", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data['status'] == 'success' and data['data']['result']:
                        metric_df = self._prometheus_containers_to_dataframe(data, metric_name)
                        container_metrics_data[metric_name] = metric_df
                        logger.info(f"SUCCESS: {metric_name} - {len(metric_df)} points collectes")
                    else:
                        logger.warning(f"EMPTY: {metric_name} - Aucune donnee")
                else:
                    logger.error(f"HTTP ERROR: {metric_name} - Status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"EXCEPTION: {metric_name} - {str(e)}")
        
        logger.info(f"Collecte containers terminee: {len(container_metrics_data)} metriques")
        return container_metrics_data
    
    def _prometheus_to_dataframe(self, prometheus_data: Dict, metric_name: str) -> pd.DataFrame:
        """Conversion donnees Prometheus nodes vers DataFrame"""
        all_rows = []
        
        for result in prometheus_data['data']['result']:
            # Extraction metadata node
            metadata = result['metric']
            node_instance = metadata.get('instance', 'unknown')
            node_name = node_instance.split(':')[0] if ':' in node_instance else node_instance
            
            # Extraction series temporelles
            for timestamp, value in result['values']:
                try:
                    numeric_value = float(value) if value != 'NaN' else np.nan
                    all_rows.append({
                        'timestamp': datetime.fromtimestamp(float(timestamp)),
                        'node': node_name,
                        'metric': metric_name,
                        'value': numeric_value,
                        'metadata': metadata
                    })
                except ValueError:
                    continue
        
        return pd.DataFrame(all_rows)
    
    def _prometheus_containers_to_dataframe(self, prometheus_data: Dict, metric_name: str) -> pd.DataFrame:
        """Conversion donnees Prometheus containers vers DataFrame"""
        all_rows = []
        
        for result in prometheus_data['data']['result']:
            metadata = result['metric']
            
            # Extraction informations container/pod
            pod_name = metadata.get('pod', 'unknown')
            namespace = metadata.get('namespace', 'unknown')
            container_name = metadata.get('container', 'unknown')
            node_instance = metadata.get('instance', metadata.get('node', 'unknown'))
            node_name = node_instance.split(':')[0] if ':' in str(node_instance) else str(node_instance)
            
            # Extraction series temporelles
            for timestamp, value in result['values']:
                try:
                    numeric_value = float(value) if value != 'NaN' else np.nan
                    all_rows.append({
                        'timestamp': datetime.fromtimestamp(float(timestamp)),
                        'pod_name': pod_name,
                        'namespace': namespace,
                        'container': container_name,
                        'node': node_name,
                        'metric': metric_name,
                        'value': numeric_value,
                        'metadata': metadata
                    })
                except ValueError:
                    continue
        
        return pd.DataFrame(all_rows)
    
    def save_metrics_data(self, metrics_data: Dict[str, pd.DataFrame], data_type: str):
        """Sauvegarde donnees metriques de maniere organisee"""
        storage_config = self.config['storage']
        base_path = storage_config['base_path']
        
        if data_type == 'nodes':
            output_path = storage_config.get('nodes_path', f"{base_path}/nodes")
        elif data_type == 'containers':
            output_path = storage_config.get('pods_path', f"{base_path}/pods")
        else:
            output_path = f"{base_path}/{data_type}"
            
        os.makedirs(output_path, exist_ok=True)
        
        total_saved_points = 0
        saved_files = []
        
        for metric_name, dataframe in metrics_data.items():
            if dataframe is not None and not dataframe.empty:
                filename = f"{output_path}/{metric_name}.parquet"
                
                # Sauvegarde en CSV pour eviter dependance parquet  
                csv_filename = filename.replace('.parquet', '.csv')
                dataframe.to_csv(csv_filename, index=False)
                
                total_saved_points += len(dataframe)
                saved_files.append(csv_filename)
                logger.info(f"SAVED: {metric_name} - {len(dataframe)} lignes -> {csv_filename}")
        
        # Creation index collection
        collection_index = {
            'collection_timestamp': datetime.now().isoformat(),
            'data_type': data_type,
            'total_files': len(saved_files),
            'total_data_points': total_saved_points,
            'metrics_collected': list(metrics_data.keys()),
            'files_created': saved_files
        }
        
        index_filename = f"{output_path}/collection_index.json"
        import json
        with open(index_filename, 'w') as f:
            json.dump(collection_index, f, indent=2)
        
        logger.info(f"COLLECTION {data_type.upper()} TERMINEE:")
        logger.info(f"  - {len(saved_files)} fichiers crees")
        logger.info(f"  - {total_saved_points:,} points de donnees")
        logger.info(f"  - Index: {index_filename}")
        
        return collection_index

if __name__ == "__main__":
    # Collecte complete des metriques
    collector = MetricsCollector()
    
    # Collecte metriques nodes
    logger.info("DEBUT COLLECTE METRIQUES NODES")
    node_metrics = collector.collect_node_metrics(days_back=7)  # Utiliser periode disponible
    node_collection_index = collector.save_metrics_data(node_metrics, 'nodes')
    
    # Collecte metriques containers  
    logger.info("DEBUT COLLECTE METRIQUES CONTAINERS")
    container_metrics = collector.collect_container_metrics(days_back=7)
    container_collection_index = collector.save_metrics_data(container_metrics, 'containers')
    
    logger.info("COLLECTE COMPLETE TERMINEE")
    logger.info(f"Total donnees nodes: {node_collection_index['total_data_points']:,}")
    logger.info(f"Total donnees containers: {container_collection_index['total_data_points']:,}")