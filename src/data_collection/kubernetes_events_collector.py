#!/usr/bin/env python3
"""
ML-Scheduler Kubernetes Events Collection
Collecte evenements systeme pour algorithmes ML
Respect .claude_code_rules - Pas d'emojis
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KubernetesEventsCollector:
    """Collecteur evenements Kubernetes pour ML-Scheduler"""
    
    def __init__(self, prometheus_url="http://10.110.190.83:9090"):
        self.prometheus_url = prometheus_url
        self.api_url = f"{prometheus_url}/api/v1"
        self.session = requests.Session()
        self.session.timeout = 30
        
    def collect_pod_events_from_prometheus(self, days_back=7):
        """Collecter evenements pods via metriques Prometheus"""
        logger.info(f"Collecte evenements pods pour {days_back} jours")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        events_data = []
        
        # Requetes evenements critiques via metriques Prometheus
        event_queries = [
            {
                'query': 'kube_pod_status_phase',
                'event_type': 'pod_phase_change',
                'description': 'Changements phase pods'
            },
            {
                'query': 'kube_pod_container_status_restarts_total',
                'event_type': 'container_restart',
                'description': 'Restarts containers'
            },
            {
                'query': 'rate(kube_pod_container_status_restarts_total[5m])',
                'event_type': 'restart_rate',
                'description': 'Taux restarts containers'
            }
        ]
        
        for event_query in event_queries:
            logger.info(f"Collecte {event_query['event_type']}")
            
            params = {
                'query': event_query['query'],
                'start': start_time.timestamp(),
                'end': end_time.timestamp(),
                'step': '1m'
            }
            
            try:
                response = self.session.get(f"{self.api_url}/query_range", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data['status'] == 'success' and data['data']['result']:
                        for result in data['data']['result']:
                            metadata = result['metric']
                            
                            for timestamp, value in result['values']:
                                try:
                                    events_data.append({
                                        'timestamp': datetime.fromtimestamp(float(timestamp)),
                                        'event_type': event_query['event_type'],
                                        'pod_name': metadata.get('pod', 'unknown'),
                                        'namespace': metadata.get('namespace', 'unknown'),
                                        'node': metadata.get('node', metadata.get('instance', 'unknown')),
                                        'container': metadata.get('container', 'unknown'),
                                        'phase': metadata.get('phase', 'unknown'),
                                        'value': float(value) if value != 'NaN' else 0,
                                        'source': 'prometheus_metrics',
                                        'severity': self._determine_severity(event_query['event_type'], float(value) if value != 'NaN' else 0)
                                    })
                                except ValueError:
                                    continue
                        
                        logger.info(f"SUCCESS: {event_query['event_type']} - {len([r for r in data['data']['result']])} series")
                    else:
                        logger.warning(f"EMPTY: {event_query['event_type']} - Aucune donnee")
                        
                else:
                    logger.error(f"HTTP ERROR: {event_query['event_type']} - Status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"EXCEPTION: {event_query['event_type']} - {str(e)}")
        
        events_df = pd.DataFrame(events_data)
        logger.info(f"Total evenements collectes: {len(events_df)}")
        
        return events_df
    
    def _determine_severity(self, event_type, value):
        """Determiner severite evenement"""
        if event_type == 'container_restart' and value > 0:
            return 'WARNING'
        elif event_type == 'restart_rate' and value > 0.1:
            return 'CRITICAL'
        elif event_type == 'pod_phase_change':
            return 'INFO'
        else:
            return 'NORMAL'
    
    def collect_node_status_events(self, days_back=7):
        """Collecter evenements statut nodes"""
        logger.info("Collecte evenements nodes")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        node_events = []
        
        # Requetes statut nodes
        node_queries = [
            'up{job="node-exporter"}',
            'node_boot_time_seconds',
            'node_load1 > 5',  # Charge elevee
            '(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9'  # Memoire critique
        ]
        
        for query in node_queries:
            params = {
                'query': query,
                'start': start_time.timestamp(),
                'end': end_time.timestamp(),
                'step': '5m'
            }
            
            try:
                response = self.session.get(f"{self.api_url}/query_range", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data['status'] == 'success' and data['data']['result']:
                        for result in data['data']['result']:
                            metadata = result['metric']
                            instance = metadata.get('instance', 'unknown')
                            node = instance.split(':')[0] if ':' in instance else instance
                            
                            for timestamp, value in result['values']:
                                node_events.append({
                                    'timestamp': datetime.fromtimestamp(float(timestamp)),
                                    'event_type': self._classify_node_event(query, float(value) if value != 'NaN' else 0),
                                    'node': node,
                                    'value': float(value) if value != 'NaN' else 0,
                                    'query': query,
                                    'source': 'node_metrics'
                                })
                            
            except Exception as e:
                logger.error(f"Erreur requete node: {e}")
        
        return pd.DataFrame(node_events)
    
    def _classify_node_event(self, query, value):
        """Classifier evenement node selon requete"""
        if 'up{' in query:
            return 'node_availability' if value > 0 else 'node_down'
        elif 'boot_time' in query:
            return 'node_boot'
        elif 'load1 >' in query:
            return 'high_load'
        elif 'memory' in query and value > 0.9:
            return 'memory_critical'
        else:
            return 'node_status'
    
    def generate_scheduling_insights(self, events_df):
        """Generer insights pour scheduler ML"""
        logger.info("Generation insights scheduling")
        
        insights = {
            'timestamp': datetime.now().isoformat(),
            'period_analyzed_days': 7,
            'total_events': len(events_df),
            'event_categories': {},
            'scheduling_patterns': [],
            'node_reliability': {},
            'pod_stability_metrics': {}
        }
        
        if not events_df.empty:
            # Categories evenements
            insights['event_categories'] = events_df['event_type'].value_counts().to_dict()
            
            # Patterns de scheduling
            restart_events = events_df[events_df['event_type'] == 'container_restart']
            if not restart_events.empty:
                problematic_nodes = restart_events['node'].value_counts().head(3).to_dict()
                insights['scheduling_patterns'].append({
                    'pattern': 'nodes_with_restarts',
                    'description': f'Nodes avec plus de restarts: {list(problematic_nodes.keys())}',
                    'ml_relevance': 'Q-Learning - Eviter ces nodes'
                })
            
            # Fiabilite nodes
            node_events = events_df.groupby('node').agg({
                'event_type': 'count',
                'severity': lambda x: (x == 'CRITICAL').sum()
            }).reset_index()
            
            for _, row in node_events.iterrows():
                insights['node_reliability'][row['node']] = {
                    'total_events': int(row['event_type']),
                    'critical_events': int(row['severity']),
                    'reliability_score': max(0, 100 - int(row['severity']) * 10)
                }
            
            # Stabilite pods par namespace
            if 'namespace' in events_df.columns:
                ns_stability = events_df[events_df['event_type'] == 'container_restart'].groupby('namespace').size()
                insights['pod_stability_metrics'] = ns_stability.to_dict()
        
        return insights
    
    def save_events_data(self, events_df, insights, output_path="./data/historical"):
        """Sauvegarder donnees evenements"""
        events_path = f"{output_path}/events"
        os.makedirs(events_path, exist_ok=True)
        
        # Sauvegarde evenements
        if not events_df.empty:
            events_df.to_csv(f"{events_path}/kubernetes_events.csv", index=False)
            logger.info(f"Evenements sauvegardes: {len(events_df)} lignes")
        
        # Sauvegarde insights
        with open(f"{events_path}/scheduling_insights.json", 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        # Index collection
        collection_index = {
            'collection_timestamp': datetime.now().isoformat(),
            'total_events': len(events_df),
            'insights_generated': len(insights.get('scheduling_patterns', [])),
            'files_created': [
                f"{events_path}/kubernetes_events.csv",
                f"{events_path}/scheduling_insights.json"
            ]
        }
        
        with open(f"{events_path}/events_collection_index.json", 'w') as f:
            json.dump(collection_index, f, indent=2)
        
        logger.info("Collection evenements terminee")
        return collection_index

if __name__ == "__main__":
    collector = KubernetesEventsCollector()
    
    # Collecte evenements pods
    pod_events = collector.collect_pod_events_from_prometheus(days_back=7)
    
    # Collecte evenements nodes
    node_events = collector.collect_node_status_events(days_back=7)
    
    # Fusion donnees
    all_events = pd.concat([pod_events, node_events], ignore_index=True)
    
    # Generation insights
    insights = collector.generate_scheduling_insights(all_events)
    
    # Sauvegarde
    collection_index = collector.save_events_data(all_events, insights)
    
    print("=" * 60)
    print("COLLECTE EVENEMENTS KUBERNETES TERMINEE")
    print("=" * 60)
    print(f"Total evenements: {collection_index['total_events']}")
    print(f"Insights generes: {collection_index['insights_generated']}")
    print("Fichiers crees:", collection_index['files_created'])