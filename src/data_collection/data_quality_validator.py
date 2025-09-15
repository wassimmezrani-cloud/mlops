#!/usr/bin/env python3
"""
ML-Scheduler Data Quality Validation
Calcul score qualite global pour validation Etape 3
Respect .claude_code_rules - Pas d'emojis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Validateur qualite donnees ML-Scheduler"""
    
    def __init__(self, data_path="./data/historical"):
        self.data_path = data_path
        self.nodes_path = f"{data_path}/nodes"
        self.pods_path = f"{data_path}/pods" 
        self.events_path = f"{data_path}/events"
        self.eda_path = f"{data_path}/eda"
        
    def validate_data_completeness(self) -> Dict:
        """Valider completude donnees"""
        logger.info("Validation completude donnees")
        
        completeness_score = 0
        max_score = 100
        results = {
            'completeness_score': 0,
            'nodes_data_present': False,
            'pods_data_present': False,
            'events_data_present': False,
            'period_coverage': 0,
            'missing_critical_metrics': []
        }
        
        # Verification donnees nodes (40 points)
        critical_node_metrics = [
            'node_cpu_seconds_total.csv',
            'node_memory_MemAvailable_bytes.csv',
            'node_memory_MemTotal_bytes.csv',
            'node_load1.csv',
            'node_load5.csv'
        ]
        
        nodes_present = 0
        for metric in critical_node_metrics:
            if os.path.exists(f"{self.nodes_path}/{metric}"):
                nodes_present += 1
            else:
                results['missing_critical_metrics'].append(metric)
        
        if nodes_present == len(critical_node_metrics):
            completeness_score += 40
            results['nodes_data_present'] = True
        else:
            completeness_score += int(40 * nodes_present / len(critical_node_metrics))
        
        # Verification donnees pods (30 points)
        critical_pod_metrics = [
            'container_cpu_usage_seconds_total.csv',
            'container_memory_usage_bytes.csv'
        ]
        
        pods_present = 0
        for metric in critical_pod_metrics:
            if os.path.exists(f"{self.pods_path}/{metric}"):
                pods_present += 1
            else:
                results['missing_critical_metrics'].append(metric)
        
        if pods_present == len(critical_pod_metrics):
            completeness_score += 30
            results['pods_data_present'] = True
        else:
            completeness_score += int(30 * pods_present / len(critical_pod_metrics))
        
        # Verification evenements (20 points)
        if os.path.exists(f"{self.events_path}/kubernetes_events.csv"):
            completeness_score += 20
            results['events_data_present'] = True
        
        # Verification periode temporelle (10 points)
        try:
            cpu_df = pd.read_csv(f"{self.nodes_path}/node_cpu_seconds_total.csv")
            cpu_df['timestamp'] = pd.to_datetime(cpu_df['timestamp'])
            period_days = (cpu_df['timestamp'].max() - cpu_df['timestamp'].min()).days
            
            if period_days >= 7:
                completeness_score += 10
                results['period_coverage'] = period_days
            else:
                completeness_score += int(10 * period_days / 7)
                results['period_coverage'] = period_days
        except:
            results['period_coverage'] = 0
        
        results['completeness_score'] = min(completeness_score, max_score)
        logger.info(f"Score completude: {results['completeness_score']}/100")
        
        return results
    
    def validate_data_consistency(self) -> Dict:
        """Valider coherence donnees"""
        logger.info("Validation coherence donnees")
        
        consistency_results = {
            'consistency_score': 0,
            'timestamp_alignment': False,
            'node_consistency': True,
            'value_ranges_valid': True,
            'anomalies_detected': []
        }
        
        consistency_score = 0
        
        try:
            # Verification alignement temporel (30 points)
            cpu_df = pd.read_csv(f"{self.nodes_path}/node_cpu_seconds_total.csv", nrows=1000)
            mem_df = pd.read_csv(f"{self.nodes_path}/node_memory_MemAvailable_bytes.csv", nrows=1000)
            
            cpu_df['timestamp'] = pd.to_datetime(cpu_df['timestamp'])
            mem_df['timestamp'] = pd.to_datetime(mem_df['timestamp'])
            
            # Verifier chevauchement temporel
            overlap_start = max(cpu_df['timestamp'].min(), mem_df['timestamp'].min())
            overlap_end = min(cpu_df['timestamp'].max(), mem_df['timestamp'].max())
            
            if overlap_end > overlap_start:
                consistency_score += 30
                consistency_results['timestamp_alignment'] = True
            
            # Verification coherence nodes (25 points)
            cpu_nodes = set(cpu_df['node'].unique())
            mem_nodes = set(mem_df['node'].unique())
            
            nodes_overlap = len(cpu_nodes.intersection(mem_nodes)) / len(cpu_nodes.union(mem_nodes))
            if nodes_overlap >= 0.8:
                consistency_score += 25
            else:
                consistency_results['node_consistency'] = False
                consistency_results['anomalies_detected'].append(f"Nodes mismatch: {nodes_overlap:.2f} overlap")
            
            # Verification plages valeurs (25 points)  
            mem_values_valid = (mem_df['value'] >= 0).all() and (mem_df['value'] <= 1e12).all()  # Max 1TB
            cpu_values_valid = (cpu_df['value'] >= 0).all() and (cpu_df['value'] <= 100).all()  # Max 100 CPU rate
            
            if mem_values_valid and cpu_values_valid:
                consistency_score += 25
            else:
                consistency_results['value_ranges_valid'] = False
                if not mem_values_valid:
                    consistency_results['anomalies_detected'].append("Memory values out of range")
                if not cpu_values_valid:
                    consistency_results['anomalies_detected'].append("CPU values out of range")
            
            # Verification continuite temporelle (20 points)
            cpu_time_gaps = cpu_df['timestamp'].diff().dt.total_seconds().describe()
            expected_interval = 60  # 1 minute
            
            if cpu_time_gaps['std'] < expected_interval * 2:  # Faible variation
                consistency_score += 20
            else:
                consistency_results['anomalies_detected'].append(f"Irregular time intervals: std={cpu_time_gaps['std']:.1f}s")
            
        except Exception as e:
            logger.error(f"Erreur validation coherence: {e}")
            consistency_results['anomalies_detected'].append(f"Validation error: {str(e)}")
        
        consistency_results['consistency_score'] = min(consistency_score, 100)
        logger.info(f"Score coherence: {consistency_results['consistency_score']}/100")
        
        return consistency_results
    
    def validate_ml_readiness(self) -> Dict:
        """Valider preparation pour ML"""
        logger.info("Validation ML readiness")
        
        ml_readiness = {
            'ml_readiness_score': 0,
            'feature_diversity': False,
            'temporal_patterns': False,
            'anomaly_detection_viable': False,
            'sufficient_volume': False
        }
        
        ml_score = 0
        
        try:
            # Diversite features (25 points)
            if os.path.exists(f"{self.nodes_path}/collection_index.json"):
                with open(f"{self.nodes_path}/collection_index.json", 'r') as f:
                    nodes_index = json.load(f)
                
                metrics_count = len(nodes_index.get('metrics_collected', []))
                if metrics_count >= 5:  # Au moins 5 metriques nodes
                    ml_score += 25
                    ml_readiness['feature_diversity'] = True
            
            # Patterns temporels identifies (25 points)
            if os.path.exists(f"{self.eda_path}/ml_patterns.json"):
                with open(f"{self.eda_path}/ml_patterns.json", 'r') as f:
                    patterns = json.load(f)
                
                if len(patterns) >= 3:
                    ml_score += 25
                    ml_readiness['temporal_patterns'] = True
            
            # Viabilite detection anomalies (25 points)
            load_df = pd.read_csv(f"{self.nodes_path}/node_load1.csv", nrows=1000)
            load_variance = load_df.groupby('node')['value'].var()
            
            if (load_variance > 0.1).any():  # Variance suffisante pour anomalies
                ml_score += 25
                ml_readiness['anomaly_detection_viable'] = True
            
            # Volume donnees suffisant (25 points)
            total_points = 0
            if os.path.exists(f"{self.nodes_path}/collection_index.json"):
                with open(f"{self.nodes_path}/collection_index.json", 'r') as f:
                    nodes_index = json.load(f)
                total_points += nodes_index.get('total_data_points', 0)
            
            if total_points >= 100000:  # Minimum 100k points pour ML
                ml_score += 25
                ml_readiness['sufficient_volume'] = True
            
        except Exception as e:
            logger.error(f"Erreur validation ML: {e}")
        
        ml_readiness['ml_readiness_score'] = min(ml_score, 100)
        logger.info(f"Score ML readiness: {ml_readiness['ml_readiness_score']}/100")
        
        return ml_readiness
    
    def calculate_global_quality_score(self) -> Dict:
        """Calculer score qualite global"""
        logger.info("Calcul score qualite global")
        
        # Validation completude
        completeness = self.validate_data_completeness()
        
        # Validation coherence  
        consistency = self.validate_data_consistency()
        
        # Validation ML readiness
        ml_readiness = self.validate_ml_readiness()
        
        # Calcul score global (moyenne ponderee)
        weights = {
            'completeness': 0.4,  # 40%
            'consistency': 0.3,   # 30%
            'ml_readiness': 0.3   # 30%
        }
        
        global_score = (
            completeness['completeness_score'] * weights['completeness'] +
            consistency['consistency_score'] * weights['consistency'] +
            ml_readiness['ml_readiness_score'] * weights['ml_readiness']
        )
        
        # Determination niveau qualite
        if global_score >= 90:
            quality_level = 'EXCELLENT'
            etape3_status = 'READY_FOR_ETAPE4'
        elif global_score >= 75:
            quality_level = 'GOOD'
            etape3_status = 'READY_FOR_ETAPE4'
        elif global_score >= 60:
            quality_level = 'ACCEPTABLE'
            etape3_status = 'CORRECTIONS_NEEDED'
        else:
            quality_level = 'INSUFFICIENT'
            etape3_status = 'MAJOR_ISSUES'
        
        # Compilation rapport final
        global_quality_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'global_quality_score': round(global_score, 1),
            'quality_level': quality_level,
            'etape3_status': etape3_status,
            'component_scores': {
                'data_completeness': completeness['completeness_score'],
                'data_consistency': consistency['consistency_score'],
                'ml_readiness': ml_readiness['ml_readiness_score']
            },
            'detailed_results': {
                'completeness': completeness,
                'consistency': consistency,
                'ml_readiness': ml_readiness
            },
            'recommendations': self._generate_recommendations(global_score, completeness, consistency, ml_readiness)
        }
        
        logger.info(f"Score qualite global: {global_score:.1f}/100 - {quality_level}")
        
        return global_quality_report
    
    def _generate_recommendations(self, global_score, completeness, consistency, ml_readiness) -> List[str]:
        """Generer recommendations amelioration"""
        recommendations = []
        
        if completeness['completeness_score'] < 80:
            recommendations.append("Completer collecte donnees - metriques critiques manquantes")
        
        if consistency['consistency_score'] < 80:
            recommendations.append("Verifier coherence temporelle et valeurs aberrantes")
        
        if ml_readiness['ml_readiness_score'] < 80:
            recommendations.append("Enrichir features et identifier plus de patterns temporels")
        
        if global_score >= 75:
            recommendations.append("Donnees ready - Passer Etape 4 XGBoost")
        else:
            recommendations.append("Corriger issues avant passage Etape 4")
        
        return recommendations
    
    def save_quality_report(self, quality_report):
        """Sauvegarder rapport qualite"""
        output_path = f"{self.data_path}/data_quality_global_report.json"
        
        with open(output_path, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        logger.info(f"Rapport qualite sauvegarde: {output_path}")
        return output_path

if __name__ == "__main__":
    validator = DataQualityValidator()
    
    # Calcul score qualite global
    quality_report = validator.calculate_global_quality_score()
    
    # Sauvegarde rapport
    report_path = validator.save_quality_report(quality_report)
    
    print("=" * 60)
    print("VALIDATION QUALITE DONNEES TERMINEE")
    print("=" * 60)
    print(f"Score Global: {quality_report['global_quality_score']}/100")
    print(f"Niveau Qualite: {quality_report['quality_level']}")
    print(f"Statut Etape 3: {quality_report['etape3_status']}")
    print()
    print("Scores Composants:")
    for component, score in quality_report['component_scores'].items():
        print(f"  {component}: {score}/100")
    print()
    print("Recommendations:")
    for rec in quality_report['recommendations']:
        print(f"  - {rec}")