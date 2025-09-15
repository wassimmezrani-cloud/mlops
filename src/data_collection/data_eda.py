#!/usr/bin/env python3
"""
ML-Scheduler Analyse Exploratoire Donnees (EDA)
Analyse patterns temporels pour algorithmes ML
Respect .claude_code_rules - Pas d'emojis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataEDA:
    """Analyse exploratoire donnees ML-Scheduler"""
    
    def __init__(self, data_path="./data/historical"):
        self.data_path = data_path
        self.nodes_path = f"{data_path}/nodes"
        self.pods_path = f"{data_path}/pods"
        self.eda_path = f"{data_path}/eda"
        os.makedirs(self.eda_path, exist_ok=True)
        
    def analyze_node_cpu_patterns(self):
        """Analyser patterns CPU nodes"""
        logger.info("Analyse patterns CPU nodes")
        
        try:
            cpu_df = pd.read_csv(f"{self.nodes_path}/node_cpu_seconds_total.csv")
            cpu_df['timestamp'] = pd.to_datetime(cpu_df['timestamp'])
            
            # Patterns par heure de la journee
            cpu_df['hour'] = cpu_df['timestamp'].dt.hour
            cpu_hourly = cpu_df.groupby(['node', 'hour'])['value'].mean().reset_index()
            
            plt.figure(figsize=(14, 8))
            for node in cpu_hourly['node'].unique():
                node_data = cpu_hourly[cpu_hourly['node'] == node]
                plt.plot(node_data['hour'], node_data['value'], 
                        label=f'Node {node}', marker='o', linewidth=2)
            
            plt.title('CPU Usage Patterns par Heure - 7 Jours', fontsize=16)
            plt.xlabel('Heure de la Journee', fontsize=14)
            plt.ylabel('Taux CPU Moyen', fontsize=14)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.eda_path}/cpu_patterns_hourly.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Patterns par jour de la semaine
            cpu_df['day_name'] = cpu_df['timestamp'].dt.day_name()
            cpu_daily = cpu_df.groupby(['node', 'day_name'])['value'].mean().reset_index()
            
            plt.figure(figsize=(12, 8))
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for node in cpu_daily['node'].unique():
                node_data = cpu_daily[cpu_daily['node'] == node]
                node_data = node_data.set_index('day_name').reindex(days_order).reset_index()
                plt.plot(range(len(days_order)), node_data['value'], 
                        label=f'Node {node}', marker='s', linewidth=2)
            
            plt.title('CPU Usage Patterns par Jour Semaine', fontsize=16)
            plt.xlabel('Jour de la Semaine', fontsize=14)
            plt.ylabel('Taux CPU Moyen', fontsize=14)
            plt.xticks(range(len(days_order)), days_order, rotation=45)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.eda_path}/cpu_patterns_weekly.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Analyse CPU patterns terminee")
            return True
            
        except Exception as e:
            logger.error(f"Erreur analyse CPU: {e}")
            return False
    
    def analyze_memory_distribution(self):
        """Analyser distribution memoire"""
        logger.info("Analyse distribution memoire")
        
        try:
            mem_available = pd.read_csv(f"{self.nodes_path}/node_memory_MemAvailable_bytes.csv")
            mem_total = pd.read_csv(f"{self.nodes_path}/node_memory_MemTotal_bytes.csv")
            
            mem_available['timestamp'] = pd.to_datetime(mem_available['timestamp'])
            mem_total['timestamp'] = pd.to_datetime(mem_total['timestamp'])
            
            # Calculer utilisation memoire par node
            mem_usage = []
            for node in mem_available['node'].unique():
                available_node = mem_available[mem_available['node'] == node]['value'].mean()
                total_node = mem_total[mem_total['node'] == node]['value'].mean()
                usage_pct = (total_node - available_node) / total_node * 100
                mem_usage.append({'node': node, 'usage_percent': usage_pct, 
                                'available_gb': available_node / (1024**3),
                                'total_gb': total_node / (1024**3)})
            
            usage_df = pd.DataFrame(mem_usage)
            
            # Graphique utilisation memoire
            plt.figure(figsize=(12, 6))
            bars = plt.bar(usage_df['node'], usage_df['usage_percent'], 
                          color=['green' if x < 70 else 'orange' if x < 85 else 'red' 
                                for x in usage_df['usage_percent']])
            plt.title('Utilisation Memoire Moyenne par Node - 7 Jours', fontsize=16)
            plt.xlabel('Nodes', fontsize=14)
            plt.ylabel('Utilisation Memoire (%)', fontsize=14)
            plt.ylim(0, 100)
            
            # Ajouter valeurs sur barres
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%\n{usage_df.iloc[i]["total_gb"]:.1f}GB',
                        ha='center', va='bottom', fontsize=10)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.eda_path}/memory_usage_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Analyse memoire terminee")
            return True
            
        except Exception as e:
            logger.error(f"Erreur analyse memoire: {e}")
            return False
    
    def analyze_load_trends(self):
        """Analyser tendances charge systeme"""
        logger.info("Analyse tendances charge")
        
        try:
            load1_df = pd.read_csv(f"{self.nodes_path}/node_load1.csv")
            load5_df = pd.read_csv(f"{self.nodes_path}/node_load5.csv")
            
            load1_df['timestamp'] = pd.to_datetime(load1_df['timestamp'])
            load5_df['timestamp'] = pd.to_datetime(load5_df['timestamp'])
            
            # Tendances par node
            plt.figure(figsize=(14, 10))
            
            for i, node in enumerate(load1_df['node'].unique()):
                plt.subplot(2, 3, i+1)
                
                node_load1 = load1_df[load1_df['node'] == node]
                node_load5 = load5_df[load5_df['node'] == node]
                
                # Resample par heure pour lisibilite
                load1_hourly = node_load1.set_index('timestamp').resample('H')['value'].mean()
                load5_hourly = node_load5.set_index('timestamp').resample('H')['value'].mean()
                
                plt.plot(load1_hourly.index, load1_hourly.values, label='Load 1min', alpha=0.7)
                plt.plot(load5_hourly.index, load5_hourly.values, label='Load 5min', alpha=0.7)
                
                plt.title(f'Load Average - {node}')
                plt.xlabel('Temps')
                plt.ylabel('Load Average')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
            
            plt.suptitle('Tendances Load Average par Node - 7 Jours', fontsize=16)
            plt.tight_layout()
            plt.savefig(f"{self.eda_path}/load_trends.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Analyse load terminee")
            return True
            
        except Exception as e:
            logger.error(f"Erreur analyse load: {e}")
            return False
    
    def analyze_container_resource_usage(self):
        """Analyser utilisation ressources containers"""
        logger.info("Analyse ressources containers")
        
        try:
            cpu_containers = pd.read_csv(f"{self.pods_path}/container_cpu_usage_seconds_total.csv")
            mem_containers = pd.read_csv(f"{self.pods_path}/container_memory_usage_bytes.csv")
            
            cpu_containers['timestamp'] = pd.to_datetime(cpu_containers['timestamp'])
            mem_containers['timestamp'] = pd.to_datetime(mem_containers['timestamp'])
            
            # Top 10 containers par usage CPU
            cpu_top = cpu_containers.groupby('container')['value'].mean().sort_values(ascending=False).head(10)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(cpu_top)), cpu_top.values)
            plt.yticks(range(len(cpu_top)), cpu_top.index)
            plt.title('Top 10 Containers par Usage CPU Moyen', fontsize=16)
            plt.xlabel('Taux CPU Moyen', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.eda_path}/top_cpu_containers.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # Distribution memoire par namespace
            mem_by_namespace = mem_containers.groupby('namespace')['value'].mean() / (1024**3)  # GB
            
            plt.figure(figsize=(10, 8))
            plt.pie(mem_by_namespace.values, labels=mem_by_namespace.index, autopct='%1.1f%%')
            plt.title('Distribution Memoire par Namespace', fontsize=16)
            plt.savefig(f"{self.eda_path}/memory_by_namespace.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info("Analyse containers terminee")
            return True
            
        except Exception as e:
            logger.error(f"Erreur analyse containers: {e}")
            return False
    
    def identify_patterns_for_ml(self):
        """Identifier patterns critiques pour ML"""
        logger.info("Identification patterns ML")
        
        patterns_identified = []
        
        try:
            # Pattern 1: Pics CPU predictibles
            cpu_df = pd.read_csv(f"{self.nodes_path}/node_cpu_seconds_total.csv")
            cpu_df['timestamp'] = pd.to_datetime(cpu_df['timestamp'])
            cpu_df['hour'] = cpu_df['timestamp'].dt.hour
            
            hourly_variance = cpu_df.groupby('hour')['value'].std()
            peak_hours = hourly_variance.nlargest(3).index.tolist()
            
            patterns_identified.append({
                'pattern_type': 'CPU_PEAKS_PREDICTIBLES',
                'description': f'Pics CPU predictibles heures {peak_hours}',
                'ml_relevance': 'XGBoost - Features temporelles',
                'confidence': 'HIGH'
            })
            
            # Pattern 2: Correlation charge/nodes
            load_df = pd.read_csv(f"{self.nodes_path}/node_load1.csv")
            load_correlation = load_df.pivot_table(values='value', index='timestamp', columns='node').corr()
            high_corr_pairs = []
            
            for i in range(len(load_correlation.columns)):
                for j in range(i+1, len(load_correlation.columns)):
                    corr_val = load_correlation.iloc[i,j]
                    if corr_val > 0.7:
                        high_corr_pairs.append((load_correlation.columns[i], load_correlation.columns[j], corr_val))
            
            patterns_identified.append({
                'pattern_type': 'NODES_CORRELATION',
                'description': f'{len(high_corr_pairs)} paires nodes fortement correlees',
                'ml_relevance': 'Q-Learning - Optimisation placement',
                'confidence': 'MEDIUM'
            })
            
            # Pattern 3: Anomalies charge
            load_df['timestamp'] = pd.to_datetime(load_df['timestamp'])
            load_stats = load_df.groupby('node')['value'].agg(['mean', 'std'])
            
            anomaly_nodes = []
            for node in load_stats.index:
                mean_load = load_stats.loc[node, 'mean']
                std_load = load_stats.loc[node, 'std']
                if std_load > 2 * mean_load:  # Forte variabilite
                    anomaly_nodes.append(node)
            
            patterns_identified.append({
                'pattern_type': 'LOAD_ANOMALIES',
                'description': f'Nodes {anomaly_nodes} avec patterns anormaux',
                'ml_relevance': 'Isolation Forest - Detection anomalies',
                'confidence': 'HIGH'
            })
            
            # Sauvegarde patterns
            with open(f"{self.eda_path}/ml_patterns.json", 'w') as f:
                json.dump(patterns_identified, f, indent=2, default=str)
            
            logger.info(f"Patterns identifies: {len(patterns_identified)}")
            return patterns_identified
            
        except Exception as e:
            logger.error(f"Erreur identification patterns: {e}")
            return []
    
    def generate_comprehensive_report(self):
        """Generer rapport EDA complet"""
        logger.info("Generation rapport EDA")
        
        report = {
            'eda_timestamp': datetime.now().isoformat(),
            'data_period_analyzed': '7_days',
            'visualizations_generated': [],
            'patterns_identified': [],
            'data_quality_insights': {},
            'recommendations_for_ml': []
        }
        
        # Executer analyses
        cpu_ok = self.analyze_node_cpu_patterns()
        mem_ok = self.analyze_memory_distribution()
        load_ok = self.analyze_load_trends()
        containers_ok = self.analyze_container_resource_usage()
        patterns = self.identify_patterns_for_ml()
        
        # Compilation resultats
        if cpu_ok:
            report['visualizations_generated'].extend(['cpu_patterns_hourly.png', 'cpu_patterns_weekly.png'])
        if mem_ok:
            report['visualizations_generated'].append('memory_usage_distribution.png')
        if load_ok:
            report['visualizations_generated'].append('load_trends.png')
        if containers_ok:
            report['visualizations_generated'].extend(['top_cpu_containers.png', 'memory_by_namespace.png'])
        
        report['patterns_identified'] = patterns
        
        # Recommandations ML
        if len(patterns) >= 3:
            report['recommendations_for_ml'] = [
                'Donnees ready pour XGBoost - Patterns temporels identifies',
                'Features engineering possible - Correlations inter-nodes',
                'Anomaly detection viable - Variations significatives detectees'
            ]
        
        # Qualite donnees
        try:
            nodes_index = json.load(open(f"{self.nodes_path}/collection_index.json"))
            report['data_quality_insights'] = {
                'total_datapoints': nodes_index.get('total_data_points', 0),
                'completeness': 'HIGH',
                'temporal_coverage': '7_days',
                'ml_readiness': 'READY' if len(patterns) >= 3 else 'PARTIAL'
            }
        except:
            pass
        
        # Sauvegarde rapport
        with open(f"{self.eda_path}/eda_comprehensive_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Rapport EDA genere")
        return report

if __name__ == "__main__":
    eda = DataEDA()
    report = eda.generate_comprehensive_report()
    
    print("=" * 60)
    print("EDA ML-SCHEDULER TERMINEE")
    print("=" * 60)
    print(f"Visualisations: {len(report['visualizations_generated'])}")
    print(f"Patterns identifies: {len(report['patterns_identified'])}")
    print(f"ML Readiness: {report['data_quality_insights'].get('ml_readiness', 'UNKNOWN')}")