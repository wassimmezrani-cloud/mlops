#!/usr/bin/env python3
"""
XGBoost Load Predictor Development - Expert IA 1/3
ML-Scheduler Premier Expert: Le Prophete
Respect .claude_code_rules - Pas d'emojis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
import json
import os
import logging
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostPredictor:
    """XGBoost Predictor - Premier Expert IA ML-Scheduler"""
    
    def __init__(self, data_path="./data/historical"):
        self.data_path = data_path
        self.nodes_path = f"{data_path}/nodes"
        self.models = {}
        self.feature_columns = []
        self.scaler = StandardScaler()
        
    def load_historical_data(self):
        """Charger donnees historiques pour XGBoost"""
        logger.info("Chargement donnees historiques")
        
        try:
            # Charger donnees nodes critiques
            cpu_df = pd.read_csv(f"{self.nodes_path}/node_cpu_seconds_total.csv")
            memory_df = pd.read_csv(f"{self.nodes_path}/node_memory_MemAvailable_bytes.csv")
            memory_total_df = pd.read_csv(f"{self.nodes_path}/node_memory_MemTotal_bytes.csv")
            load1_df = pd.read_csv(f"{self.nodes_path}/node_load1.csv")
            load5_df = pd.read_csv(f"{self.nodes_path}/node_load5.csv")
            
            # Convertir timestamps
            for df in [cpu_df, memory_df, memory_total_df, load1_df, load5_df]:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"CPU data: {len(cpu_df):,} points sur {cpu_df['node'].nunique()} nodes")
            logger.info(f"Memory data: {len(memory_df):,} points")
            logger.info(f"Load data: {len(load1_df):,} + {len(load5_df):,} points")
            
            # Fusionner donnees par timestamp + node
            master_df = cpu_df[['timestamp', 'node', 'value']].copy()
            master_df.columns = ['timestamp', 'node', 'cpu_rate']
            
            # Ajouter memoire disponible
            memory_pivot = memory_df.groupby(['timestamp', 'node'])['value'].mean().reset_index()
            memory_pivot.columns = ['timestamp', 'node', 'memory_available_bytes']
            master_df = master_df.merge(memory_pivot, on=['timestamp', 'node'], how='inner')
            
            # Ajouter memoire totale
            memory_total_pivot = memory_total_df.groupby(['timestamp', 'node'])['value'].mean().reset_index()
            memory_total_pivot.columns = ['timestamp', 'node', 'memory_total_bytes']
            master_df = master_df.merge(memory_total_pivot, on=['timestamp', 'node'], how='inner')
            
            # Calculer utilisation memoire
            master_df['memory_utilization'] = 1.0 - (master_df['memory_available_bytes'] / master_df['memory_total_bytes'])
            
            # Ajouter load averages
            for load_df, col_name in [(load1_df, 'load1'), (load5_df, 'load5')]:
                load_pivot = load_df.groupby(['timestamp', 'node'])['value'].mean().reset_index()
                load_pivot.columns = ['timestamp', 'node', col_name]
                master_df = master_df.merge(load_pivot, on=['timestamp', 'node'], how='left')
            
            # Trier chronologiquement
            master_df = master_df.sort_values(['node', 'timestamp']).reset_index(drop=True)
            
            # Nettoyer valeurs aberrantes
            master_df['cpu_rate'] = master_df['cpu_rate'].clip(0, 2.0)  # CPU max 200%
            master_df['memory_utilization'] = master_df['memory_utilization'].clip(0, 1.0)  # Memory max 100%
            master_df['load1'] = master_df['load1'].fillna(0).clip(0, 20)  # Load max 20
            master_df['load5'] = master_df['load5'].fillna(0).clip(0, 20)
            
            logger.info(f"Dataset fusionne: {len(master_df):,} lignes, {master_df['node'].nunique()} nodes")
            logger.info(f"Periode: {master_df['timestamp'].min()} -> {master_df['timestamp'].max()}")
            
            return master_df
            
        except Exception as e:
            logger.error(f"Erreur chargement donnees: {e}")
            raise
    
    def create_prediction_features(self, df):
        """Creer features specialisees prediction temporelle"""
        logger.info("Creation features predictives")
        
        features_df = df.copy()
        
        # 1. Features temporelles
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['day_of_month'] = features_df['timestamp'].dt.day
        features_df['hour_of_week'] = features_df['day_of_week'] * 24 + features_df['hour']
        
        # Business hours et weekend
        features_df['is_business_hours'] = (
            (features_df['hour'] >= 9) & (features_df['hour'] <= 17) &
            (features_df['day_of_week'] < 5)
        ).astype(int)
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        
        # 2. Features lag par node
        logger.info("Creation features lag")
        for node in features_df['node'].unique():
            node_mask = features_df['node'] == node
            node_data = features_df[node_mask].copy()
            
            # Lag CPU
            for lag in [1, 3, 6, 12]:
                lag_col = f'cpu_rate_lag_{lag}'
                features_df.loc[node_mask, lag_col] = node_data['cpu_rate'].shift(lag)
            
            # Lag memoire
            for lag in [1, 6, 12]:
                lag_col = f'memory_util_lag_{lag}'
                features_df.loc[node_mask, lag_col] = node_data['memory_utilization'].shift(lag)
        
        # 3. Moyennes mobiles
        logger.info("Creation moyennes mobiles")
        for node in features_df['node'].unique():
            node_mask = features_df['node'] == node
            node_data = features_df[node_mask].copy()
            
            # MA CPU
            for window in [5, 15, 30, 60]:
                ma_col = f'cpu_ma_{window}'
                features_df.loc[node_mask, ma_col] = node_data['cpu_rate'].rolling(
                    window=window, min_periods=1
                ).mean()
            
            # MA memoire
            for window in [15, 30, 60]:
                ma_col = f'memory_ma_{window}'
                features_df.loc[node_mask, ma_col] = node_data['memory_utilization'].rolling(
                    window=window, min_periods=1
                ).mean()
        
        # 4. Features tendances
        logger.info("Creation features tendances")
        for node in features_df['node'].unique():
            node_mask = features_df['node'] == node
            node_data = features_df[node_mask].copy()
            
            # Slope CPU derniere heure
            def calculate_slope(x):
                if len(x) < 2:
                    return 0
                try:
                    return np.polyfit(range(len(x)), x, 1)[0]
                except:
                    return 0
            
            features_df.loc[node_mask, 'cpu_slope_1h'] = node_data['cpu_rate'].rolling(
                window=60, min_periods=5
            ).apply(calculate_slope)
            
            # Volatilite CPU
            features_df.loc[node_mask, 'cpu_volatility_30m'] = node_data['cpu_rate'].rolling(
                window=30, min_periods=5
            ).std().fillna(0)
        
        # 5. Features contextuelles nodes
        logger.info("Ajout features contextuelles")
        
        # Capacite node
        node_capacity = features_df.groupby('node')['memory_total_bytes'].max()
        features_df['node_memory_capacity'] = features_df['node'].map(node_capacity)
        
        # Rang capacite
        capacity_ranks = node_capacity.rank(ascending=False)
        features_df['node_capacity_rank'] = features_df['node'].map(capacity_ranks)
        
        # One-hot encoding nodes
        if features_df['node'].nunique() <= 10:
            node_dummies = pd.get_dummies(features_df['node'], prefix='node')
            features_df = pd.concat([features_df, node_dummies], axis=1)
        
        # Nettoyer NaN
        features_df = features_df.dropna()
        
        logger.info(f"Features creees: {features_df.shape[1]} colonnes")
        logger.info(f"Apres nettoyage: {len(features_df):,} lignes exploitables")
        
        return features_df
    
    def create_prediction_targets(self, df, horizons=[30, 60, 120]):
        """Creer targets prediction multi-horizons"""
        logger.info(f"Creation targets pour horizons: {horizons} minutes")
        
        targets_df = df.copy()
        
        for horizon in horizons:
            logger.info(f"Traitement horizon {horizon} minutes")
            
            for node in targets_df['node'].unique():
                node_mask = targets_df['node'] == node
                node_data = targets_df[node_mask].copy()
                
                # Targets CPU et memoire futures
                target_col_cpu = f'target_cpu_{horizon}m'
                target_col_mem = f'target_memory_{horizon}m'
                
                targets_df.loc[node_mask, target_col_cpu] = node_data['cpu_rate'].shift(-horizon)
                targets_df.loc[node_mask, target_col_mem] = node_data['memory_utilization'].shift(-horizon)
        
        # Supprimer lignes sans targets
        targets_df = targets_df.dropna()
        
        logger.info(f"Targets crees: {targets_df.shape[1]} colonnes, {len(targets_df):,} samples")
        
        return targets_df
    
    def create_temporal_splits(self, df, train_ratio=0.7, val_ratio=0.15):
        """Creer splits temporels pour eviter data leakage"""
        logger.info("Creation splits temporels")
        
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)
        
        n_total = len(df_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_df = df_sorted[:n_train].copy()
        val_df = df_sorted[n_train:n_train+n_val].copy()
        test_df = df_sorted[n_train+n_val:].copy()
        
        logger.info(f"Train: {len(train_df):,} samples")
        logger.info(f"Validation: {len(val_df):,} samples")
        logger.info(f"Test: {len(test_df):,} samples")
        
        return train_df, val_df, test_df
    
    def train_xgboost_models(self, train_df, val_df, horizons=[30, 60, 120]):
        """Entrainer modeles XGBoost pour chaque horizon"""
        logger.info("TRAINING MODELES XGBOOST")
        
        # Definir features colonnes
        exclude_cols = ['timestamp', 'node', 'cpu_rate', 'memory_utilization', 
                       'memory_available_bytes', 'memory_total_bytes']
        target_cols = [col for col in train_df.columns if col.startswith('target_')]
        exclude_cols.extend(target_cols)
        
        self.feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        logger.info(f"Features utilisees: {len(self.feature_columns)} colonnes")
        
        X_train = train_df[self.feature_columns]
        X_val = val_df[self.feature_columns]
        
        results = {}
        
        for horizon in horizons:
            for metric in ['cpu', 'memory']:
                target_col = f'target_{metric}_{horizon}m'
                
                if target_col not in train_df.columns:
                    continue
                
                model_name = f"{metric}_{horizon}m"
                logger.info(f"Training {model_name}")
                
                y_train = train_df[target_col]
                y_val = val_df[target_col]
                
                # Configuration XGBoost
                xgb_params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 200,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'tree_method': 'hist'
                }
                
                # Training
                model = xgb.XGBRegressor(**xgb_params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=20,
                    verbose=False
                )
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_val = model.predict(X_val)
                
                # Metriques
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
                val_r2 = r2_score(y_val, y_pred_val)
                val_mae = mean_absolute_error(y_val, y_pred_val)
                
                # Stocker modele et resultats
                self.models[model_name] = model
                results[model_name] = {
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'val_mae': val_mae,
                    'features_count': len(self.feature_columns)
                }
                
                logger.info(f"  {model_name}: Val RMSE={val_rmse:.4f}, R2={val_r2:.4f}")
        
        return results
    
    def validate_model_performance(self, test_df, performance_targets):
        """Validation rigoureuse performance modeles"""
        logger.info("VALIDATION PERFORMANCE MODELES")
        
        X_test = test_df[self.feature_columns]
        validation_results = {}
        
        for model_name, model in self.models.items():
            target_col = f"target_{model_name}"
            
            if target_col not in test_df.columns:
                continue
            
            logger.info(f"Validation {model_name}")
            
            y_test = test_df[target_col]
            y_pred = model.predict(X_test)
            
            # Metriques ML
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # MAPE (business metric)
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 0.001))) * 100
            
            # Detection saturation
            saturation_threshold = 0.8 if 'cpu' in model_name else 0.85
            
            y_saturated_true = (y_test > saturation_threshold).astype(int)
            y_saturated_pred = (y_pred > saturation_threshold).astype(int)
            
            saturation_accuracy = (y_saturated_true == y_saturated_pred).mean()
            
            try:
                saturation_precision = precision_score(y_saturated_true, y_saturated_pred, zero_division=0)
                saturation_recall = recall_score(y_saturated_true, y_saturated_pred, zero_division=0)
            except:
                saturation_precision = 0
                saturation_recall = 0
            
            validation_results[model_name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'saturation_accuracy': saturation_accuracy,
                'saturation_precision': saturation_precision,
                'saturation_recall': saturation_recall
            }
            
            # Verification targets
            target_rmse = performance_targets.get(f"{model_name}_rmse", 0.15)
            target_r2 = performance_targets.get(f"{model_name}_r2", 0.75)
            target_saturation = performance_targets.get(f"{model_name}_saturation", 0.85)
            
            rmse_ok = rmse <= target_rmse
            r2_ok = r2 >= target_r2
            saturation_ok = saturation_accuracy >= target_saturation
            
            status = "PASS" if (rmse_ok and r2_ok and saturation_ok) else "FAIL"
            
            logger.info(f"  RMSE: {rmse:.4f} (target <={target_rmse}) {'OK' if rmse_ok else 'FAIL'}")
            logger.info(f"  R2: {r2:.4f} (target >={target_r2}) {'OK' if r2_ok else 'FAIL'}")
            logger.info(f"  Saturation Acc: {saturation_accuracy:.4f} {'OK' if saturation_ok else 'FAIL'}")
            logger.info(f"  STATUS: {status}")
        
        return validation_results
    
    def analyze_business_impact(self, results):
        """Analyser impact business predictions"""
        logger.info("ANALYSE IMPACT BUSINESS")
        
        all_rmse = [r['rmse'] for r in results.values()]
        all_r2 = [r['r2'] for r in results.values()]
        all_saturation = [r['saturation_accuracy'] for r in results.values()]
        
        avg_rmse = np.mean(all_rmse)
        avg_r2 = np.mean(all_r2)
        avg_saturation_accuracy = np.mean(all_saturation)
        
        # Score global (0-100)
        rmse_score = max(0, (0.30 - avg_rmse) / 0.15 * 100)
        r2_score = max(0, (avg_r2 - 0.50) / 0.35 * 100)
        saturation_score = max(0, (avg_saturation_accuracy - 0.70) / 0.20 * 100)
        
        global_score = (rmse_score + r2_score + saturation_score) / 3
        
        if global_score >= 85:
            status = "EXCELLENT - Ready pour production"
            next_action = "Deployer immediatement en KServe"
        elif global_score >= 75:
            status = "BON - Ready avec monitoring renforce"
            next_action = "Deployer avec surveillance active"
        elif global_score >= 60:
            status = "ACCEPTABLE - Necessites ameliorations"
            next_action = "Retraining avec plus de donnees"
        else:
            status = "INSUFFISANT - Redeveloppement requis"
            next_action = "Revoir feature engineering"
        
        logger.info(f"Score Global: {global_score:.1f}/100")
        logger.info(f"Status: {status}")
        logger.info(f"Action: {next_action}")
        
        return {
            'global_score': global_score,
            'status': status,
            'rmse_score': rmse_score,
            'r2_score': r2_score,
            'saturation_score': saturation_score,
            'production_ready': global_score >= 75,
            'next_action': next_action
        }
    
    def select_best_model_for_production(self, results):
        """Selectionner meilleur modele pour production"""
        logger.info("SELECTION MODELE PRODUCTION")
        
        # Priorite modeles 60min
        priority_models = [name for name in self.models.keys() if '60m' in name]
        
        best_model_name = None
        best_score = 0
        
        for model_name in priority_models:
            if model_name in results:
                result = results[model_name]
                composite_score = result['r2'] * 0.6 + result['saturation_accuracy'] * 0.4
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_model_name = model_name
        
        if best_model_name:
            logger.info(f"Meilleur modele: {best_model_name} (score: {best_score:.3f})")
            return best_model_name, self.models[best_model_name]
        else:
            # Fallback
            fallback_name = 'cpu_60m'
            if fallback_name in self.models:
                logger.info(f"Fallback: {fallback_name}")
                return fallback_name, self.models[fallback_name]
        
        return None, None
    
    def prepare_model_for_production(self, model, model_name, performance_results):
        """Preparer modele pour deploiement production"""
        logger.info(f"PREPARATION MODELE PRODUCTION: {model_name}")
        
        # Creer dossier modele
        model_dir = f"./models/xgboost_predictor"
        os.makedirs(model_dir, exist_ok=True)
        
        # Sauvegarder modele XGBoost
        model_path = f"{model_dir}/model.bst"
        model.save_model(model_path)
        logger.info(f"Modele sauve: {model_path}")
        
        # Sauvegarder feature columns
        with open(f"{model_dir}/features.json", 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        
        # Metadonnees modele
        metadata = {
            'model_name': model_name,
            'model_type': 'xgboost',
            'version': '1.0.0',
            'creation_date': datetime.now().isoformat(),
            'features_count': len(self.feature_columns),
            'features': self.feature_columns,
            'input_shape': [len(self.feature_columns)],
            'output_shape': [1],
            'performance_metrics': performance_results.get(model_name, {}),
            'usage': 'Load prediction for ML-Scheduler'
        }
        
        with open(f"{model_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Modele pret pour production: {model_dir}")
        return model_dir
    
    def create_kserve_deployment_config(self, model_name):
        """Creer configuration deploiement KServe"""
        logger.info("Creation configuration KServe")
        
        kserve_yaml = f'''apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: xgboost-load-predictor
  namespace: ml-scheduler
  annotations:
    serving.kserve.io/deploymentMode: "Serverless"
spec:
  predictor:
    xgboost:
      storageUri: "pvc://ml-models-storage/xgboost_predictor"
      resources:
        requests:
          cpu: 500m
          memory: 1Gi
        limits:
          cpu: 2
          memory: 4Gi
      env:
      - name: STORAGE_URI
        value: "pvc://ml-models-storage/xgboost_predictor"
    minReplicas: 1
    maxReplicas: 5
    scaleTarget: 10
    scaleMetric: concurrency
'''
        
        # Sauvegarder YAML
        yaml_path = './models/xgboost_predictor/kserve_deployment.yaml'
        with open(yaml_path, 'w') as f:
            f.write(kserve_yaml)
        
        logger.info(f"Configuration KServe creee: {yaml_path}")
        return yaml_path

def main():
    """Main execution XGBoost development"""
    print("=" * 60)
    print("DEVELOPPEMENT XGBOOST PREDICTOR - LE PROPHETE")
    print("=" * 60)
    
    # Initialisation
    predictor = XGBoostPredictor()
    
    # 1. Chargement donnees
    historical_data = predictor.load_historical_data()
    
    # 2. Feature engineering
    prediction_features = predictor.create_prediction_features(historical_data)
    
    # 3. Creation targets
    prediction_data = predictor.create_prediction_targets(prediction_features)
    
    # 4. Splits temporels
    train_data, val_data, test_data = predictor.create_temporal_splits(prediction_data)
    
    # 5. Training modeles
    training_results = predictor.train_xgboost_models(train_data, val_data)
    
    # 6. Validation performance
    performance_targets = {
        'cpu_30m_rmse': 0.15, 'cpu_60m_rmse': 0.15, 'cpu_120m_rmse': 0.18,
        'memory_30m_rmse': 0.15, 'memory_60m_rmse': 0.15, 'memory_120m_rmse': 0.18,
        'cpu_30m_r2': 0.80, 'cpu_60m_r2': 0.75, 'cpu_120m_r2': 0.70,
        'memory_30m_r2': 0.75, 'memory_60m_r2': 0.75, 'memory_120m_r2': 0.70,
        'cpu_30m_saturation': 0.85, 'memory_30m_saturation': 0.85
    }
    
    validation_results = predictor.validate_model_performance(test_data, performance_targets)
    
    # 7. Analyse business impact
    business_analysis = predictor.analyze_business_impact(validation_results)
    
    # 8. Selection meilleur modele
    best_model_name, best_model = predictor.select_best_model_for_production(validation_results)
    
    # 9. Preparation production
    if best_model:
        model_dir = predictor.prepare_model_for_production(
            best_model, best_model_name, validation_results
        )
        kserve_config = predictor.create_kserve_deployment_config(best_model_name)
    
    # 10. Rapport final
    print("\n" + "=" * 60)
    print("RAPPORT FINAL XGBOOST PREDICTOR")
    print("=" * 60)
    print(f"Score Global: {business_analysis['global_score']:.1f}/100")
    print(f"Status: {business_analysis['status']}")
    print(f"Production Ready: {business_analysis['production_ready']}")
    print(f"Meilleur Modele: {best_model_name}")
    print(f"Features Count: {len(predictor.feature_columns)}")
    print()
    
    print("Performance par Modele:")
    for model_name, result in validation_results.items():
        print(f"  {model_name}:")
        print(f"    RMSE: {result['rmse']:.4f}")
        print(f"    R2: {result['r2']:.4f}")
        print(f"    Saturation Acc: {result['saturation_accuracy']:.4f}")
    
    # Sauvegarde resultats
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'business_analysis': business_analysis,
        'validation_results': validation_results,
        'training_results': training_results,
        'best_model': best_model_name,
        'features_count': len(predictor.feature_columns),
        'data_samples': len(prediction_data)
    }
    
    with open('./models/xgboost_predictor/final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nResultats sauvegardes: ./models/xgboost_predictor/")
    
    return final_results

if __name__ == "__main__":
    results = main()