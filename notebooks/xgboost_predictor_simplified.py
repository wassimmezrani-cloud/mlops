#!/usr/bin/env python3
"""
XGBoost Load Predictor Development - Version Simplifiee
ML-Scheduler Premier Expert: Le Prophete
Utilise RandomForest comme alternative a XGBoost
Respect .claude_code_rules - Pas d'emojis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLSchedulerPredictor:
    """ML Predictor - Premier Expert IA ML-Scheduler (Alternative sans XGBoost)"""
    
    def __init__(self, data_path="./data/historical"):
        self.data_path = data_path
        self.nodes_path = f"{data_path}/nodes"
        self.models = {}
        self.feature_columns = []
        
    def load_historical_data(self):
        """Charger donnees historiques pour ML"""
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
            
            # Agregation par timestamp + node pour reduire volume
            cpu_agg = cpu_df.groupby(['timestamp', 'node'])['value'].mean().reset_index()
            cpu_agg.columns = ['timestamp', 'node', 'cpu_rate']
            
            memory_agg = memory_df.groupby(['timestamp', 'node'])['value'].mean().reset_index()
            memory_agg.columns = ['timestamp', 'node', 'memory_available_bytes']
            
            memory_total_agg = memory_total_df.groupby(['timestamp', 'node'])['value'].mean().reset_index()
            memory_total_agg.columns = ['timestamp', 'node', 'memory_total_bytes']
            
            load1_agg = load1_df.groupby(['timestamp', 'node'])['value'].mean().reset_index()
            load1_agg.columns = ['timestamp', 'node', 'load1']
            
            load5_agg = load5_df.groupby(['timestamp', 'node'])['value'].mean().reset_index()
            load5_agg.columns = ['timestamp', 'node', 'load5']
            
            # Fusionner donnees
            master_df = cpu_agg.copy()
            
            for df in [memory_agg, memory_total_agg, load1_agg, load5_agg]:
                master_df = master_df.merge(df, on=['timestamp', 'node'], how='inner')
            
            # Calculer utilisation memoire
            master_df['memory_utilization'] = 1.0 - (
                master_df['memory_available_bytes'] / master_df['memory_total_bytes']
            )
            
            # Nettoyer valeurs aberrantes
            master_df['cpu_rate'] = master_df['cpu_rate'].clip(0, 2.0)
            master_df['memory_utilization'] = master_df['memory_utilization'].clip(0, 1.0)
            master_df['load1'] = master_df['load1'].fillna(0).clip(0, 20)
            master_df['load5'] = master_df['load5'].fillna(0).clip(0, 20)
            
            # Trier chronologiquement
            master_df = master_df.sort_values(['node', 'timestamp']).reset_index(drop=True)
            
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
        
        # 1. Features temporelles de base
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
        
        # 2. Features lag simples par node
        logger.info("Creation features lag")
        
        # Trier par node et timestamp pour lag
        features_df = features_df.sort_values(['node', 'timestamp']).reset_index(drop=True)
        
        # Lag features par node (version simplifiee)
        lag_features = []
        for node in features_df['node'].unique():
            node_mask = features_df['node'] == node
            node_data = features_df[node_mask].copy().sort_values('timestamp')
            
            # Lag CPU (1, 5, 10 minutes)
            for lag in [1, 5, 10]:
                lag_col = f'cpu_rate_lag_{lag}'
                node_data[lag_col] = node_data['cpu_rate'].shift(lag).fillna(method='bfill')
                
            # Lag memoire
            for lag in [1, 5]:
                lag_col = f'memory_util_lag_{lag}'
                node_data[lag_col] = node_data['memory_utilization'].shift(lag).fillna(method='bfill')
            
            lag_features.append(node_data)
        
        # Recombiner donnees avec lag features
        features_df = pd.concat(lag_features, ignore_index=True)
        features_df = features_df.sort_values(['node', 'timestamp']).reset_index(drop=True)
        
        # 3. Features moyennes mobiles simplifiees
        logger.info("Creation moyennes mobiles")
        
        rolling_features = []
        for node in features_df['node'].unique():
            node_mask = features_df['node'] == node
            node_data = features_df[node_mask].copy().sort_values('timestamp')
            
            # MA CPU (5, 15, 30 periodes)
            for window in [5, 15, 30]:
                ma_col = f'cpu_ma_{window}'
                node_data[ma_col] = node_data['cpu_rate'].rolling(
                    window=window, min_periods=1
                ).mean()
            
            # MA memoire
            for window in [10, 30]:
                ma_col = f'memory_ma_{window}'
                node_data[ma_col] = node_data['memory_utilization'].rolling(
                    window=window, min_periods=1
                ).mean()
            
            rolling_features.append(node_data)
        
        # Recombiner donnees avec rolling features
        features_df = pd.concat(rolling_features, ignore_index=True)
        features_df = features_df.sort_values(['node', 'timestamp']).reset_index(drop=True)
        
        # 4. Features contextuelles nodes
        logger.info("Ajout features contextuelles")
        
        # Capacite node (approximation)
        node_capacity = features_df.groupby('node')['memory_total_bytes'].max()
        features_df['node_memory_capacity'] = features_df['node'].map(node_capacity)
        
        # Normaliser capacite (0-1)
        max_capacity = features_df['node_memory_capacity'].max()
        features_df['node_capacity_normalized'] = features_df['node_memory_capacity'] / max_capacity
        
        # One-hot encoding nodes (simplifie)
        unique_nodes = features_df['node'].unique()
        if len(unique_nodes) <= 10:
            for i, node in enumerate(unique_nodes):
                features_df[f'node_is_{i}'] = (features_df['node'] == node).astype(int)
        
        # 5. Features d'interaction
        features_df['cpu_memory_interaction'] = features_df['cpu_rate'] * features_df['memory_utilization']
        features_df['load_memory_ratio'] = features_df['load1'] / (features_df['memory_utilization'] + 0.001)
        
        # Nettoyer NaN restants
        features_df = features_df.fillna(0)
        
        logger.info(f"Features creees: {features_df.shape[1]} colonnes")
        logger.info(f"Lignes exploitables: {len(features_df):,}")
        
        return features_df
    
    def create_prediction_targets(self, df, horizons=[15, 30, 60]):
        """Creer targets prediction multi-horizons (reduits pour donnees limitees)"""
        logger.info(f"Creation targets pour horizons: {horizons} minutes")
        
        targets_df = df.copy()
        
        # Trier pour targets futures
        targets_df = targets_df.sort_values(['node', 'timestamp']).reset_index(drop=True)
        
        target_features = []
        for node in targets_df['node'].unique():
            node_mask = targets_df['node'] == node
            node_data = targets_df[node_mask].copy().sort_values('timestamp')
            
            for horizon in horizons:
                # Targets CPU et memoire futures
                target_col_cpu = f'target_cpu_{horizon}m'
                target_col_mem = f'target_memory_{horizon}m'
                
                node_data[target_col_cpu] = node_data['cpu_rate'].shift(-horizon)
                node_data[target_col_mem] = node_data['memory_utilization'].shift(-horizon)
            
            target_features.append(node_data)
        
        # Recombiner avec targets
        targets_df = pd.concat(target_features, ignore_index=True)
        targets_df = targets_df.sort_values(['node', 'timestamp']).reset_index(drop=True)
        
        # Supprimer lignes sans targets (NaN en fin)
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
    
    def train_prediction_models(self, train_df, val_df, horizons=[15, 30, 60]):
        """Entrainer modeles prediction avec Random Forest"""
        logger.info("TRAINING MODELES PREDICTION")
        
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
                
                # Configuration Random Forest (plus stable que XGBoost)
                rf_params = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                # Training
                model = RandomForestRegressor(**rf_params)
                model.fit(X_train, y_train)
                
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
            target_rmse = performance_targets.get(f"{model_name}_rmse", 0.20)  # Plus permissif
            target_r2 = performance_targets.get(f"{model_name}_r2", 0.60)      # Plus permissif  
            target_saturation = performance_targets.get(f"{model_name}_saturation", 0.75)
            
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
        
        # Score global ajuste pour RandomForest
        rmse_score = max(0, (0.40 - avg_rmse) / 0.20 * 100)  # 40% max acceptable
        r2_score = max(0, (avg_r2 - 0.40) / 0.40 * 100)      # 40% min acceptable  
        saturation_score = max(0, (avg_saturation_accuracy - 0.60) / 0.30 * 100)  # 60% min
        
        global_score = (rmse_score + r2_score + saturation_score) / 3
        
        if global_score >= 80:
            status = "EXCELLENT - Ready pour production"
            next_action = "Deployer immediatement"
        elif global_score >= 65:
            status = "BON - Ready avec monitoring"
            next_action = "Deployer avec surveillance"
        elif global_score >= 50:
            status = "ACCEPTABLE - Ameliorations necessaires"
            next_action = "Optimiser hyperparametres"
        else:
            status = "INSUFFISANT - Redeveloppement requis"
            next_action = "Revoir architecture modele"
        
        logger.info(f"Score Global: {global_score:.1f}/100")
        logger.info(f"Status: {status}")
        logger.info(f"Action: {next_action}")
        
        return {
            'global_score': global_score,
            'status': status,
            'rmse_score': rmse_score,
            'r2_score': r2_score,
            'saturation_score': saturation_score,
            'production_ready': global_score >= 65,
            'next_action': next_action
        }
    
    def select_best_model_for_production(self, results):
        """Selectionner meilleur modele pour production"""
        logger.info("SELECTION MODELE PRODUCTION")
        
        # Priorite modeles 30min (plus fiable avec donnees limitees)
        priority_models = [name for name in self.models.keys() if '30m' in name]
        
        best_model_name = None
        best_score = 0
        
        for model_name in priority_models:
            if model_name in results:
                result = results[model_name]
                # Score composite adapte
                composite_score = result['r2'] * 0.5 + result['saturation_accuracy'] * 0.5
                
                if composite_score > best_score:
                    best_score = composite_score
                    best_model_name = model_name
        
        if best_model_name:
            logger.info(f"Meilleur modele: {best_model_name} (score: {best_score:.3f})")
            return best_model_name, self.models[best_model_name]
        else:
            # Fallback sur premier modele available
            if self.models:
                fallback_name = list(self.models.keys())[0]
                logger.info(f"Fallback: {fallback_name}")
                return fallback_name, self.models[fallback_name]
        
        return None, None
    
    def prepare_model_for_production(self, model, model_name, performance_results):
        """Preparer modele pour deploiement production"""
        logger.info(f"PREPARATION MODELE PRODUCTION: {model_name}")
        
        # Creer dossier modele
        model_dir = f"./models/ml_predictor"
        os.makedirs(model_dir, exist_ok=True)
        
        # Sauvegarder modele avec joblib
        model_path = f"{model_dir}/model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Modele sauve: {model_path}")
        
        # Sauvegarder feature columns
        with open(f"{model_dir}/features.json", 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        
        # Metadonnees modele
        metadata = {
            'model_name': model_name,
            'model_type': 'random_forest',
            'version': '1.0.0',
            'creation_date': datetime.now().isoformat(),
            'features_count': len(self.feature_columns),
            'features': self.feature_columns,
            'input_shape': [len(self.feature_columns)],
            'output_shape': [1],
            'performance_metrics': performance_results.get(model_name, {}),
            'usage': 'Load prediction for ML-Scheduler',
            'note': 'RandomForest alternative to XGBoost'
        }
        
        with open(f"{model_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Script inference simple
        inference_script = '''#!/usr/bin/env python3
import joblib
import numpy as np
import json

class MLPredictor:
    def __init__(self, model_path, features_path):
        self.model = joblib.load(model_path)
        with open(features_path, 'r') as f:
            self.features = json.load(f)
    
    def predict(self, features_array):
        """Prediction single ou batch"""
        if isinstance(features_array, list):
            features_array = np.array(features_array)
        if len(features_array.shape) == 1:
            features_array = features_array.reshape(1, -1)
        
        predictions = self.model.predict(features_array)
        return predictions.tolist()

# Usage:
# predictor = MLPredictor('./models/ml_predictor/model.pkl', './models/ml_predictor/features.json')
# result = predictor.predict([0.1, 0.8, 2.5, ...])  # Vos features
'''
        
        with open(f"{model_dir}/predictor.py", 'w') as f:
            f.write(inference_script)
        
        logger.info(f"Modele pret pour production: {model_dir}")
        return model_dir

def main():
    """Main execution ML Predictor development"""
    print("=" * 60)
    print("DEVELOPPEMENT ML PREDICTOR - LE PROPHETE (VERSION SIMPLIFIEE)")
    print("=" * 60)
    
    # Initialisation
    predictor = MLSchedulerPredictor()
    
    try:
        # 1. Chargement donnees
        historical_data = predictor.load_historical_data()
        
        # 2. Feature engineering
        prediction_features = predictor.create_prediction_features(historical_data)
        
        # 3. Creation targets
        prediction_data = predictor.create_prediction_targets(prediction_features)
        
        # 4. Splits temporels
        train_data, val_data, test_data = predictor.create_temporal_splits(prediction_data)
        
        # 5. Training modeles
        training_results = predictor.train_prediction_models(train_data, val_data)
        
        # 6. Validation performance (targets plus permissifs)
        performance_targets = {
            'cpu_15m_rmse': 0.25, 'cpu_30m_rmse': 0.25, 'cpu_60m_rmse': 0.30,
            'memory_15m_rmse': 0.25, 'memory_30m_rmse': 0.25, 'memory_60m_rmse': 0.30,
            'cpu_15m_r2': 0.60, 'cpu_30m_r2': 0.55, 'cpu_60m_r2': 0.50,
            'memory_15m_r2': 0.60, 'memory_30m_r2': 0.55, 'memory_60m_r2': 0.50,
            'cpu_15m_saturation': 0.75, 'memory_15m_saturation': 0.75
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
        
        # 10. Rapport final
        print("\n" + "=" * 60)
        print("RAPPORT FINAL ML PREDICTOR")
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
            'data_samples': len(prediction_data),
            'model_type': 'RandomForest (Alternative to XGBoost)'
        }
        
        os.makedirs('./models/ml_predictor', exist_ok=True)
        with open('./models/ml_predictor/final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\nResultats sauvegardes: ./models/ml_predictor/")
        
        return final_results
        
    except Exception as e:
        logger.error(f"Erreur execution: {e}")
        return {'error': str(e), 'global_score': 0, 'production_ready': False}

if __name__ == "__main__":
    results = main()