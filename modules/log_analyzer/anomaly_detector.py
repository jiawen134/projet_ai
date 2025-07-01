"""
Module de détection d'anomalies
Utilise des algorithmes d'apprentissage automatique pour détecter les modèles anormaux dans les logs réseau
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config.log_analyzer_config import get_config

class LogAnomalyDetector:
    """Détecteur d'anomalies de logs"""
    
    def __init__(self):
        self.config = get_config()
        self.anomaly_config = self.config.anomaly_detection
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_columns = []
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Préparer les caractéristiques pour la détection d'anomalies
        
        Args:
            df: DataFrame de logs bruts
            
        Returns:
            DataFrame de caractéristiques
        """
        if df.empty:
            return pd.DataFrame()
        
        features_df = df.copy()
        
        # Extraire les caractéristiques temporelles
        if 'parsed_timestamp' in df.columns:
            features_df['hour'] = pd.to_datetime(df['parsed_timestamp']).dt.hour
            features_df['day_of_week'] = pd.to_datetime(df['parsed_timestamp']).dt.dayofweek
            features_df['day_of_month'] = pd.to_datetime(df['parsed_timestamp']).dt.day
        
        # Traiter les caractéristiques d'adresse IP
        if 'ip' in df.columns:
            # Fréquence IP
            ip_counts = df['ip'].value_counts()
            features_df['ip_frequency'] = df['ip'].map(ip_counts)
            
            # IP privée ou non
            features_df['is_private_ip'] = df['ip'].apply(self._is_private_ip)
        
        # Traiter les codes de statut HTTP
        if 'status' in df.columns:
            features_df['status_numeric'] = pd.to_numeric(df['status'], errors='coerce')
            features_df['is_error_status'] = (features_df['status_numeric'] >= 400).astype(int)
        
        # Traiter la taille des requêtes
        if 'size' in df.columns:
            features_df['size_numeric'] = pd.to_numeric(df['size'], errors='coerce').fillna(0)
            features_df['log_size'] = np.log1p(features_df['size_numeric'])
        
        # Traiter les caractéristiques de chemin URL
        if 'path' in df.columns:
            features_df['path_length'] = df['path'].str.len()
            features_df['has_query_params'] = df['path'].str.contains('\\?').astype(int)
            features_df['path_depth'] = df['path'].str.count('/')
        
        # Traiter la chaîne user agent (si présente)
        if 'user_agent' in df.columns:
            features_df['user_agent_length'] = df['user_agent'].str.len()
            features_df['is_bot'] = df['user_agent'].str.contains(
                'bot|crawler|spider', case=False, na=False
            ).astype(int)
        
        # Encoder les caractéristiques catégorielles
        categorical_columns = ['method', 'protocol', 'log_format']
        for col in categorical_columns:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        features_df[col].fillna('unknown')
                    )
                else:
                    # Traiter les nouvelles valeurs catégorielles
                    known_classes = set(self.label_encoders[col].classes_)
                    features_df[col] = features_df[col].fillna('unknown')
                    unknown_mask = ~features_df[col].isin(known_classes)
                    features_df.loc[unknown_mask, col] = 'unknown'
                    
                    features_df[f'{col}_encoded'] = self.label_encoders[col].transform(features_df[col])
        
        # Sélectionner les caractéristiques numériques
        numeric_features = [
            'hour', 'day_of_week', 'day_of_month', 'ip_frequency', 'is_private_ip',
            'status_numeric', 'is_error_status', 'size_numeric', 'log_size',
            'path_length', 'has_query_params', 'path_depth'
        ]
        
        # Ajouter les caractéristiques catégorielles encodées
        for col in categorical_columns:
            if f'{col}_encoded' in features_df.columns:
                numeric_features.append(f'{col}_encoded')
        
        # Filtrer les caractéristiques disponibles
        available_features = [col for col in numeric_features if col in features_df.columns]
        
        if not available_features:
            raise ValueError("Aucune caractéristique numérique disponible pour la détection d'anomalies")
        
        # Remplir les valeurs manquantes
        features_df[available_features] = features_df[available_features].fillna(0)
        
        self.feature_columns = available_features
        return features_df[available_features]
    
    def _is_private_ip(self, ip: str) -> int:
        """
        Vérifier si l'IP est une adresse IP privée
        
        Args:
            ip: Chaîne d'adresse IP
            
        Returns:
            1 pour IP privée, 0 pour IP publique
        """
        try:
            parts = [int(x) for x in ip.split('.')]
            if len(parts) != 4:
                return 0
            
            # Plages d'IP privées
            if (parts[0] == 10 or
                (parts[0] == 172 and 16 <= parts[1] <= 31) or
                (parts[0] == 192 and parts[1] == 168)):
                return 1
            return 0
        except:
            return 0
    
    def train_anomaly_detector(self, df: pd.DataFrame) -> Dict:
        """
        训练异常检测模型
        
        Args:
            df: 训练数据DataFrame
            
        Returns:
            训练结果信息
        """
        features_df = self.prepare_features(df)
        
        if features_df.empty:
            return {'status': 'error', 'message': '没有可用的特征数据'}
        
        # 标准化特征
        features_scaled = self.scaler.fit_transform(features_df)
        
        # 创建并训练模型
        contamination = self.anomaly_config['contamination']
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # 训练模型
        self.model.fit(features_scaled)
        
        # 预测训练数据的异常
        anomaly_scores = self.model.decision_function(features_scaled)
        predictions = self.model.predict(features_scaled)
        
        # 统计信息
        num_anomalies = np.sum(predictions == -1)
        anomaly_rate = num_anomalies / len(predictions)
        
        return {
            'status': 'success',
            'total_samples': len(predictions),
            'anomalies_detected': num_anomalies,
            'anomaly_rate': anomaly_rate,
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns
        }
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测日志中的异常
        
        Args:
            df: 待检测的日志DataFrame
            
        Returns:
            包含异常检测结果的DataFrame
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_anomaly_detector")
        
        features_df = self.prepare_features(df)
        
        if features_df.empty:
            result_df = df.copy()
            result_df['is_anomaly'] = False
            result_df['anomaly_score'] = 0.0
            return result_df
        
        # 标准化特征
        features_scaled = self.scaler.transform(features_df)
        
        # 预测异常
        anomaly_scores = self.model.decision_function(features_scaled)
        predictions = self.model.predict(features_scaled)
        
        # 添加结果到原DataFrame
        result_df = df.copy()
        result_df['is_anomaly'] = predictions == -1
        result_df['anomaly_score'] = anomaly_scores
        
        return result_df
    
    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        """
        获取异常检测摘要
        
        Args:
            df: 包含异常检测结果的DataFrame
            
        Returns:
            异常摘要信息
        """
        if 'is_anomaly' not in df.columns:
            return {'status': 'error', 'message': '缺少异常检测结果'}
        
        anomalies = df[df['is_anomaly']]
        
        summary = {
            'total_logs': len(df),
            'anomaly_count': len(anomalies),
            'anomaly_rate': len(anomalies) / len(df) if len(df) > 0 else 0,
            'top_anomaly_ips': [],
            'anomaly_time_distribution': {},
            'anomaly_status_codes': {}
        }
        
        if not anomalies.empty:
            # 异常IP统计
            if 'ip' in anomalies.columns:
                top_ips = anomalies['ip'].value_counts().head(10)
                summary['top_anomaly_ips'] = top_ips.to_dict()
            
            # 异常时间分布
            if 'parsed_timestamp' in anomalies.columns:
                anomalies_with_time = anomalies.dropna(subset=['parsed_timestamp'])
                if not anomalies_with_time.empty:
                    hour_dist = pd.to_datetime(anomalies_with_time['parsed_timestamp']).dt.hour.value_counts()
                    summary['anomaly_time_distribution'] = hour_dist.to_dict()
            
            # 异常状态码
            if 'status' in anomalies.columns:
                status_dist = anomalies['status'].value_counts()
                summary['anomaly_status_codes'] = status_dist.to_dict()
        
        return summary

class RealTimeAnomalyDetector:
    """实时异常检测器"""
    
    def __init__(self, window_size: int = 1000):
        self.detector = LogAnomalyDetector()
        self.window_size = window_size
        self.log_buffer = []
        self.is_trained = False
        
    def add_log_entry(self, log_entry: Dict) -> Optional[Dict]:
        """
        添加新的日志条目并检测异常
        
        Args:
            log_entry: 日志条目字典
            
        Returns:
            如果检测到异常，返回异常信息，否则返回None
        """
        self.log_buffer.append(log_entry)
        
        # 保持缓冲区大小
        if len(self.log_buffer) > self.window_size:
            self.log_buffer.pop(0)
        
        # 如果缓冲区达到一定大小且尚未训练，则训练模型
        if len(self.log_buffer) >= self.window_size and not self.is_trained:
            df = pd.DataFrame(self.log_buffer)
            result = self.detector.train_anomaly_detector(df)
            if result['status'] == 'success':
                self.is_trained = True
        
        # 如果模型已训练，检测当前日志条目
        if self.is_trained:
            df = pd.DataFrame([log_entry])
            result_df = self.detector.detect_anomalies(df)
            
            if result_df.iloc[0]['is_anomaly']:
                return {
                    'timestamp': log_entry.get('parsed_timestamp'),
                    'ip': log_entry.get('ip'),
                    'anomaly_score': result_df.iloc[0]['anomaly_score'],
                    'log_entry': log_entry
                }
        
        return None 