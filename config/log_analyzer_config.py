"""
Fichier de configuration pour l'outil d'analyse automatisée de logs réseau
"""
from pydantic import BaseModel
from typing import Dict, List, Optional
import os

class LogAnalyzerConfig(BaseModel):
    """Configuration de l'analyseur de logs"""
    
    # Configuration de la base de données
    database_url: str = "sqlite:///data/logs.db"
    
    # Chemin des fichiers de logs
    log_directory: str = "data/logs"
    supported_formats: List[str] = ["apache", "nginx", "syslog", "firewall"]
    
    # Configuration de la détection d'anomalies
    anomaly_detection: Dict = {
        "algorithm": "isolation_forest",
        "contamination": 0.1,
        "window_size": 1000,
        "threshold": 0.5
    }
    
    # Configuration des alertes
    alert_settings: Dict = {
        "email_enabled": False,
        "email_recipients": [],
        "severity_threshold": "high",
        "cooldown_minutes": 60
    }
    
    # Configuration de la visualisation
    dashboard: Dict = {
        "refresh_interval": 30,  # secondes
        "max_display_logs": 10000,
        "default_time_range": "24h"
    }
    
    # Expressions régulières pour les formats de logs
    log_patterns: Dict[str, str] = {
        "apache": r'^(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" (?P<status>\d+) (?P<size>\S+)',
        "nginx": r'^(?P<ip>\S+) - \S+ \[(?P<timestamp>[^\]]+)\] "(?P<method>\S+) (?P<path>\S+) (?P<protocol>\S+)" (?P<status>\d+) (?P<size>\d+)',
        "syslog": r'^(?P<timestamp>\w+\s+\d+\s+\d+:\d+:\d+) (?P<hostname>\S+) (?P<process>\S+): (?P<message>.*)',
        "firewall": r'^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<action>\S+) (?P<src_ip>\S+) (?P<dst_ip>\S+) (?P<port>\d+)'
    }

# Créer une instance de configuration globale
config = LogAnalyzerConfig()

def get_config() -> LogAnalyzerConfig:
    """Obtenir l'instance de configuration"""
    return config

def update_config(**kwargs):
    """Mettre à jour la configuration"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value) 