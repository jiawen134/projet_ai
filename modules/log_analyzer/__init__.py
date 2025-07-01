"""
Module d'analyse automatisée de logs réseau
Fournit l'analyse de logs, la détection d'anomalies et les fonctionnalités de visualisation
"""

from .parser import LogParser, LogBatchProcessor
from .anomaly_detector import LogAnomalyDetector, RealTimeAnomalyDetector
from .dashboard import LogAnalyzerDashboard

__version__ = "1.0.0"
__author__ = "Équipe Projet IA"

__all__ = [
    'LogParser',
    'LogBatchProcessor', 
    'LogAnomalyDetector',
    'RealTimeAnomalyDetector',
    'LogAnalyzerDashboard'
] 