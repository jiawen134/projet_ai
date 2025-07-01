"""
Module d'analyseur de logs
Supporte l'analyse des logs Apache, Nginx, syslog et firewall
"""
import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Generator, Tuple
from pathlib import Path
import chardet
from config.log_analyzer_config import get_config

class LogParser:
    """Classe d'analyseur de logs"""
    
    def __init__(self):
        self.config = get_config()
        self.patterns = self.config.log_patterns
        
    def detect_log_format(self, log_line: str) -> Optional[str]:
        """
        Détection automatique du format de log
        
        Args:
            log_line: Ligne de log
            
        Returns:
            Format de log détecté, ou None si non reconnu
        """
        for format_name, pattern in self.patterns.items():
            if re.match(pattern, log_line.strip()):
                return format_name
        return None
    
    def parse_log_line(self, log_line: str, log_format: str) -> Optional[Dict]:
        """
        Analyser une ligne de log
        
        Args:
            log_line: Ligne de log
            log_format: Format de log
            
        Returns:
            Dictionnaire analysé contenant les champs extraits
        """
        if log_format not in self.patterns:
            return None
            
        pattern = self.patterns[log_format]
        match = re.match(pattern, log_line.strip())
        
        if not match:
            return None
            
        result = match.groupdict()
        
        # Normaliser le timestamp
        if 'timestamp' in result:
            result['parsed_timestamp'] = self._parse_timestamp(
                result['timestamp'], log_format
            )
        
        # Ajouter la ligne brute et l'information de format
        result['raw_line'] = log_line.strip()
        result['log_format'] = log_format
        
        return result
    
    def _parse_timestamp(self, timestamp_str: str, log_format: str) -> Optional[datetime]:
        """
        Analyser une chaîne de timestamp
        
        Args:
            timestamp_str: Chaîne de timestamp
            log_format: Format de log
            
        Returns:
            Objet datetime analysé
        """
        formats = {
            'apache': ['%d/%b/%Y:%H:%M:%S %z', '%d/%b/%Y:%H:%M:%S'],
            'nginx': ['%d/%b/%Y:%H:%M:%S %z', '%d/%b/%Y:%H:%M:%S'],
            'syslog': ['%b %d %H:%M:%S'],
            'firewall': ['%Y-%m-%d %H:%M:%S']
        }
        
        if log_format not in formats:
            return None
            
        for fmt in formats[log_format]:
            try:
                if log_format == 'syslog':
                    # Les logs système ont besoin d'ajouter l'année
                    current_year = datetime.now().year
                    timestamp_str = f"{current_year} {timestamp_str}"
                    fmt = f"%Y {fmt}"
                    
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
                
        return None
    
    def parse_file(self, file_path: str, log_format: Optional[str] = None) -> Generator[Dict, None, None]:
        """
        Analyser un fichier de log
        
        Args:
            file_path: Chemin du fichier de log
            log_format: Format de log spécifié, détection automatique si None
            
        Yields:
            Dictionnaire d'enregistrement de log analysé
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Fichier de log non trouvé: {file_path}")
        
        # Détecter l'encodage du fichier
        encoding = self._detect_encoding(file_path)
        
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            first_line = f.readline()
            
            # Si aucun format n'est spécifié, détection automatique
            if log_format is None:
                log_format = self.detect_log_format(first_line)
                if log_format is None:
                    raise ValueError(f"Format de log non reconnu: {first_line[:100]}...")
            
            # Remettre le pointeur de fichier au début
            f.seek(0)
            
            line_number = 0
            for line in f:
                line_number += 1
                parsed = self.parse_log_line(line, log_format)
                
                if parsed:
                    parsed['line_number'] = line_number
                    parsed['file_path'] = str(file_path)
                    yield parsed
    
    def _detect_encoding(self, file_path: Path) -> str:
        """
        Détecter l'encodage du fichier
        
        Args:
            file_path: Chemin du fichier
            
        Returns:
            Encodage détecté
        """
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Lire les premiers 10KB pour détecter l'encodage
            result = chardet.detect(raw_data)
            return result['encoding'] or 'utf-8'
    
    def parse_to_dataframe(self, file_path: str, log_format: Optional[str] = None) -> pd.DataFrame:
        """
        Analyser un fichier de log en DataFrame
        
        Args:
            file_path: Chemin du fichier de log
            log_format: Format de log
            
        Returns:
            DataFrame contenant les données analysées
        """
        records = list(self.parse_file(file_path, log_format))
        
        if not records:
            return pd.DataFrame()
            
        df = pd.DataFrame(records)
        
        # Définir le timestamp comme index (si présent)
        if 'parsed_timestamp' in df.columns:
            df['parsed_timestamp'] = pd.to_datetime(df['parsed_timestamp'])
            df.set_index('parsed_timestamp', inplace=True)
            
        return df
    
    def get_supported_formats(self) -> List[str]:
        """
        Obtenir la liste des formats de log supportés
        
        Returns:
            Liste des formats supportés
        """
        return list(self.patterns.keys())

class LogBatchProcessor:
    """Processeur de logs en lot"""
    
    def __init__(self):
        self.parser = LogParser()
        self.config = get_config()
    
    def process_directory(self, directory_path: str) -> pd.DataFrame:
        """
        Traiter en lot tous les fichiers de log d'un répertoire
        
        Args:
            directory_path: Chemin du répertoire
            
        Returns:
            DataFrame fusionné
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Répertoire non trouvé: {directory}")
        
        all_dataframes = []
        
        # Parcourir tous les fichiers du répertoire
        for file_path in directory.rglob("*.log"):
            try:
                print(f"Traitement en cours: {file_path}")
                df = self.parser.parse_to_dataframe(str(file_path))
                
                if not df.empty:
                    all_dataframes.append(df)
                    
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {file_path}: {e}")
                continue
        
        # Fusionner tous les DataFrames
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Obtenir les statistiques des logs
        
        Args:
            df: DataFrame des logs
            
        Returns:
            Dictionnaire de statistiques
        """
        if df.empty:
            return {}
        
        stats = {
            'total_logs': len(df),
            'date_range': {
                'start': df.index.min() if hasattr(df.index, 'min') else None,
                'end': df.index.max() if hasattr(df.index, 'max') else None
            },
            'log_formats': df['log_format'].value_counts().to_dict() if 'log_format' in df.columns else {},
            'unique_ips': df['ip'].nunique() if 'ip' in df.columns else 0,
            'status_codes': df['status'].value_counts().to_dict() if 'status' in df.columns else {}
        }
        
        return stats 