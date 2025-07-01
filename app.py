"""
Programme principal de l'outil d'analyse automatisée de logs réseau
Lance le tableau de bord Streamlit
"""

import sys
import os
from pathlib import Path

# Ajouter le répertoire racine du projet au chemin Python
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from modules.log_analyzer import LogAnalyzerDashboard

def main():
    """Fonction principale"""
    # S'assurer que le répertoire de données existe
    data_dir = project_root / "data" / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Lancer le tableau de bord
    dashboard = LogAnalyzerDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 