# 🔍 Outil d'Analyse Automatisée de Logs Réseau avec IA

Un système complet d'analyse de logs réseau alimenté par l'intelligence artificielle, avec interface web interactive et assistant de chat intégré.

## 📋 Fonctionnalités

### Module 1: Analyse des Logs Réseau ✅
- **Analyse multi-format** : Apache, Nginx, Syslog, Firewall
- **Détection d'anomalies** : Algorithme Isolation Forest pour identifier les comportements suspects
- **Visualisation interactive** : Graphiques en temps réel avec Plotly
- **Tableaux de bord web** : Interface Streamlit intuitive

### Module 2: Assistant IA Intégré ✅
- **Chatbot spécialisé** : Expert en cybersécurité et analyse de logs
- **Multi-LLM** : Support OpenAI GPT-4o (principal) et Ollama (local)
- **Analyse contextuelle** : Compréhension intelligente des données de logs
- **Recommandations de sécurité** : Conseils professionnels automatisés

## 🚀 Installation Rapide

### Prérequis
- **Python 3.12+** (testé avec Python 3.12)
- **Git** pour cloner le repository
- **Clé API OpenAI** (recommandé pour de meilleures performances)

### Installation

```cmd
# 1. Cloner le repository
git clone <URL_DU_REPOSITORY>
cd projet_ai

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API OpenAI
echo OPENAI_API_KEY=votre_clé_api_ici > .env

# 5. Lancer l'application
streamlit run modules/log_analyzer/dashboard.py --server.port 8501
```

### Installation avec Ollama (Local, Optionnel)

#### Linux/macOS:
```bash
# Installation d'Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Démarrer le service
ollama serve

# Télécharger un modèle (recommandé)
ollama pull qwen2.5:7b
```

#### Windows:
```cmd
# Télécharger et installer Ollama depuis https://ollama.ai/download
# Puis dans un terminal:
ollama pull qwen2.5:7b
```

## 🖥️ Utilisation

### 1. Accès à l'Interface Web
Ouvrez votre navigateur à : `http://localhost:8501`

### 2. Onglets Disponibles

#### 📊 Analyse des Logs
- **Chargement des données** : Fichier unique, scan de répertoire, ou données d'exemple
- **Visualisations** : Séries temporelles, codes de statut, analyse IP
- **Détection d'anomalies** : Algorithmes ML pour identifier les comportements suspects
- **Export** : Tableaux filtrables et exportables

#### 🤖 Assistant IA
- **Chat intelligent** : Questions en langage naturel sur vos logs
- **Analyse contextuelle** : Compréhension automatique de vos données
- **Recommandations** : Conseils de sécurité personnalisés
- **Support multilingue** : Français, anglais, chinois

### 3. Exemples de Questions pour l'Assistant IA

```
"Peux-tu me donner un résumé de mes logs?"
"Y a-t-il des comportements anormaux dans mes logs?"
"Analyse les erreurs 404 et donne-moi des recommandations"
"Quelles sont les IPs les plus suspectes?"
"Comment améliorer ma sécurité réseau?"
```

## 📁 Structure du Projet

```
projet_ai/
├── modules/
│   ├── log_analyzer/          # Module d'analyse de logs
│   │   ├── dashboard.py       # Interface Streamlit principale
│   │   ├── parser.py          # Analyseurs de logs multi-format
│   │   └── anomaly_detector.py # Détection d'anomalies ML
│   └── chatbot/               # Module chatbot IA
│       └── core.py            # Logique principale du chatbot
├── tools/
│   └── llm_api.py            # API unifiée pour LLM (OpenAI/Ollama)
├── config/
│   └── log_analyzer_config.py # Configuration système
├── data/
│   ├── logs/                 # Dossier pour vos fichiers de logs
│   ├── models/              # Modèles ML (à venir)
│   └── videos/              # Vidéos pour analyse (à venir)
├── requirements.txt          # Dépendances Python
├── .env                     # Configuration API (créer manuellement)
└── README.md               # Ce fichier
```

