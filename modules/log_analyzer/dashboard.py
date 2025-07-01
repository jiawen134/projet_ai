"""
Tableau de bord de visualisation d'analyse de logs
Construit une interface web interactive avec Streamlit
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

from modules.log_analyzer.parser import LogParser, LogBatchProcessor
from modules.log_analyzer.anomaly_detector import LogAnomalyDetector
from config.log_analyzer_config import get_config

class LogAnalyzerDashboard:
    """Tableau de bord d'analyse de logs"""
    
    def __init__(self):
        self.config = get_config()
        self.parser = LogParser()
        self.batch_processor = LogBatchProcessor()
        self.anomaly_detector = LogAnomalyDetector()
        
    def run(self):
        """Exécuter le tableau de bord"""
        st.set_page_config(
            page_title="Tableau de bord d'analyse de logs réseau",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔍 Outil d'analyse automatisée de logs réseau")
        st.markdown("---")
        
        # Barre latérale
        self._render_sidebar()
        
        # Zone de contenu principal
        if 'data' not in st.session_state:
            self._render_upload_section()
        else:
            self._render_analysis_dashboard()
    
    def _render_sidebar(self):
        """Rendre la barre latérale"""
        st.sidebar.title("🛠️ Panneau de contrôle")
        
        # Options de chargement des données
        st.sidebar.header("Chargement des données")
        load_option = st.sidebar.radio(
            "Choisir la source de données :",
            ["Télécharger fichier", "Scanner répertoire", "Données d'exemple"]
        )
        
        if load_option == "Télécharger fichier":
            uploaded_file = st.sidebar.file_uploader(
                "Télécharger fichier de logs",
                type=['log', 'txt', 'csv'],
                help="Supporte les formats Apache, Nginx, syslog, etc."
            )
            
            if uploaded_file is not None:
                self._load_uploaded_file(uploaded_file)
        
        elif load_option == "Scanner répertoire":
            directory_path = st.sidebar.text_input(
                "Chemin du répertoire de logs",
                value="data/logs",
                help="Entrer le chemin du répertoire contenant les fichiers de logs"
            )
            
            if st.sidebar.button("Scanner répertoire"):
                self._load_directory(directory_path)
        
        elif load_option == "Données d'exemple":
            if st.sidebar.button("Charger données d'exemple"):
                self._load_sample_data()
        
        # Options d'analyse
        if 'data' in st.session_state:
            st.sidebar.header("Options d'analyse")
            
            # Filtrage par plage de temps
            if 'parsed_timestamp' in st.session_state.data.columns:
                self._render_time_filter()
            
            # Options de détection d'anomalies
            self._render_anomaly_options()
            
            # Effacer les données
            if st.sidebar.button("Effacer les données"):
                self._clear_data()
    
    def _render_time_filter(self):
        """Rendre le filtre de temps"""
        data = st.session_state.data
        
        if data.empty or 'parsed_timestamp' not in data.columns:
            return
        
        timestamps = pd.to_datetime(data['parsed_timestamp'])
        min_time = timestamps.min()
        max_time = timestamps.max()
        
        st.sidebar.subheader("Plage de temps")
        
        time_range = st.sidebar.slider(
            "Sélectionner la plage de temps",
            min_value=min_time.to_pydatetime(),
            max_value=max_time.to_pydatetime(),
            value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
            format="MM/DD/YY HH:mm"
        )
        
        # Appliquer le filtrage temporel
        mask = (timestamps >= time_range[0]) & (timestamps <= time_range[1])
        st.session_state.filtered_data = data[mask]
    
    def _render_anomaly_options(self):
        """Rendre les options de détection d'anomalies"""
        st.sidebar.subheader("Détection d'anomalies")
        
        enable_anomaly = st.sidebar.checkbox("Activer la détection d'anomalies", value=True)
        
        if enable_anomaly:
            contamination = st.sidebar.slider(
                "Proportion d'anomalies",
                min_value=0.01,
                max_value=0.3,
                value=0.1,
                step=0.01,
                help="Proportion attendue de données anormales"
            )
            
            if st.sidebar.button("Réentraîner le modèle de détection d'anomalies"):
                self._train_anomaly_model(contamination)
    
    def _load_uploaded_file(self, uploaded_file):
        """Charger le fichier téléchargé"""
        try:
            # Sauvegarder le fichier téléchargé
            file_path = f"data/logs/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Analyser le fichier
            with st.spinner("Analyse du fichier de logs en cours..."):
                df = self.parser.parse_to_dataframe(file_path)
                
            if not df.empty:
                st.session_state.data = df
                st.session_state.filtered_data = df
                st.sidebar.success(f"Chargement réussi de {len(df)} enregistrements de logs")
            else:
                st.sidebar.error("Échec de l'analyse du fichier ou fichier vide")
                
        except Exception as e:
            st.sidebar.error(f"Échec du chargement du fichier: {str(e)}")
    
    def _load_directory(self, directory_path):
        """Charger les fichiers de logs du répertoire"""
        try:
            with st.spinner("Scan du répertoire en cours..."):
                df = self.batch_processor.process_directory(directory_path)
            
            if not df.empty:
                st.session_state.data = df
                st.session_state.filtered_data = df
                st.sidebar.success(f"Chargement réussi de {len(df)} enregistrements de logs")
            else:
                st.sidebar.warning("Aucun fichier de logs valide trouvé dans le répertoire")
                
        except Exception as e:
            st.sidebar.error(f"Échec du scan du répertoire: {str(e)}")
    
    def _load_sample_data(self):
        """Charger les données d'exemple"""
        # Créer des données de logs d'exemple
        sample_logs = []
        base_time = datetime.now() - timedelta(days=1)
        
        for i in range(1000):
            timestamp = base_time + timedelta(minutes=i)
            ip = f"192.168.1.{np.random.randint(1, 255)}"
            status = np.random.choice([200, 404, 500], p=[0.8, 0.15, 0.05])
            method = np.random.choice(['GET', 'POST', 'PUT'], p=[0.7, 0.2, 0.1])
            path = np.random.choice(['/index.html', '/api/data', '/login', '/admin'])
            
            sample_logs.append({
                'timestamp': timestamp.strftime('%d/%b/%Y:%H:%M:%S'),
                'parsed_timestamp': timestamp,
                'ip': ip,
                'method': method,
                'path': path,
                'status': str(status),
                'size': str(np.random.randint(100, 10000)),
                'log_format': 'apache',
                'raw_line': f'{ip} - - [{timestamp.strftime("%d/%b/%Y:%H:%M:%S")}] "{method} {path} HTTP/1.1" {status} {np.random.randint(100, 10000)}'
            })
        
        df = pd.DataFrame(sample_logs)
        st.session_state.data = df
        st.session_state.filtered_data = df
        st.sidebar.success("Données d'exemple chargées avec succès")
    
    def _train_anomaly_model(self, contamination):
        """Entraîner le modèle de détection d'anomalies"""
        try:
            # Mettre à jour la configuration
            self.config.anomaly_detection['contamination'] = contamination
            
            with st.spinner("Entraînement du modèle de détection d'anomalies..."):
                result = self.anomaly_detector.train_anomaly_detector(st.session_state.data)
            
            if result['status'] == 'success':
                st.sidebar.success(f"Entraînement du modèle terminé ! {result['anomalies_detected']} anomalies détectées")
                
                # Détecter les anomalies
                anomaly_df = self.anomaly_detector.detect_anomalies(st.session_state.filtered_data)
                st.session_state.anomaly_data = anomaly_df
            else:
                st.sidebar.error(f"Échec de l'entraînement du modèle: {result.get('message', 'erreur inconnue')}")
                
        except Exception as e:
            st.sidebar.error(f"Échec de la détection d'anomalies: {str(e)}")
    
    def _clear_data(self):
        """Effacer les données"""
        for key in ['data', 'filtered_data', 'anomaly_data']:
            if key in st.session_state:
                del st.session_state[key]
        st.sidebar.success("Données effacées")
    
    def _render_upload_section(self):
        """Rendre la section de téléchargement"""
        st.markdown("""
        ## Bienvenue dans l'outil d'analyse de logs réseau !
        
        Veuillez charger vos données de logs via le panneau de gauche :
        
        - **Télécharger fichier**: Supporte les formats Apache, Nginx, syslog, etc.
        - **Scanner répertoire**: Traitement en lot de tous les fichiers .log d'un répertoire
        - **Données d'exemple**: Expérience rapide des fonctionnalités de l'outil
        
        ### Formats de logs supportés:
        - Apache Common/Combined Log Format
        - Logs d'accès Nginx
        - Logs système (Syslog)
        - Logs de pare-feu
        """)
        
        # Afficher les informations de configuration
        with st.expander("Voir la configuration actuelle"):
            st.json({
                "Formats supportés": self.config.supported_formats,
                "Algorithme de détection d'anomalies": self.config.anomaly_detection['algorithm'],
                "Seuil d'anomalies": self.config.anomaly_detection['contamination']
            })
    
    def _render_analysis_dashboard(self):
        """Rendre le tableau de bord d'analyse"""
        data = st.session_state.get('filtered_data', st.session_state.data)
        
        # Métriques de vue d'ensemble
        self._render_overview_metrics(data)
        
        # Zone des graphiques
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_time_series_chart(data)
            self._render_ip_analysis(data)
        
        with col2:
            self._render_status_code_chart(data)
            self._render_method_analysis(data)
        
        # Résultats de détection d'anomalies
        if 'anomaly_data' in st.session_state:
            self._render_anomaly_analysis()
        
        # Tableau de données détaillées
        self._render_data_table(data)
    
    def _render_overview_metrics(self, data):
        """Rendre les métriques de vue d'ensemble"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total logs", len(data))
        
        with col2:
            unique_ips = data['ip'].nunique() if 'ip' in data.columns else 0
            st.metric("IPs uniques", unique_ips)
        
        with col3:
            if 'status' in data.columns:
                error_rate = (data['status'].astype(str).str.startswith('4') | 
                             data['status'].astype(str).str.startswith('5')).mean() * 100
                st.metric("Taux d'erreur", f"{error_rate:.1f}%")
            else:
                st.metric("Taux d'erreur", "N/A")
        
        with col4:
            if 'anomaly_data' in st.session_state:
                anomaly_count = st.session_state.anomaly_data['is_anomaly'].sum()
                st.metric("Anomalies", anomaly_count)
            else:
                st.metric("Anomalies", "Non détectées")
    
    def _render_time_series_chart(self, data):
        """Rendre le graphique de série temporelle"""
        st.subheader("📈 Tendance des accès dans le temps")
        
        if 'parsed_timestamp' not in data.columns:
            st.warning("Données d'horodatage manquantes")
            return
        
        # Agréger les données par heure
        hourly_data = data.groupby(pd.to_datetime(data['parsed_timestamp']).dt.floor('H')).size().reset_index()
        hourly_data.columns = ['hour', 'count']
        
        fig = px.line(
            hourly_data, 
            x='hour', 
            y='count',
            title="Volume d'accès par heure",
            labels={'hour': 'Temps', 'count': 'Nombre d\'accès'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_status_code_chart(self, data):
        """Rendre le graphique des codes de statut"""
        st.subheader("📊 Distribution des codes de statut HTTP")
        
        if 'status' not in data.columns:
            st.warning("Données de codes de statut manquantes")
            return
        
        status_counts = data['status'].value_counts()
        
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Distribution des codes de statut"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_ip_analysis(self, data):
        """Rendre l'analyse des adresses IP"""
        st.subheader("🌐 Top adresses IP")
        
        if 'ip' not in data.columns:
            st.warning("Données d'adresses IP manquantes")
            return
        
        top_ips = data['ip'].value_counts().head(10)
        
        fig = px.bar(
            x=top_ips.values,
            y=top_ips.index,
            orientation='h',
            title="Adresses IP avec le plus d'accès",
            labels={'x': 'Nombre d\'accès', 'y': 'Adresse IP'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_method_analysis(self, data):
        """Rendre l'analyse des méthodes de requête"""
        st.subheader("🔧 Distribution des méthodes HTTP")
        
        if 'method' not in data.columns:
            st.warning("Données de méthodes HTTP manquantes")
            return
        
        method_counts = data['method'].value_counts()
        
        fig = px.bar(
            x=method_counts.index,
            y=method_counts.values,
            title="Statistiques d'utilisation des méthodes HTTP",
            labels={'x': 'Méthode HTTP', 'y': 'Nombre d\'utilisations'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_anomaly_analysis(self):
        """Rendre l'analyse des anomalies"""
        st.subheader("🚨 Résultats de détection d'anomalies")
        
        anomaly_data = st.session_state.anomaly_data
        anomalies = anomaly_data[anomaly_data['is_anomaly']]
        
        if anomalies.empty:
            st.success("Aucun comportement anormal détecté")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution des IPs anormales
            if 'ip' in anomalies.columns:
                anomaly_ips = anomalies['ip'].value_counts().head(10)
                fig = px.bar(
                    x=anomaly_ips.values,
                    y=anomaly_ips.index,
                    orientation='h',
                    title="Adresses IP anormales",
                    labels={'x': 'Nombre d\'anomalies', 'y': 'Adresse IP'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribution des scores d'anomalies
            fig = px.histogram(
                anomalies,
                x='anomaly_score',
                title="Distribution des scores d'anomalies",
                labels={'anomaly_score': 'Score d\'anomalie', 'count': 'Fréquence'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Tableau des détails des anomalies
        st.subheader("Détails des anomalies")
        anomaly_display = anomalies[['parsed_timestamp', 'ip', 'method', 'path', 'status', 'anomaly_score']].copy()
        anomaly_display = anomaly_display.sort_values('anomaly_score')
        st.dataframe(anomaly_display, use_container_width=True)
    
    def _render_data_table(self, data):
        """Rendre le tableau de données"""
        st.subheader("📋 Données détaillées des logs")
        
        # Options d'affichage
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_rows = st.selectbox("Lignes à afficher", [50, 100, 500, 1000], index=1)
        
        with col2:
            if 'ip' in data.columns:
                selected_ip = st.selectbox(
                    "Filtrer par IP", 
                    ['Toutes'] + list(data['ip'].unique())
                )
            else:
                selected_ip = 'Toutes'
        
        with col3:
            if 'status' in data.columns:
                selected_status = st.selectbox(
                    "Filtrer par code de statut",
                    ['Tous'] + list(data['status'].unique())
                )
            else:
                selected_status = 'Tous'
        
        # Appliquer les filtres
        filtered_data = data.copy()
        
        if selected_ip != 'Toutes':
            filtered_data = filtered_data[filtered_data['ip'] == selected_ip]
        
        if selected_status != 'Tous':
            filtered_data = filtered_data[filtered_data['status'] == selected_status]
        
        # Afficher les données
        display_data = filtered_data.head(show_rows)
        st.dataframe(display_data, use_container_width=True)
        
        # Options de téléchargement
        if st.button("Télécharger les données actuelles en CSV"):
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Cliquer pour télécharger",
                data=csv,
                file_name=f"analyse_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def main():
    """Fonction principale"""
    dashboard = LogAnalyzerDashboard()
    dashboard.run()

if __name__ == "__main__":
    main() 