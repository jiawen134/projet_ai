"""
Chatbot d'analyse de logs réseau
Intelligence artificielle spécialisée dans l'analyse et l'explication de logs
"""
import pandas as pd
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import re

from tools.llm_api import LLMManager, ChatMessage, create_message


class LogAnalysisChatbot:
    """Chatbot spécialisé dans l'analyse de logs réseau"""
    
    def __init__(self, llm_provider: str = "openai"):
        self.llm_manager = LLMManager()
        self.llm_provider = llm_provider
        self.conversation_history = []
        self.data_context = None
        self.anomaly_context = None
        
        # Messages système pour guider le comportement du chatbot
        self.system_prompt = """Tu es un expert en cybersécurité et analyse de logs réseau. 
        Ton rôle est d'aider les utilisateurs à comprendre et analyser leurs logs réseau.
        
        Capacités principales:
        - Analyser les patterns de trafic réseau
        - Expliquer les anomalies et comportements suspects
        - Fournir des recommandations de sécurité
        - Interpréter les codes de statut HTTP et erreurs
        - Identifier les menaces potentielles
        
        Réponds de manière claire, professionnelle et actionable.
        Utilise les données de logs fournies pour donner des réponses précises et contextuelles.
        Si tu n'as pas assez d'informations, demande des clarifications.
        """
    
    def set_data_context(self, log_data: pd.DataFrame, anomaly_data: Optional[pd.DataFrame] = None):
        """
        Définir le contexte des données de logs
        
        Args:
            log_data: DataFrame contenant les logs
            anomaly_data: DataFrame contenant les données d'anomalies (optionnel)
        """
        self.data_context = log_data
        self.anomaly_context = anomaly_data
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Obtenir un résumé des données de logs"""
        if self.data_context is None or self.data_context.empty:
            return {"error": "Aucune donnée de logs disponible"}
        
        df = self.data_context
        summary = {
            "total_logs": len(df),
            "periode": {
                "debut": str(df['parsed_timestamp'].min()) if 'parsed_timestamp' in df.columns else "Non disponible",
                "fin": str(df['parsed_timestamp'].max()) if 'parsed_timestamp' in df.columns else "Non disponible"
            },
            "ips_uniques": df['ip'].nunique() if 'ip' in df.columns else 0,
            "codes_statut": df['status'].value_counts().to_dict() if 'status' in df.columns else {},
            "methodes_http": df['method'].value_counts().to_dict() if 'method' in df.columns else {},
            "formats_logs": df['log_format'].value_counts().to_dict() if 'log_format' in df.columns else {}
        }
        
        # Ajouter les informations sur les anomalies
        if self.anomaly_context is not None and not self.anomaly_context.empty:
            anomalies = self.anomaly_context[self.anomaly_context['is_anomaly']]
            summary["anomalies"] = {
                "total": len(anomalies),
                "pourcentage": round(len(anomalies) / len(df) * 100, 2),
                "ips_anormales": anomalies['ip'].value_counts().head(5).to_dict() if 'ip' in anomalies.columns else {},
                "codes_statut_anormaux": anomalies['status'].value_counts().to_dict() if 'status' in anomalies.columns else {}
            }
        
        return summary
    
    def analyze_query_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Analyser l'intention de la requête utilisateur
        
        Args:
            user_input: Entrée utilisateur
            
        Returns:
            Dictionnaire contenant l'intention analysée
        """
        intent = {
            "type": "general",
            "entities": {},
            "requires_data": False
        }
        
        user_lower = user_input.lower()
        
        # Détecter les types de requêtes
        if any(keyword in user_lower for keyword in ['anomalie', 'anormal', 'suspect', 'bizarre']):
            intent["type"] = "anomaly_analysis"
            intent["requires_data"] = True
        
        elif any(keyword in user_lower for keyword in ['ip', 'adresse']):
            intent["type"] = "ip_analysis"
            intent["requires_data"] = True
            # Extraire les IPs mentionnées
            ip_pattern = r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b'
            ips = re.findall(ip_pattern, user_input)
            if ips:
                intent["entities"]["ips"] = ips
        
        elif any(keyword in user_lower for keyword in ['erreur', '404', '500', 'statut']):
            intent["type"] = "error_analysis"
            intent["requires_data"] = True
        
        elif any(keyword in user_lower for keyword in ['résumé', 'rapport', 'synthèse', 'vue d\'ensemble']):
            intent["type"] = "summary"
            intent["requires_data"] = True
        
        elif any(keyword in user_lower for keyword in ['tendance', 'évolution', 'temps']):
            intent["type"] = "trend_analysis"
            intent["requires_data"] = True
        
        elif any(keyword in user_lower for keyword in ['sécurité', 'recommandation', 'conseil']):
            intent["type"] = "security_advice"
            intent["requires_data"] = True
        
        return intent
    
    def generate_data_context_prompt(self, intent: Dict[str, Any]) -> str:
        """
        Générer un prompt avec le contexte des données
        
        Args:
            intent: Intention analysée de la requête
            
        Returns:
            Prompt enrichi avec les données contextuelles
        """
        if not intent["requires_data"] or self.data_context is None:
            return ""
        
        summary = self.get_data_summary()
        
        context_prompt = f"""
CONTEXTE DES DONNÉES:
===================
• Total des logs: {summary['total_logs']}
• Période: {summary['periode']['debut']} à {summary['periode']['fin']}
• IPs uniques: {summary['ips_uniques']}

CODES DE STATUT HTTP:
{json.dumps(summary['codes_statut'], indent=2)}

MÉTHODES HTTP:
{json.dumps(summary['methodes_http'], indent=2)}
"""

        # Ajouter le contexte des anomalies si disponible
        if 'anomalies' in summary:
            context_prompt += f"""
ANOMALIES DÉTECTÉES:
==================
• Total: {summary['anomalies']['total']} ({summary['anomalies']['pourcentage']}%)
• IPs anormales principales:
{json.dumps(summary['anomalies']['ips_anormales'], indent=2)}
"""

        # Ajouter des données spécifiques selon l'intention
        if intent["type"] == "ip_analysis" and "ips" in intent["entities"]:
            for ip in intent["entities"]["ips"]:
                ip_data = self._get_ip_details(ip)
                if ip_data:
                    context_prompt += f"\nDÉTAILS POUR IP {ip}:\n{json.dumps(ip_data, indent=2)}\n"
        
        return context_prompt
    
    def _get_ip_details(self, ip: str) -> Optional[Dict]:
        """Obtenir les détails d'une IP spécifique"""
        if self.data_context is None or 'ip' not in self.data_context.columns:
            return None
        
        ip_logs = self.data_context[self.data_context['ip'] == ip]
        if ip_logs.empty:
            return None
        
        details = {
            "total_requetes": len(ip_logs),
            "codes_statut": ip_logs['status'].value_counts().to_dict() if 'status' in ip_logs.columns else {},
            "methodes": ip_logs['method'].value_counts().to_dict() if 'method' in ip_logs.columns else {},
            "chemins_acces": ip_logs['path'].value_counts().head(10).to_dict() if 'path' in ip_logs.columns else {},
            "premiere_occurrence": str(ip_logs['parsed_timestamp'].min()) if 'parsed_timestamp' in ip_logs.columns else "Non disponible",
            "derniere_occurrence": str(ip_logs['parsed_timestamp'].max()) if 'parsed_timestamp' in ip_logs.columns else "Non disponible"
        }
        
        # Vérifier si cette IP est dans les anomalies
        if self.anomaly_context is not None and 'ip' in self.anomaly_context.columns:
            ip_anomalies = self.anomaly_context[
                (self.anomaly_context['ip'] == ip) & 
                (self.anomaly_context['is_anomaly'] == True)
            ]
            if not ip_anomalies.empty:
                details["est_anormale"] = True
                details["score_anomalie_moyen"] = ip_anomalies['anomaly_score'].mean()
            else:
                details["est_anormale"] = False
        
        return details
    
    def process_query(self, user_input: str) -> str:
        """
        Traiter une requête utilisateur
        
        Args:
            user_input: Entrée utilisateur
            
        Returns:
            Réponse du chatbot
        """
        # Analyser l'intention
        intent = self.analyze_query_intent(user_input)
        
        # Construire les messages pour le LLM
        messages = []
        
        # Message système
        messages.append(create_message("system", self.system_prompt))
        
        # Ajouter le contexte des données si nécessaire
        data_context = self.generate_data_context_prompt(intent)
        if data_context:
            messages.append(create_message("system", data_context))
        
        # Ajouter l'historique de conversation (derniers messages)
        for msg in self.conversation_history[-6:]:  # Garder 6 derniers messages pour le contexte
            messages.append(msg)
        
        # Ajouter la requête utilisateur
        user_message = create_message("user", user_input)
        messages.append(user_message)
        
        # Envoyer à l'LLM
        result = self.llm_manager.chat(messages, provider=self.llm_provider)
        
        if result["success"]:
            response = result["content"]
            
            # Ajouter à l'historique
            self.conversation_history.append(user_message)
            self.conversation_history.append(create_message("assistant", response))
            
            # Limiter l'historique à 20 messages
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return response
        else:
            return f"Désolé, je rencontre un problème technique: {result['error']}"
    
    def get_conversation_history(self) -> List[ChatMessage]:
        """Obtenir l'historique de conversation"""
        return self.conversation_history.copy()
    
    def clear_conversation(self):
        """Effacer l'historique de conversation"""
        self.conversation_history = []
    
    def suggest_questions(self) -> List[str]:
        """Suggérer des questions que l'utilisateur peut poser"""
        if self.data_context is None or self.data_context.empty:
            return [
                "Comment puis-je charger mes données de logs?",
                "Quels formats de logs sont supportés?",
                "Peux-tu m'expliquer les fonctionnalités disponibles?"
            ]
        
        suggestions = [
            "Peux-tu me donner un résumé de mes logs?",
            "Y a-t-il des comportements anormaux dans mes logs?",
            "Quelles sont les IPs les plus actives?",
            "Peux-tu analyser les erreurs dans mes logs?"
        ]
        
        # Ajouter des suggestions spécifiques si des anomalies sont détectées
        if self.anomaly_context is not None and not self.anomaly_context.empty:
            anomalies = self.anomaly_context[self.anomaly_context['is_anomaly']]
            if not anomalies.empty:
                suggestions.extend([
                    "Pourquoi ces IPs sont-elles considérées comme anormales?",
                    "Que dois-je faire concernant ces anomalies?",
                    "Ces anomalies représentent-elles une menace de sécurité?"
                ])
        
        return suggestions


def create_chatbot(llm_provider: str = "openai") -> LogAnalysisChatbot:
    """Créer une instance de chatbot"""
    return LogAnalysisChatbot(llm_provider=llm_provider) 