"""
Outil API LLM unifié
Supporte Ollama (principal), OpenAI, Anthropic et d'autres fournisseurs
"""
import json
import requests
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  


@dataclass
class ChatMessage:
    """Message de chat"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: Optional[datetime] = None


class OllamaClient:
    """Client Ollama pour les modèles locaux"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.default_model = "qwen2.5:7b"
        
    def is_available(self) -> bool:
        """Vérifier si Ollama est disponible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def list_models(self) -> List[str]:
        """Lister les modèles disponibles"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
        except:
            pass
        return []
    
    def chat(self, messages: List[ChatMessage], model: Optional[str] = None, 
             temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Envoyer une requête de chat à Ollama
        
        Args:
            messages: Liste des messages de conversation
            model: Nom du modèle à utiliser
            temperature: Température de génération
            max_tokens: Nombre maximum de tokens
            
        Returns:
            Réponse du modèle
        """
        model = model or self.default_model
        
        # Essayer d'abord l'API chat
        result = self._try_chat_api(messages, model, temperature, max_tokens)
        if result["success"]:
            return result
        
        # Si chat API échoue, utiliser generate API
        print("Chat API échoué, utilisation de Generate API...")
        return self._try_generate_api(messages, model, temperature, max_tokens)
    
    def _try_chat_api(self, messages: List[ChatMessage], model: str, 
                      temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Essayer l'API chat d'Ollama"""
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        payload = {
            "model": model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=30  # Réduire le timeout pour l'API chat
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "content": data["message"]["content"],
                    "model": model,
                    "api_used": "chat",
                    "total_duration": data.get("total_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "eval_count": data.get("eval_count", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"Chat API HTTP {response.status_code}: {response.text}"
                }
        
        except requests.RequestException as e:
            return {
                "success": False,
                "error": f"Chat API erreur: {str(e)}"
            }
    
    def _try_generate_api(self, messages: List[ChatMessage], model: str,
                          temperature: float, max_tokens: int) -> Dict[str, Any]:
        """Essayer l'API generate d'Ollama (fallback)"""
        # Convertir les messages en prompt unique
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "content": data.get("response", ""),
                    "model": model,
                    "api_used": "generate",
                    "total_duration": data.get("total_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "eval_count": data.get("eval_count", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"Generate API HTTP {response.status_code}: {response.text}"
                }
        
        except requests.RequestException as e:
            return {
                "success": False,
                "error": f"Generate API erreur: {str(e)}"
            }
    
    def _messages_to_prompt(self, messages: List[ChatMessage]) -> str:
        """Convertir une liste de messages en prompt pour l'API generate"""
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        
        prompt += "Assistant: "
        return prompt


class OpenAIClient:
    """Client OpenAI (optionnel, pour backup)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1"
        self.default_model = "gpt-4o"
    
    def chat(self, messages: List[ChatMessage], model: Optional[str] = None,
             temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        """Chat avec OpenAI"""
        if not self.api_key:
            return {
                "success": False,
                "error": "Clé API OpenAI non configurée"
            }
        
        model = model or self.default_model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "content": data["choices"][0]["message"]["content"],
                    "model": model,
                    "usage": data.get("usage", {})
                }
            else:
                return {
                    "success": False,
                    "error": f"Erreur OpenAI: {response.text}"
                }
        
        except requests.RequestException as e:
            return {
                "success": False,
                "error": f"Erreur de connexion OpenAI: {str(e)}"
            }


class LLMManager:
    """Gestionnaire unifié des LLM"""
    
    def __init__(self):
        self.providers = {
            'ollama': OllamaClient(),
            'openai': OpenAIClient()
        }
        self.current_provider = 'openai'  
        self.fallback_provider = 'ollama'  
    
    def get_available_providers(self) -> List[str]:
        """Obtenir les fournisseurs disponibles"""
        available = []
        
        # Vérifier Ollama
        if self.providers['ollama'].is_available():
            available.append('ollama')
        
        # Vérifier OpenAI
        if self.providers['openai'].api_key:
            available.append('openai')
        
        return available
    
    def set_provider(self, provider: str) -> bool:
        """Définir le fournisseur actuel"""
        if provider in self.providers:
            self.current_provider = provider
            return True
        return False
    
    def chat(self, messages: List[ChatMessage], provider: Optional[str] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Envoyer une requête de chat
        
        Args:
            messages: Messages de conversation
            provider: Fournisseur à utiliser (optionnel)
            **kwargs: Arguments supplémentaires
        
        Returns:
            Réponse du modèle
        """
        provider = provider or self.current_provider
        
        if provider not in self.providers:
            return {
                "success": False,
                "error": f"Fournisseur non supporté: {provider}"
            }
        
        client = self.providers[provider]
        result = client.chat(messages, **kwargs)
        
        # Si échec et fallback disponible
        if not result["success"] and provider != self.fallback_provider:
            if self.fallback_provider in self.providers:
                print(f"Échec de {provider}, tentative avec {self.fallback_provider}")
                result = self.providers[self.fallback_provider].chat(messages, **kwargs)
                result["used_fallback"] = True
        
        result["provider"] = provider
        return result
    
    def get_models(self, provider: Optional[str] = None) -> List[str]:
        """Obtenir les modèles disponibles"""
        provider = provider or self.current_provider
        
        if provider == 'ollama':
            return self.providers['ollama'].list_models()
        elif provider == 'openai':
            return ["gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        
        return []


def create_message(role: str, content: str) -> ChatMessage:
    """Créer un message de chat"""
    return ChatMessage(role=role, content=content, timestamp=datetime.now())


def quick_chat(prompt: str, provider: str = "ollama", model: Optional[str] = None) -> str:
    """Fonction utilitaire pour un chat rapide"""
    manager = LLMManager()
    
    messages = [create_message("user", prompt)]
    result = manager.chat(messages, provider=provider, model=model)
    
    if result["success"]:
        return result["content"]
    else:
        return f"Erreur: {result['error']}"


if __name__ == "__main__":
    # Test du module
    import sys
    
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        response = quick_chat(prompt)
        print(response)
    else:
        # Test interactif
        manager = LLMManager()
        available = manager.get_available_providers()
        
        print("=== Test LLM API ===")
        print(f"Fournisseurs disponibles: {available}")
        
        if 'ollama' in available:
            models = manager.get_models('ollama')
            print(f"Modèles Ollama: {models}")
        
        # Test simple
        test_prompt = "Bonjour, peux-tu me dire ton nom et tes capacités ?"
        print(f"\nTest: {test_prompt}")
        
        result = manager.chat([create_message("user", test_prompt)])
        
        if result["success"]:
            print(f"\nRéponse ({result['provider']}): {result['content']}")
        else:
            print(f"\nErreur: {result['error']}") 