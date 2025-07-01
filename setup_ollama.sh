#!/bin/bash
"""
Script d'installation et de configuration d'Ollama
Pour l'assistant IA du projet d'analyse de logs
"""

echo "🚀 Installation d'Ollama pour l'assistant IA"
echo "============================================="

# Vérifier si Ollama est déjà installé
if command -v ollama &> /dev/null; then
    echo "✅ Ollama est déjà installé"
    ollama --version
else
    echo "📥 Installation d'Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo "✅ Ollama installé avec succès"
    else
        echo "❌ Échec de l'installation d'Ollama"
        exit 1
    fi
fi

echo ""
echo "🔧 Configuration d'Ollama..."

# Démarrer le service Ollama en arrière-plan
echo "📡 Démarrage du service Ollama..."
nohup ollama serve > ollama.log 2>&1 &
OLLAMA_PID=$!

# Attendre que le service soit prêt
echo "⏳ Attente du démarrage du service..."
sleep 5

# Vérifier si le service fonctionne
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Service Ollama démarré avec succès"
else
    echo "❌ Le service Ollama ne répond pas"
    echo "💡 Vous pouvez démarrer manuellement avec: ollama serve"
    exit 1
fi

echo ""
echo "📚 Téléchargement des modèles recommandés..."

# Modèles recommandés pour l'analyse de logs
MODELS=("qwen2.5:7b" "llama3.1:8b")

for model in "${MODELS[@]}"; do
    echo "📥 Téléchargement de $model..."
    ollama pull $model
    
    if [ $? -eq 0 ]; then
        echo "✅ $model téléchargé avec succès"
    else
        echo "⚠️ Échec du téléchargement de $model"
        echo "💡 Vous pouvez le télécharger plus tard avec: ollama pull $model"
    fi
    echo ""
done

echo "🎯 Installation terminée!"
echo ""
echo "📋 Résumé:"
echo "- Service Ollama: en cours d'exécution (PID: $OLLAMA_PID)"
echo "- URL du service: http://localhost:11434"
echo "- Log du service: ollama.log"
echo ""
echo "🧪 Pour tester:"
echo "python test_chatbot.py"
echo ""
echo "🚀 Pour lancer l'application:"
echo "streamlit run app.py"
echo ""
echo "🛑 Pour arrêter Ollama:"
echo "kill $OLLAMA_PID"

# Sauvegarder le PID pour pouvoir arrêter le service plus tard
echo $OLLAMA_PID > ollama.pid
echo "💾 PID sauvegardé dans ollama.pid" 