#!/bin/bash

# Script de démarrage pour l'assistant juridique marocain

echo "🇲🇦 Démarrage de l'Assistant Juridique Marocain..."

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé"
    exit 1
fi

# Démarrer les services
echo "🚀 Démarrage des services..."
docker-compose up -d

# Attendre que les services soient prêts
echo "⏳ Attente de l'initialisation des services..."
sleep 30

# Vérifier le statut
echo "🔍 Vérification du statut..."

# Vérifier Ollama
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama est prêt"
else
    echo "❌ Ollama n'est pas accessible"
fi

# Vérifier le backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend est prêt"
else
    echo "❌ Backend n'est pas accessible"
fi

echo ""
echo "🎉 Services démarrés !"
echo "📱 Frontend: http://localhost:4200"
echo "🔧 API: http://localhost:8000"
echo "📚 Documentation: http://localhost:8000/docs"
echo ""
echo "Pour démarrer le frontend:"
echo "cd frontend && npm install && npm start"
