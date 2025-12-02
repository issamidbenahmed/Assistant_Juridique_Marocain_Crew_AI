#!/usr/bin/env python
"""Script pour tester CrewAI avec une vraie question"""
import requests
import json
import time

API_URL = "http://localhost:8000"

def test_question():
    """Pose une question pour voir CrewAI en action"""
    
    print("=" * 80)
    print("🧪 TEST CREWAI MULTI-AGENT")
    print("=" * 80)
    
    # Vérifier que l'API est disponible
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ L'API n'est pas disponible. Démarrez le backend d'abord.")
            return
        print("✅ API disponible")
    except Exception as e:
        print(f"❌ Impossible de se connecter à l'API: {e}")
        print("💡 Démarrez le backend avec: python -m uvicorn app.main:app --reload")
        return
    
    # Vérifier le statut
    status = requests.get(f"{API_URL}/status").json()
    print(f"\n📊 Statut du service:")
    print(f"   - Documents: {status['status']['vector_store_stats']['total_documents']}")
    print(f"   - Ollama: {'✅' if status['status']['ollama_available'] else '❌'}")
    print(f"   - Gemini: {'✅' if status['status']['gemini_available'] else '❌'}")
    print(f"   - CrewAI: {'✅' if status['status']['crew_agents_enabled'] else '❌'}")
    
    if not status['status']['crew_agents_enabled']:
        print("\n⚠️  CrewAI n'est pas activé!")
        return
    
    # Poser une question
    question = "Quel est le capital minimum pour créer une société anonyme au Maroc?"
    
    print(f"\n❓ Question: {question}")
    print("\n⏳ Envoi de la question... (les agents CrewAI vont travailler en parallèle)")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "context_limit": 5
            },
            timeout=300  # 5 minutes max pour laisser le temps aux agents
        )
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print("\n" + "=" * 80)
            print("✅ RÉPONSE REÇUE")
            print("=" * 80)
            print(f"\n📝 Réponse:\n{result['answer']}\n")
            print(f"📊 Statistiques:")
            print(f"   - Temps de traitement: {result['processing_time']:.2f}s")
            print(f"   - Score de confiance: {result['confidence_score']:.2%}")
            print(f"   - Nombre de sources: {len(result['sources'])}")
            
            if result['sources']:
                print(f"\n📚 Sources utilisées:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"   {i}. {source['source_file']} - {source['article']} (score: {source['relevance_score']:.2f})")
        else:
            print(f"\n❌ Erreur: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("\n⏱️  Timeout - La requête a pris trop de temps")
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
    
    print("\n" + "=" * 80)
    print("💡 Consultez les logs du backend pour voir CrewAI en action!")
    print("=" * 80)

if __name__ == "__main__":
    test_question()
