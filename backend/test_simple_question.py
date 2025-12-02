#!/usr/bin/env python
"""Test simple sans CrewAI pour comparer"""
import requests
import time

API_URL = "http://localhost:8000"

def test_simple():
    question = "Quel est le capital minimum pour créer une société anonyme au Maroc?"
    
    print("🧪 Test avec RAG classique (sans CrewAI)")
    print(f"❓ Question: {question}\n")
    
    start = time.time()
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question, "context_limit": 3},
        timeout=30
    )
    elapsed = time.time() - start
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Réponse en {elapsed:.2f}s")
        print(f"📝 {result['answer'][:200]}...")
        print(f"📊 Confiance: {result['confidence_score']:.2%}")
        print(f"📚 Sources: {len(result['sources'])}")
    else:
        print(f"❌ Erreur: {response.text}")

if __name__ == "__main__":
    test_simple()
