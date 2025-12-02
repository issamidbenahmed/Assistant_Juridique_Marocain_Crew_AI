"""
Test du cache basé sur l'historique
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.rag_service import RAGService
from app.models import QuestionRequest
import asyncio
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_history_cache():
    """Test que les questions répétées utilisent le cache"""
    logger.info("=" * 80)
    logger.info("TEST DU CACHE BASÉ SUR L'HISTORIQUE")
    logger.info("=" * 80)
    
    # Initialiser le service
    rag_service = RAGService()
    success = await rag_service.initialize()
    
    if not success:
        logger.error("❌ Échec de l'initialisation")
        return
    
    logger.info("✅ Service initialisé")
    
    # Question de test
    question = "Qu'est-ce que la TVA au Maroc?"
    request = QuestionRequest(question=question, context_limit=5)
    
    # Première fois - processus complet
    logger.info("\n" + "=" * 80)
    logger.info("🔄 PREMIÈRE QUESTION (processus complet)")
    logger.info("=" * 80)
    start1 = time.time()
    response1 = await rag_service.ask_question(request)
    time1 = time.time() - start1
    
    logger.info(f"⏱️  Temps: {time1:.2f} secondes")
    logger.info(f"📝 Réponse: {response1.answer[:100]}...")
    logger.info(f"🎯 Confiance: {response1.confidence_score}")
    
    # Deuxième fois - devrait utiliser le cache
    logger.info("\n" + "=" * 80)
    logger.info("⚡ DEUXIÈME QUESTION (devrait utiliser le cache)")
    logger.info("=" * 80)
    start2 = time.time()
    response2 = await rag_service.ask_question(request)
    time2 = time.time() - start2
    
    logger.info(f"⏱️  Temps: {time2:.2f} secondes")
    logger.info(f"📝 Réponse: {response2.answer[:100]}...")
    logger.info(f"🎯 Confiance: {response2.confidence_score}")
    
    # Vérification
    logger.info("\n" + "=" * 80)
    logger.info("📊 RÉSULTATS")
    logger.info("=" * 80)
    
    if time2 < 1.0:  # Cache devrait être instantané
        logger.info(f"✅ CACHE FONCTIONNE! ({time2:.3f}s vs {time1:.2f}s)")
        logger.info(f"⚡ Accélération: {time1/time2:.0f}x plus rapide")
    else:
        logger.warning(f"⚠️  Cache n'a pas été utilisé ({time2:.2f}s)")
    
    # Vérifier que les réponses sont identiques
    if response1.answer == response2.answer:
        logger.info("✅ Les réponses sont identiques")
    else:
        logger.warning("⚠️  Les réponses sont différentes")
    
    # Test avec question légèrement différente (majuscules/espaces)
    logger.info("\n" + "=" * 80)
    logger.info("🔄 QUESTION AVEC VARIATIONS (majuscules/espaces)")
    logger.info("=" * 80)
    
    request3 = QuestionRequest(question="  QU'EST-CE QUE LA TVA AU MAROC?  ", context_limit=5)
    start3 = time.time()
    response3 = await rag_service.ask_question(request3)
    time3 = time.time() - start3
    
    logger.info(f"⏱️  Temps: {time3:.2f} secondes")
    
    if time3 < 1.0:
        logger.info(f"✅ Cache fonctionne même avec variations! ({time3:.3f}s)")
    else:
        logger.warning(f"⚠️  Cache n'a pas détecté la question similaire ({time3:.2f}s)")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_history_cache())
