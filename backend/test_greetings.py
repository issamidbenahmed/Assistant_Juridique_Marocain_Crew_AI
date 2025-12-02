"""
Test du handler de salutations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.rag_service import RAGService
from app.models import QuestionRequest
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_greetings():
    """Test que les salutations sont détectées et répondues instantanément"""
    logger.info("=" * 80)
    logger.info("TEST DU HANDLER DE SALUTATIONS")
    logger.info("=" * 80)
    
    # Initialiser le service
    rag_service = RAGService()
    success = await rag_service.initialize()
    
    if not success:
        logger.error("❌ Échec de l'initialisation")
        return
    
    logger.info("✅ Service initialisé\n")
    
    # Liste de salutations à tester
    test_cases = [
        "Bonjour",
        "bonjour",
        "Salut",
        "Hello",
        "Hi",
        "Salam",
        "Merci",
        "Au revoir",
        "Bye",
        "Bonsoir",
        "Good morning",
        "شكرا",
        "Bonjour, comment ça va ?",
        "Qu'est-ce que la TVA au Maroc?",  # Question juridique (ne devrait PAS être détectée)
    ]
    
    for test_input in test_cases:
        logger.info(f"📝 Test: '{test_input}'")
        request = QuestionRequest(question=test_input, context_limit=5)
        response = await rag_service.ask_question(request)
        
        # Vérifier si c'est une réponse de salutation (instantanée)
        if response.processing_time < 0.1:
            logger.info(f"   ✅ Salutation détectée - Réponse: {response.answer[:80]}...")
        else:
            logger.info(f"   🔍 Question juridique traitée ({response.processing_time:.2f}s)")
        
        logger.info("")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_greetings())
