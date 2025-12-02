"""
Test de la persistence de l'historique dans un fichier JSON
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.rag_service import RAGService
from app.models import QuestionRequest
import asyncio
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_history_persistence():
    """Test que l'historique est sauvegardé et rechargé depuis le fichier JSON"""
    logger.info("=" * 80)
    logger.info("TEST DE LA PERSISTENCE DE L'HISTORIQUE")
    logger.info("=" * 80)
    
    # Supprimer le fichier d'historique s'il existe
    history_file = Path("conversation_history.json")
    if history_file.exists():
        history_file.unlink()
        logger.info("🗑️  Ancien fichier d'historique supprimé")
    
    # Créer le premier service et poser une question
    logger.info("\n📝 ÉTAPE 1: Créer le service et poser une question")
    rag_service1 = RAGService()
    await rag_service1.initialize()
    
    question = "Test de persistence"
    request = QuestionRequest(question=question, context_limit=5)
    response = await rag_service1.ask_question(request)
    
    logger.info(f"✅ Question posée: {question}")
    logger.info(f"📊 Historique contient: {len(rag_service1.conversation_history)} entrée(s)")
    
    # Vérifier que le fichier JSON existe
    logger.info("\n📁 ÉTAPE 2: Vérifier que le fichier JSON a été créé")
    if history_file.exists():
        logger.info(f"✅ Fichier créé: {history_file}")
        
        # Lire le contenu
        with open(history_file, 'r', encoding='utf-8') as f:
            saved_history = json.load(f)
        logger.info(f"📊 Fichier contient: {len(saved_history)} entrée(s)")
        
        # Afficher un extrait
        if saved_history:
            first_entry = saved_history[0]
            logger.info(f"📝 Première entrée:")
            logger.info(f"   - Question: {first_entry['question']}")
            logger.info(f"   - Réponse: {first_entry['answer'][:100]}...")
            logger.info(f"   - Confiance: {first_entry['confidence_score']}")
    else:
        logger.error("❌ Fichier JSON non créé!")
        return
    
    # Créer un nouveau service (simule un redémarrage)
    logger.info("\n🔄 ÉTAPE 3: Créer un nouveau service (simule redémarrage)")
    rag_service2 = RAGService()
    await rag_service2.initialize()
    
    logger.info(f"📊 Historique rechargé: {len(rag_service2.conversation_history)} entrée(s)")
    
    if len(rag_service2.conversation_history) > 0:
        logger.info("✅ L'historique a été rechargé depuis le fichier!")
        
        # Vérifier que la question est dans l'historique
        if rag_service2.conversation_history[0]['question'] == question:
            logger.info("✅ La question est bien présente dans l'historique rechargé")
        else:
            logger.warning("⚠️  La question ne correspond pas")
    else:
        logger.error("❌ L'historique n'a pas été rechargé!")
        return
    
    # Tester le cache avec le service rechargé
    logger.info("\n⚡ ÉTAPE 4: Tester le cache avec le service rechargé")
    import time
    start = time.time()
    response2 = await rag_service2.ask_question(request)
    elapsed = time.time() - start
    
    if elapsed < 1.0:
        logger.info(f"✅ Cache fonctionne après rechargement! ({elapsed:.3f}s)")
    else:
        logger.warning(f"⚠️  Cache n'a pas fonctionné ({elapsed:.2f}s)")
    
    # Afficher le contenu du fichier JSON
    logger.info("\n📄 CONTENU DU FICHIER JSON:")
    logger.info("=" * 80)
    with open(history_file, 'r', encoding='utf-8') as f:
        content = f.read()
    logger.info(content[:500] + "..." if len(content) > 500 else content)
    logger.info("=" * 80)
    
    logger.info("\n✅ TEST TERMINÉ")

if __name__ == "__main__":
    asyncio.run(test_history_persistence())
