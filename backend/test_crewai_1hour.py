"""
Test CrewAI avec timeout de 1 heure
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.services.crew_agent_service import CrewMultiAgentService
from app.services.vector_store import VectorStore
from app.services.csv_processor import CSVProcessor
from app.config import settings
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_crewai_with_long_timeout():
    """Test CrewAI avec timeout de 1 heure"""
    logger.info("=" * 80)
    logger.info("TEST CREWAI AVEC TIMEOUT DE 1 HEURE")
    logger.info("=" * 80)
    logger.info(f"Timeout configuré: {settings.CREW_TIMEOUT} secondes ({settings.CREW_TIMEOUT/60:.1f} minutes)")
    
    # Initialisation
    csv_processor = CSVProcessor(settings.DATA_DIRECTORY)
    documents = csv_processor.load_all_csv_files()
    logger.info(f"Documents chargés: {len(documents)}")
    
    vector_store = VectorStore()
    vector_store.clear_collection()
    vector_store.add_documents(documents)
    
    dataset_files = sorted({doc.source_file for doc in documents})
    logger.info(f"Datasets: {dataset_files}")
    
    # Créer le service CrewAI
    crew_service = CrewMultiAgentService(
        vector_store=vector_store,
        dataset_files=dataset_files
    )
    
    if not crew_service.is_available:
        logger.error("❌ CrewAI n'est pas disponible!")
        return
    
    logger.info("✅ CrewAI initialisé avec succès")
    logger.info(f"Nombre d'agents: {len(crew_service.dataset_agents)}")
    
    # Question de test
    question = "Qu'est-ce que la TVA au Maroc?"
    logger.info(f"\n📝 Question: {question}")
    logger.info("⏳ Lancement de CrewAI (peut prendre jusqu'à 1 heure)...")
    logger.info("Les 3 agents vont travailler en parallèle (async_execution=True)")
    
    start_time = time.time()
    
    try:
        result = crew_service.run(question=question, context_limit=5)
        elapsed = time.time() - start_time
        
        if result:
            logger.info("=" * 80)
            logger.info("✅ SUCCÈS!")
            logger.info(f"⏱️  Temps écoulé: {elapsed:.1f} secondes ({elapsed/60:.1f} minutes)")
            logger.info(f"📊 Réponse: {result.get('answer', 'N/A')[:200]}...")
            logger.info(f"🎯 Confiance: {result.get('confidence', 0)}")
            logger.info(f"📚 Sources: {len(result.get('sources', []))}")
            logger.info("=" * 80)
        else:
            logger.warning("⚠️  Aucun résultat retourné par CrewAI")
            logger.info(f"⏱️  Temps écoulé: {elapsed:.1f} secondes")
            
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("=" * 80)
        logger.error(f"❌ ERREUR après {elapsed:.1f} secondes ({elapsed/60:.1f} minutes)")
        logger.error(f"Type: {type(e).__name__}")
        logger.error(f"Message: {str(e)}")
        logger.error("=" * 80)
        raise

if __name__ == "__main__":
    test_crewai_with_long_timeout()
