#!/usr/bin/env python
"""Script de test pour diagnostiquer l'initialisation CrewAI"""
import logging
import sys

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    logger.info("=== Test 1: Import des modules ===")
    from crewai import Agent, Task, Crew, Process, LLM
    logger.info("✓ Imports CrewAI OK")
    
    logger.info("\n=== Test 2: Configuration ===")
    from app.config import settings
    logger.info(f"ENABLE_CREW_AGENTS: {settings.ENABLE_CREW_AGENTS}")
    logger.info(f"OLLAMA_BASE_URL: {settings.OLLAMA_BASE_URL}")
    logger.info(f"OLLAMA_MODEL: {settings.OLLAMA_MODEL}")
    logger.info(f"CREW_MODEL: {settings.CREW_MODEL}")
    logger.info(f"CREW_TEMPERATURE: {settings.CREW_TEMPERATURE}")
    
    logger.info("\n=== Test 3: Création LLM ===")
    model_name = settings.CREW_MODEL or f"ollama/{settings.OLLAMA_MODEL}"
    logger.info(f"Tentative de création LLM avec model: {model_name}")
    
    kwargs = {
        "model": model_name,
        "temperature": settings.CREW_TEMPERATURE,
        "max_tokens": 1000,
    }
    if model_name.startswith("ollama/"):
        kwargs["base_url"] = settings.OLLAMA_BASE_URL
    
    logger.info(f"Paramètres LLM: {kwargs}")
    llm = LLM(**kwargs)
    logger.info(f"✓ LLM créé: {llm}")
    
    logger.info("\n=== Test 4: Création Agent ===")
    agent = Agent(
        role="Test Agent",
        goal="Tester l'initialisation",
        backstory="Agent de test",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
    logger.info(f"✓ Agent créé: {agent}")
    
    logger.info("\n=== Test 5: VectorStore ===")
    from app.services.vector_store import VectorStore
    vs = VectorStore()
    logger.info(f"✓ VectorStore créé")
    
    logger.info("\n=== Test 6: CrewMultiAgentService ===")
    from app.services.crew_agent_service import CrewMultiAgentService
    cms = CrewMultiAgentService(vs, ['dataset1.csv', 'dataset2.csv', 'dataset3.csv'])
    logger.info(f"✓ CrewMultiAgentService créé")
    logger.info(f"is_available: {cms.is_available}")
    logger.info(f"Nombre d'agents dataset: {len(cms.dataset_agents)}")
    logger.info(f"Superviseur: {cms.supervisor_agent}")
    
    if cms.is_available:
        logger.info("\n✅ SUCCÈS: CrewAI est correctement initialisé!")
    else:
        logger.error("\n❌ ÉCHEC: CrewAI n'est pas disponible")
        sys.exit(1)
        
except Exception as e:
    logger.exception(f"\n❌ ERREUR: {e}")
    sys.exit(1)
