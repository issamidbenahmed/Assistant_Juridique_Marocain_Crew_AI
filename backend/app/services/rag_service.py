import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import uuid
import json
from pathlib import Path

from ..models import LegalDocument, QuestionRequest, AnswerResponse, Source, ReloadResponse
from .csv_processor import CSVProcessor
from .vector_store import VectorStore
from .llm_service import LLMService
from .crew_agent_service import CrewMultiAgentService
from ..config import settings

logger = logging.getLogger(__name__)

class RAGService:
    """Service principal pour le pipeline RAG"""
    
    HISTORY_FILE = "conversation_history.json"
    
    def __init__(self):
        self.csv_processor = CSVProcessor(settings.DATA_DIRECTORY)
        self.vector_store = VectorStore()
        self.llm_service = LLMService()
        self.multi_agent_service: Optional[CrewMultiAgentService] = None
        self.conversation_history: List[Dict[str, Any]] = []
        self._is_initialized = False
        self._load_history_from_file()  # Charger l'historique au démarrage
    
    async def initialize(self) -> bool:
        """Initialise le service RAG"""
        try:
            logger.info("Initialisation du service RAG...")
            documents = self.csv_processor.load_all_csv_files()
            if not documents:
                logger.error("Aucun document chargé")
                return False
            
            # Clear et reload
            self.vector_store.clear_collection()
            success = self.vector_store.add_documents(documents)
            if not success:
                logger.error("Échec de l'ajout des documents à la base vectorielle")
                return False
            
            logger.info("Vérification post-indexation : collection stats -> %s", self.vector_store.get_collection_stats())
            dataset_files = sorted({doc.source_file for doc in documents})
            self._initialize_multi_agent(dataset_files)
            self._is_initialized = True
            logger.info(f"Service RAG initialisé avec {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.exception(f"Erreur lors de l'initialisation du service RAG: {str(e)}")
            return False
    
    async def ask_question(self, request: QuestionRequest) -> AnswerResponse:
        """Traite une question utilisateur"""
        if not self._is_initialized:
            raise Exception("Service RAG non initialisé")
        
        start_time = time.time()
        
        try:
            # Vérifier si c'est une salutation ou message non-juridique
            greeting_response = self._handle_greetings(request.question)
            if greeting_response:
                logger.info(f"👋 Salutation détectée - réponse directe")
                return greeting_response
            
            # Vérifier si la question existe déjà dans l'historique (cache)
            cached_response = self._check_history_cache(request.question)
            if cached_response:
                logger.info(f"✅ Question trouvée dans l'historique (cache) - réponse instantanée")
                return cached_response
            
            logger.info(f"📥 Recherche de documents pertinents pour: {request.question}")
            multi_agent_result = None
            if self.multi_agent_service and self.multi_agent_service.is_available:
                logger.info("🤖 Tentative d'utilisation de CrewAI multi-agent...")
                multi_agent_result = self.multi_agent_service.run(
                    question=request.question,
                    context_limit=request.context_limit
                )
                if multi_agent_result:
                    logger.info("✅ Réponse générée via CrewAI multi-agent!")
                else:
                    logger.info("⚠️  CrewAI n'a pas pu générer de réponse, fallback vers RAG classique")
            
            if multi_agent_result and multi_agent_result.get("answer"):
                sources = multi_agent_result.get("sources") or []
                if not sources:
                    sources = self.vector_store.search_similar_documents(
                        query=request.question,
                        n_results=max(5, request.context_limit),
                        min_score=0.0  # Accepter tous les résultats
                    )
                answer_text = multi_agent_result["answer"]
                confidence_score = float(multi_agent_result.get("confidence", 0.75))
            else:
                sources = self.vector_store.search_similar_documents(
                    query=request.question,
                    n_results=max(5, request.context_limit),
                    min_score=0.0  # Accepter tous les résultats
                )
                
                if not sources:
                    logger.warning("Aucun source trouvé — renvoi message utilisateur (confiance 0)")
                    return AnswerResponse(
                        answer="Je n'ai pas trouvé d'informations pertinentes dans les documents juridiques disponibles pour répondre à votre question.",
                        sources=[],
                        confidence_score=0.0,
                        processing_time=time.time() - start_time,
                        timestamp=datetime.now()
                    )
                
                logger.info(f"{len(sources)} source(s) récupérées — top1: {sources[0].source_file} (score={sources[0].relevance_score})")
                ollama_result = self.llm_service.generate_answer_with_ollama(
                    question=request.question,
                    sources=sources,
                    context_limit=request.context_limit
                )
                answer_text = ollama_result["answer"]
                confidence_score = ollama_result["confidence_score"]
            
            if self.llm_service.is_gemini_available():
                validation_result = self.llm_service.validate_answer_with_gemini(
                    question=request.question,
                    answer=answer_text,
                    sources=sources
                )
                answer_text = validation_result.get("validated_answer", answer_text)
                confidence_score = max(confidence_score, validation_result.get("validation_score", confidence_score))
            
            response = AnswerResponse(
                answer=answer_text,
                sources=sources,
                confidence_score=confidence_score,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
            self._save_to_history(request.question, response)
            logger.info(f"Réponse générée (confidence={confidence_score:.2f})")
            return response
            
        except Exception as e:
            logger.exception(f"Erreur lors du traitement de la question: {str(e)}")
            return AnswerResponse(
                answer=f"Une erreur s'est produite lors du traitement de votre question: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    async def reload_data(self) -> ReloadResponse:
        """Recharge les données CSV et met à jour la base vectorielle"""
        start_time = time.time()
        
        try:
            logger.info("Rechargement des données...")
            documents = self.csv_processor.load_all_csv_files()
            if not documents:
                raise Exception("Aucun document trouvé")
            
            self.vector_store.clear_collection()
            success = self.vector_store.add_documents(documents)
            if not success:
                raise Exception("Échec de l'ajout des documents à la base vectorielle")
            
            dataset_files = sorted({doc.source_file for doc in documents})
            self._initialize_multi_agent(dataset_files)
            self._is_initialized = True
            processing_time = time.time() - start_time
            logger.info(f"Données rechargées avec succès: {len(documents)} documents")
            return ReloadResponse(
                message=f"Données rechargées avec succès. {len(documents)} documents traités.",
                documents_processed=len(documents),
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.exception(f"Erreur lors du rechargement des données: {str(e)}")
            return ReloadResponse(
                message=f"Erreur lors du rechargement: {str(e)}",
                documents_processed=0,
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    def get_conversation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.conversation_history[-limit:]
    
    def clear_history(self) -> bool:
        try:
            self.conversation_history.clear()
            logger.info("Historique des conversations vidé")
            return True
        except Exception as e:
            logger.exception(f"Erreur lors du vidage de l'historique: {str(e)}")
            return False
    
    def _handle_greetings(self, question: str) -> Optional[AnswerResponse]:
        """Détecte et répond aux salutations et messages non-juridiques"""
        try:
            normalized = question.strip().lower()
            
            # Liste des salutations et messages courants
            greetings = [
                "bonjour", "bonsoir", "salut", "hello", "hi", "hey", "coucou",
                "bonne journée", "bonne soirée", "bon matin", "good morning",
                "good evening", "good afternoon", "salam", "salam alaykoum",
                "merci", "thank you", "thanks", "شكرا", "au revoir", "bye",
                "à bientôt", "goodbye"
            ]
            
            # Vérifier si c'est EXACTEMENT une salutation (pas juste contient)
            # On vérifie que le message est court ET contient une salutation
            if normalized in greetings:
                for greeting in greetings:
                    if greeting == normalized:
                        response_text = (
                            "Bonjour ! Je suis votre assistant juridique marocain. "
                            "Je peux vous aider à répondre à vos questions sur le droit marocain. "
                            "N'hésitez pas à me poser une question juridique !"
                        )
                        
                        # Réponses spécifiques
                        if any(word in normalized for word in ["merci", "thank", "شكرا"]):
                            response_text = "De rien ! N'hésitez pas si vous avez d'autres questions juridiques."
                        elif any(word in normalized for word in ["au revoir", "bye", "goodbye", "à bientôt"]):
                            response_text = "Au revoir ! N'hésitez pas à revenir si vous avez des questions juridiques."
                        
                        return AnswerResponse(
                            answer=response_text,
                            sources=[],
                            confidence_score=1.0,
                            processing_time=0.0,
                            timestamp=datetime.now()
                        )
            
            return None
            
        except Exception as e:
            logger.exception(f"Erreur lors de la détection des salutations: {str(e)}")
            return None
    
    def _check_history_cache(self, question: str) -> Optional[AnswerResponse]:
        """Vérifie si la question existe déjà dans l'historique et retourne la réponse en cache"""
        try:
            # Normaliser la question pour la comparaison
            normalized_question = question.strip().lower()
            
            # Chercher dans l'historique (du plus récent au plus ancien)
            for entry in reversed(self.conversation_history):
                if entry["question"].strip().lower() == normalized_question:
                    # Reconstruire les sources
                    sources = [
                        Source(
                            doc=s["doc"],
                            titre=s["titre"],
                            chapitre=s["chapitre"],
                            article=s["article"],
                            contenu=s["contenu"],
                            source_file=s["source_file"],
                            relevance_score=s["relevance_score"]
                        )
                        for s in entry["sources"]
                    ]
                    
                    # Retourner la réponse en cache
                    return AnswerResponse(
                        answer=entry["answer"],
                        sources=sources,
                        confidence_score=entry["confidence_score"],
                        processing_time=0.0,  # Instantané
                        timestamp=datetime.now()
                    )
            
            return None
            
        except Exception as e:
            logger.exception(f"Erreur lors de la vérification du cache: {str(e)}")
            return None
    
    def _save_to_history(self, question: str, response: AnswerResponse):
        try:
            history_entry = {
                "id": str(uuid.uuid4()),
                "question": question,
                "answer": response.answer,
                "sources": [
                    {
                        "doc": source.doc,
                        "titre": source.titre,
                        "chapitre": source.chapitre,
                        "article": source.article,
                        "contenu": source.contenu[:200] + "..." if len(source.contenu) > 200 else source.contenu,
                        "source_file": source.source_file,
                        "relevance_score": source.relevance_score
                    }
                    for source in response.sources
                ],
                "confidence_score": response.confidence_score,
                "timestamp": response.timestamp.isoformat()
            }
            
            self.conversation_history.append(history_entry)
            if len(self.conversation_history) > 100:
                self.conversation_history = self.conversation_history[-100:]
            
            # Persister l'historique après chaque ajout
            self._persist_history_to_file()
                
        except Exception as e:
            logger.exception(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
    
    def _load_history_from_file(self):
        """Charge l'historique depuis le fichier JSON au démarrage"""
        try:
            history_path = Path(self.HISTORY_FILE)
            if history_path.exists():
                with open(history_path, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                logger.info(f"✅ Historique chargé: {len(self.conversation_history)} conversations")
            else:
                logger.info("Aucun fichier d'historique trouvé, démarrage avec historique vide")
        except Exception as e:
            logger.exception(f"Erreur lors du chargement de l'historique: {str(e)}")
            self.conversation_history = []
    
    def _persist_history_to_file(self):
        """Sauvegarde l'historique dans un fichier JSON"""
        try:
            history_path = Path(self.HISTORY_FILE)
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            logger.debug(f"Historique sauvegardé: {len(self.conversation_history)} conversations")
        except Exception as e:
            logger.exception(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
    
    def get_service_status(self) -> Dict[str, Any]:
        return {
            "is_initialized": self._is_initialized,
            "ollama_available": self.llm_service.is_ollama_available(),
            "gemini_available": self.llm_service.is_gemini_available(),
            "crew_agents_enabled": bool(self.multi_agent_service and self.multi_agent_service.is_available),
            "vector_store_stats": self.vector_store.get_collection_stats(),
            "csv_stats": self.csv_processor.get_statistics(),
            "history_count": len(self.conversation_history)
        }

    def _initialize_multi_agent(self, dataset_files: List[str]):
        """Initialise ou met à jour le service CrewAI."""
        if not settings.ENABLE_CREW_AGENTS or not dataset_files:
            return
        if not self.multi_agent_service:
            self.multi_agent_service = CrewMultiAgentService(
                vector_store=self.vector_store,
                dataset_files=dataset_files
            )
        else:
            self.multi_agent_service.update_datasets(dataset_files)
