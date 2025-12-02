import ollama
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import logging
import time
from ..models import Source
from ..config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """Service pour gérer Ollama et Gemini"""
    
    def __init__(self):
        self.ollama_client = None
        self.gemini_model = None
        self._initialize_ollama()
        self._initialize_gemini()
    
    def _initialize_ollama(self):
        """Initialise Ollama"""
        try:
            self.ollama_client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            # Vérifier la présence du modèle
            try:
                models = self.ollama_client.list()
                model_names = [m.get('name') if isinstance(m, dict) else m for m in models.get('models', [])]
            except Exception:
                # fallback: ignorer lister si incompatible
                model_names = []
            
            if settings.OLLAMA_MODEL and settings.OLLAMA_MODEL not in model_names:
                logger.warning(f"Modèle {settings.OLLAMA_MODEL} non trouvé localement (liste: {model_names}). On tente pull (si supporté).")
                try:
                    self.ollama_client.pull(settings.OLLAMA_MODEL)
                except Exception as e:
                    logger.warning(f"Impossible de pull le modèle {settings.OLLAMA_MODEL}: {e}")
            
            logger.info(f"Ollama initialisé (base_url={settings.OLLAMA_BASE_URL}, model={settings.OLLAMA_MODEL})")
        except Exception as e:
            logger.exception(f"Erreur lors de l'initialisation d'Ollama: {str(e)}")
            self.ollama_client = None
    
    def _initialize_gemini(self):
        """Initialise Gemini"""
        try:
            if settings.GEMINI_API_KEY:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_model = genai.GenerativeModel(settings.GEMINI_MODEL)
                logger.info(f"Gemini initialisé avec le modèle {settings.GEMINI_MODEL}")
            else:
                logger.warning("Clé API Gemini non fournie")
                self.gemini_model = None
        except Exception as e:
            logger.exception(f"Erreur lors de l'initialisation de Gemini: {str(e)}")
            self.gemini_model = None
    
    def generate_answer_with_ollama(
        self, 
        question: str, 
        sources: List[Source],
        context_limit: int = 5
    ) -> Dict[str, Any]:
        """Génère une réponse en utilisant Ollama"""
        if not self.ollama_client:
            raise Exception("Ollama n'est pas initialisé")
        
        try:
            context = self._prepare_context(sources[:context_limit])
            prompt = self._create_legal_prompt(question, context)
            start_time = time.time()
            
            # Appeler Ollama - wrapper compatible selon version
            try:
                response = self.ollama_client.generate(
                    model=settings.OLLAMA_MODEL,
                    prompt=prompt,
                    options={
                        "temperature": 0.2,
                        "top_p": 0.9,
                        "max_tokens": 1000
                    }
                )
                # Plusieurs versions de client Ollama: normaliser l'accès au texte
                if isinstance(response, dict):
                    # chercher keys possibles
                    ans = response.get("response") or response.get("output") or ""
                    if isinstance(ans, list) and ans:
                        # extraire texte si structure différente
                        first = ans[0]
                        answer_text = first.get("content") if isinstance(first, dict) else str(first)
                    elif isinstance(ans, str):
                        answer_text = ans
                    else:
                        answer_text = str(response)
                else:
                    # fallback
                    answer_text = str(response)
            except Exception as e:
                logger.exception(f"Erreur lors de l'appel à Ollama.generate: {e}")
                raise
            
            processing_time = time.time() - start_time
            answer = (answer_text or "").strip()
            confidence_score = self._calculate_confidence_score(answer, sources)
            
            return {
                "answer": answer,
                "confidence_score": confidence_score,
                "processing_time": processing_time,
                "model_used": settings.OLLAMA_MODEL,
                "sources_used": len(sources[:context_limit])
            }
            
        except Exception as e:
            logger.exception(f"Erreur lors de la génération avec Ollama: {str(e)}")
            raise
    
    def validate_answer_with_gemini(
        self, 
        question: str, 
        answer: str, 
        sources: List[Source]
    ) -> Dict[str, Any]:
        """Valide et améliore la réponse avec Gemini"""
        if not self.gemini_model:
            logger.warning("Gemini non disponible pour la validation")
            return {
                "validated_answer": answer,
                "validation_score": 0.5,
                "improvements": []
            }
        
        try:
            validation_prompt = self._create_validation_prompt(question, answer, sources)
            response = self.gemini_model.generate_content(validation_prompt)
            validation_result = response.text.strip() if hasattr(response, "text") else str(response)
            
            try:
                import json
                result = json.loads(validation_result)
                return {
                    "validated_answer": result.get("improved_answer", answer),
                    "validation_score": result.get("confidence_score", 0.5),
                    "improvements": result.get("improvements", []),
                    "validation_notes": result.get("notes", "")
                }
            except Exception:
                return {
                    "validated_answer": validation_result,
                    "validation_score": 0.7,
                    "improvements": ["Réponse validée par Gemini"],
                    "validation_notes": validation_result
                }
                
        except Exception as e:
            logger.exception(f"Erreur lors de la validation avec Gemini: {str(e)}")
            return {
                "validated_answer": answer,
                "validation_score": 0.5,
                "improvements": [],
                "error": str(e)
            }
    
    def _prepare_context(self, sources: List[Source]) -> str:
        """Prépare le contexte à partir des sources"""
        context_parts = []
        for i, source in enumerate(sources, 1):
            context_part = f"Source {i}:\n"
            if source.doc:
                context_part += f"Document: {source.doc}\n"
            if source.titre:
                context_part += f"Titre: {source.titre}\n"
            if source.chapitre:
                context_part += f"Chapitre: {source.chapitre}\n"
            if source.article:
                context_part += f"Article: {source.article}\n"
            context_part += f"Contenu: {source.contenu}\n"
            context_part += f"Source file: {source.source_file}\n"
            context_parts.append(context_part)
        return "\n\n".join(context_parts)
    
    def _create_legal_prompt(self, question: str, context: str) -> str:
        """Crée le prompt pour l'assistant juridique"""
        return f"""Tu es un assistant juridique marocain spécialisé. Tu dois répondre aux questions en te basant UNIQUEMENT sur les sources juridiques fournies ci-dessous.

IMPORTANT:
- Réponds uniquement en français
- Cite toujours la source exacte (document, article, etc.)
- Ne donne pas d'avis personnel ou d'interprétation
- Si l'information n'est pas dans les sources, dis-le clairement
- Sois précis et professionnel

QUESTION: {question}

SOURCES JURIDIQUES:
{context}

RÉPONSE:"""
    
    def _create_validation_prompt(self, question: str, answer: str, sources: List[Source]) -> str:
        """Crée le prompt de validation pour Gemini"""
        context = self._prepare_context(sources)
        return f"""Tu es un expert juridique marocain. Valide et améliore cette réponse d'assistant juridique.

QUESTION: {question}

RÉPONSE À VALIDER: {answer}

SOURCES UTILISÉES:
{context}

Évalue la réponse selon ces critères:
1. Exactitude juridique
2. Correspondance avec les sources
3. Clarté et professionnalisme
4. Citations appropriées

Réponds au format JSON:
{{
    "improved_answer": "Réponse améliorée si nécessaire",
    "confidence_score": 0.0,
    "improvements": ["liste des améliorations"],
    "notes": "commentaires additionnels"
}}"""
    
    def _calculate_confidence_score(self, answer: str, sources: List[Source]) -> float:
        """Calcule un score de confiance basique"""
        if not answer or not sources:
            return 0.0
        
        score = 0.5
        if any(keyword in answer.lower() for keyword in ["article", "loi", "décret", "code"]):
            score += 0.2
        if any((source.doc or "").lower() in answer.lower() for source in sources):
            score += 0.2
        if 50 <= len(answer) <= 2000:
            score += 0.1
        
        return min(score, 1.0)
    
    def is_ollama_available(self) -> bool:
        return self.ollama_client is not None
    
    def is_gemini_available(self) -> bool:
        return self.gemini_model is not None
