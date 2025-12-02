from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class LegalDocument(BaseModel):
    """Modèle pour un document juridique"""
    doc: str
    titre: Optional[str] = None
    chapitre: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    contenu: str
    pages: Optional[str] = None
    index: Optional[str] = None
    source_file: str

class QuestionRequest(BaseModel):
    """Modèle pour une question utilisateur"""
    question: str
    context_limit: int = 5  # Nombre de documents contextuels à récupérer

class Source(BaseModel):
    """Modèle pour une source de réponse"""
    doc: str
    titre: Optional[str] = None
    chapitre: Optional[str] = None
    article: Optional[str] = None
    contenu: str
    pages: Optional[str] = None
    source_file: str
    relevance_score: float

class AnswerResponse(BaseModel):
    """Modèle pour une réponse de l'assistant"""
    answer: str
    sources: List[Source]
    confidence_score: float
    processing_time: float
    timestamp: datetime

class HistoryEntry(BaseModel):
    """Modèle pour un historique de conversation"""
    id: str
    question: str
    answer: str
    sources: List[Source]
    timestamp: datetime

class ReloadResponse(BaseModel):
    """Modèle pour la réponse de rechargement des données"""
    message: str
    documents_processed: int
    processing_time: float
    timestamp: datetime

class ErrorResponse(BaseModel):
    """Modèle pour les erreurs"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime