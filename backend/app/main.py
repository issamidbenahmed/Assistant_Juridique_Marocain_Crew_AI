from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import os
from datetime import datetime
from typing import List, Dict, Any

from .config import settings

# Configuration des timeouts pour litellm/httpx
os.environ["LITELLM_REQUEST_TIMEOUT"] = str(settings.CREW_TIMEOUT)
os.environ["HTTPX_TIMEOUT"] = str(settings.CREW_TIMEOUT)
from .models import (
    QuestionRequest, 
    AnswerResponse, 
    ReloadResponse, 
    ErrorResponse,
    HistoryEntry
)
from .services.rag_service import RAGService

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Assistant Juridique Marocain",
    description="API pour l'assistant juridique marocain avec RAG, Ollama et Gemini",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instance globale du service RAG
rag_service: RAGService = None

@app.on_event("startup")
async def startup_event():
    """Initialise le service RAG au démarrage"""
    global rag_service
    try:
        rag_service = RAGService()
        success = await rag_service.initialize()
        if success:
            logger.info("Service RAG initialisé avec succès")
        else:
            logger.error("Échec de l'initialisation du service RAG")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {str(e)}")

def get_rag_service() -> RAGService:
    """Dependency pour obtenir le service RAG"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Service RAG non initialisé")
    return rag_service

@app.get("/")
async def root():
    """Endpoint de base"""
    return {
        "message": "Assistant Juridique Marocain API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Vérification de l'état de santé de l'API"""
    try:
        if rag_service is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "Service RAG non initialisé"}
            )
        
        status = rag_service.get_service_status()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": status
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy", 
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest,
    service: RAGService = Depends(get_rag_service)
):
    """Pose une question à l'assistant juridique"""
    try:
        logger.info(f"Nouvelle question reçue: {request.question}")
        
        if not request.question.strip():
            raise HTTPException(
                status_code=400, 
                detail="La question ne peut pas être vide"
            )
        
        response = await service.ask_question(request)
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la question: {str(e)}"
        )

@app.post("/reload-data", response_model=ReloadResponse)
async def reload_data(
    service: RAGService = Depends(get_rag_service)
):
    """Recharge les données CSV et met à jour la base vectorielle"""
    try:
        logger.info("Demande de rechargement des données")
        response = await service.reload_data()
        return response
        
    except Exception as e:
        logger.error(f"Erreur lors du rechargement des données: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du rechargement des données: {str(e)}"
        )

@app.get("/history")
async def get_history(
    limit: int = 50,
    service: RAGService = Depends(get_rag_service)
):
    """Récupère l'historique des conversations"""
    try:
        history = service.get_conversation_history(limit)
        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'historique: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération de l'historique: {str(e)}"
        )

@app.delete("/history")
async def clear_history(
    service: RAGService = Depends(get_rag_service)
):
    """Vide l'historique des conversations"""
    try:
        success = service.clear_history()
        if success:
            return {"message": "Historique vidé avec succès"}
        else:
            raise HTTPException(
                status_code=500,
                detail="Erreur lors du vidage de l'historique"
            )
            
    except Exception as e:
        logger.error(f"Erreur lors du vidage de l'historique: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du vidage de l'historique: {str(e)}"
        )

@app.get("/status")
async def get_status(
    service: RAGService = Depends(get_rag_service)
):
    """Retourne le statut détaillé du service"""
    try:
        status = service.get_service_status()
        return {
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du statut: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération du statut: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire d'exceptions global"""
    logger.error(f"Exception non gérée: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne du serveur",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
