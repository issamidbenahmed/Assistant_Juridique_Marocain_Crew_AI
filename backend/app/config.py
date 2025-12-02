import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"  # ou "mistral", "codellama"
    
    # Gemini Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-pro"
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "legal_documents"
    
    # Data Configuration
    DATA_DIRECTORY: str = "../data"
    
    # CrewAI Multi-agent Configuration
    ENABLE_CREW_AGENTS: bool = True
    CREW_AGENT_TOP_K: int = 3
    CREW_MIN_SCORE: float = 0.05
    CREW_MODEL: Optional[str] = None  # fallback vers ollama/<OLLAMA_MODEL>
    CREW_TEMPERATURE: float = 0.2
    CREW_SUPERVISOR_TEMPERATURE: float = 0.15
    CREW_MAX_TOKENS: int = 800  # Tokens maximum par réponse
    CREW_TIMEOUT: int = 3600  # Timeout en secondes (1 heure)
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # CORS Configuration
    CORS_ORIGINS: list = ["http://localhost:4200", "http://127.0.0.1:4200", "http://localhost:51885", "http://127.0.0.1:51885"]
    
    class Config:
        env_file = ".env"

settings = Settings()
