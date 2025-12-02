import pytest
import requests
from typing import Dict, Any

class TestAssistantAPI:
    """Tests pour l'API de l'assistant juridique"""
    
    BASE_URL = "http://localhost:8000"
    
    def test_health_endpoint(self):
        """Test du endpoint de santé"""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_status_endpoint(self):
        """Test du endpoint de statut"""
        response = requests.get(f"{self.BASE_URL}/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "is_initialized" in data["status"]
    
    def test_ask_endpoint(self):
        """Test du endpoint de questions"""
        question = {
            "question": "Test question",
            "context_limit": 3
        }
        response = requests.post(f"{self.BASE_URL}/ask", json=question)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence_score" in data
    
    def test_history_endpoint(self):
        """Test du endpoint d'historique"""
        response = requests.get(f"{self.BASE_URL}/history")
        assert response.status_code == 200
        data = response.json()
        assert "history" in data
        assert isinstance(data["history"], list)
    
    def test_reload_data_endpoint(self):
        """Test du endpoint de rechargement"""
        response = requests.post(f"{self.BASE_URL}/reload-data")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "documents_processed" in data

if __name__ == "__main__":
    pytest.main([__file__])
