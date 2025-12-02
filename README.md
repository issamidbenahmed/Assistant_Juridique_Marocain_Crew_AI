# 🇲🇦 Assistant Juridique Marocain

Assistant juridique intelligent basé sur l'IA qui répond aux questions selon la législation marocaine avec sources traçables.

## ✨ Fonctionnalités

- **RAG (Retrieval-Augmented Generation)** : Recherche sémantique dans les documents juridiques
- **CrewAI Multi-Agent** : 3 agents spécialisés + 1 superviseur pour analyse parallèle
- **Sources Traçables** : Citations précises avec articles, lois et scores de pertinence
- **Interface Moderne** : Landing page 3D interactive avec Spline, design dark mode
- **Historique Persistant** : Sauvegarde automatique des conversations
- **Cache Intelligent** : Réponses rapides pour questions similaires
- **LLM Hybride** : Ollama (local)

## 🏗️ Architecture

```
assistjur/
├── backend/                    # API FastAPI
│   ├── app/
│   │   ├── main.py           # Point d'entrée
│   │   ├── config.py         # Configuration
│   │   ├── models/           # Modèles Pydantic
│   │   ├── services/         # Services métier
│   │   │   ├── csv_processor.py    # Traitement CSV
│   │   │   ├── vector_store.py     # ChromaDB
│   │   │   ├── rag_service.py      # Pipeline RAG
│   │   │   └── llm_service.py      # Ollama
│   │   └── api/              # Endpoints
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                  # Interface Angular
│   ├── src/app/
│   │   ├── components/        # Composants UI
│   │   ├── services/          # Services API
│   │   ├── models/            # Modèles TypeScript
│   │   └── app.component.*    # Composant principal
│   ├── package.json
│   └── angular.json
├── data/                      # Documents CSV
│   ├── dataset1.csv
│   ├── dataset2.csv
│   └── dataset3.csv
├── docker-compose.yml         # Orchestration
└── README.md
```

## 🤖 Orchestration multi-agent (CrewAI)

- **3 agents spécialistes** : chacun ne consulte qu'un fichier CSV (`dataset1/2/3.csv`) et extrait les articles pertinents en parallèle.
- **Agent superviseur** : consolide les briefs, arbitre les divergences et renvoie une réponse unique citant les sources.
- **LLM commun** : les agents s'appuient sur Ollama (par défaut) mais peuvent être redirigés vers un autre modèle via `CREW_MODEL`.
- **Fallback automatique** : si CrewAI est désactivé ou indisponible, le pipeline RAG classique (Ollama) prend le relais.
- **Configuration** : ajustez `ENABLE_CREW_AGENTS`, `CREW_AGENT_TOP_K`, `CREW_MIN_SCORE` et les températures pour équilibrer vitesse / précision.

## 🚀 Démarrage Rapide

### Prérequis

- **Python** 3.11+
- **Node.js** 18+
- **Ollama** (pour LLM local)
- **Git**

### Installation

**1. Cloner le projet**
```bash
git clone <repository-url>
cd assistjur
```

**2. Installer Ollama**
```bash
# Linux/Mac
curl -fsSL https://ollama.ai/install.sh | sh

# Windows : télécharger depuis https://ollama.ai/download
```

**3. Télécharger un modèle LLM**
```bash
ollama pull qwen2.5:7b
```

**4. Backend - Installation**
```bash
cd backend
pip install -r requirements.txt
```

**5. Backend - Configuration**
```bash
# Copier et éditer le fichier .env
cp env.example .env
```

**6. Backend - Démarrage**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**7. Frontend - Installation**
```bash
cd frontend
npm install
```

**8. Frontend - Démarrage**
```bash
npm start
```

**9. Accéder à l'application**
- Frontend : http://localhost:4200
- API : http://localhost:8000
- Documentation API : http://localhost:8000/docs

## ⚙️ Configuration

### Variables d'environnement

Créer un fichier `.env` dans le dossier `backend/` :

```env
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Gemini (optionnel)
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-pro

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=legal_documents

# Données
DATA_DIRECTORY=../data

# CrewAI (multi-agent)
ENABLE_CREW_AGENTS=true
CREW_AGENT_TOP_K=3
CREW_MIN_SCORE=0.05
CREW_MODEL=ollama/qwen2.5:7b  # vide = même modèle que Ollama
CREW_TEMPERATURE=0.20
CREW_SUPERVISOR_TEMPERATURE=0.15

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# CORS
CORS_ORIGINS=["http://localhost:4200"]
```

### Modèles Ollama recommandés

- **qwen2.5:7b** : Modèle généraliste et rapide, bon pour le français
- **mistral** : Modèle compact et efficace
- **codellama** : Spécialisé pour le code et la logique

## 📊 Structure des données CSV

Les fichiers CSV doivent contenir les colonnes suivantes :

```csv
DOC,Titre,Chapitre,Section,Article,Contenu,Pages
Loi n° 17-95,TITRE PREMIER,,,Article 2,"La forme, la durée...",[5]
```

**Colonnes requises :**
- `DOC` : Type de document (loi, décret, code, etc.)
- `Contenu` : Texte juridique principal
- `source_file` : Nom du fichier source (ajouté automatiquement)

**Colonnes optionnelles :**
- `Titre`, `Chapitre`, `Section`, `Article` : Structure hiérarchique
- `Pages` : Référence aux pages

## 🔧 API Endpoints

### Questions juridiques
```http
POST /ask
Content-Type: application/json

{
  "question": "Quelles sont les conditions pour créer une société anonyme?",
  "context_limit": 5
}
```

### Historique
```http
GET /history?limit=50
DELETE /history
```

### Gestion des données
```http
POST /reload-data
GET /status
GET /health
```

## 🎨 Interface Utilisateur

### Landing Page
- **Animation 3D Spline** : Orbe interactif réactif
- **Design Dark Mode** : Dégradés purple/pink avec glassmorphism
- **Feature Cards** : Animations hover modernes avec effets de brillance
- **Responsive** : Optimisé mobile et desktop

### Chat Interface
- **Messages en temps réel** : Interface conversationnelle fluide
- **Sources expandables** : Affichage détaillé des articles juridiques
- **Scores de pertinence** : Indicateurs visuels de fiabilité
- **Historique sidebar** : Accès rapide aux conversations précédentes
- **Configuration dynamique** : Ajustement du nombre de sources (3-10)

## 🔍 Utilisation

### Exemples de questions

```
"Quelles sont les conditions pour créer une société anonyme au Maroc?"
"Quel est le capital minimum requis pour une SA?"
"Quelles sont les obligations de publicité des sociétés?"
"Comment fonctionne l'immatriculation au registre du commerce?"
```

### Réponse type

```json
{
  "answer": "Selon l'article 6 de la loi n° 17-95, le capital social d'une société anonyme ne peut être inférieur à trois millions de dirhams si la société fait publiquement appel à l'épargne et à trois cent mille dirhams dans le cas contraire.",
  "sources": [
    {
      "doc": "Loi n° 17-95",
      "article": "Article 6",
      "contenu": "Le capital social d'une société anonyme...",
      "source_file": "dataset1.csv",
      "relevance_score": 0.95
    }
  ],
  "confidence_score": 0.92,
  "processing_time": 2.3
}
```

## 🛠️ Stack Technique

### Backend
- **FastAPI** : API REST moderne et performante
- **ChromaDB** : Base vectorielle pour embeddings
- **CrewAI** : Orchestration multi-agent
- **Ollama** : LLM local (qwen2.5:7b/mistral)
- **Pydantic** : Validation des données

### Frontend
- **Angular 17** : Framework SPA
- **Material Design** : Composants UI
- **TailwindCSS** : Styling utilitaire
- **Spline** : Animation 3D interactive
- **TypeScript** : Typage statique

### DevOps
- **Docker** : Containerisation
- **Git** : Versioning
- **Uvicorn** : Serveur ASGI

## 🚨 Dépannage

### Problèmes courants

1. **Ollama ne démarre pas**
```bash
# Vérifier les ports
netstat -tulpn | grep 11434

# Redémarrer Ollama
docker-compose restart ollama
```

2. **Modèle non trouvé**
```bash
# Lister les modèles
ollama list

# Télécharger un modèle
ollama pull qwen2.5:7b
```

3. **Erreur de vectorisation**
```bash
# Vider ChromaDB
rm -rf backend/chroma_db/

# Recharger les données
curl -X POST http://localhost:8000/reload-data
```

4. **Frontend ne se connecte pas**
```bash
# Vérifier l'API
curl http://localhost:8000/health

# Vérifier CORS dans .env
CORS_ORIGINS=["http://localhost:4200"]
```

### Performance

- **Premier démarrage** : 5-10 minutes (téléchargement modèle)
- **Rechargement données** : 1-3 minutes selon la taille
- **Réponse moyenne** : presque 10 min (multi agents , ressources matérielles ...) optimisable en cas d'utilisation d'une config avancée (nvidia rtx ...)
- **Mémoire recommandée** : 16GB+ RAM

## 🎯 État Actuel

### ✅ Fonctionnel
- Backend FastAPI avec RAG pipeline complet
- CrewAI multi-agent (3 agents + superviseur)
- ChromaDB vectorisation et recherche sémantique
- Ollama LLM intégration
- Frontend Angular avec Material Design
- Landing page 3D interactive (Spline)
- Dark mode avec animations modernes
- Historique persistant des conversations
- Cache intelligent des réponses
- API REST complète avec documentation Swagger

### 🚧 Améliorations Futures
- [ ] Support multilingue (arabe)
- [ ] Export des conversations (PDF/JSON)
- [ ] Authentification utilisateur
- [ ] Déploiement Docker optimisé
- [ ] Tests unitaires et e2e complets
- [ ] Monitoring et analytics

## 🤝 Contribution

Les contributions sont les bienvenues ! Merci de :

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

---

**Note importante** : Cet assistant fournit des informations basées sur les documents fournis et ne remplace pas l'avis d'un avocat professionnel. Toujours consulter un juriste qualifié pour des conseils juridiques spécifiques.

---
## Développé par : ID BEN AHMED Aissam
