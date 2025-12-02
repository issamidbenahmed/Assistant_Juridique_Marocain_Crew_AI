import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import os
import numpy as np
from ..models import LegalDocument, Source
from ..config import settings
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStore:
    """Service pour gérer ChromaDB et les embeddings"""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.embedding_dim = None
        self._initialize_chroma()
        self._initialize_embedding_model()
    
    def _initialize_chroma(self):
        """Initialise ChromaDB"""
        try:
            persist_path = Path(settings.CHROMA_PERSIST_DIRECTORY).expanduser().resolve()
            os.makedirs(persist_path, exist_ok=True)
            
            # Utiliser PersistentClient si disponible, sinon Client (compatibilité)
            try:
                self.client = chromadb.PersistentClient(
                    path=str(persist_path),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            except Exception:
                logger.warning("PersistentClient unavailable, trying regular Client()")
                self.client = chromadb.Client()
            
            # Créer ou récupérer la collection
            try:
                self.collection = self.client.get_collection(name=settings.CHROMA_COLLECTION_NAME)
                logger.info(f"Collection '{settings.CHROMA_COLLECTION_NAME}' récupérée")
            except Exception:
                self.collection = self.client.create_collection(
                    name=settings.CHROMA_COLLECTION_NAME,
                    metadata={"description": "Documents juridiques marocains"}
                )
                logger.info(f"Collection '{settings.CHROMA_COLLECTION_NAME}' créée")
                
        except Exception as e:
            logger.exception(f"Erreur lors de l'initialisation de ChromaDB: {str(e)}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialise le modèle d'embedding"""
        try:
            # Utiliser un modèle multilingue pour le français et l'arabe
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            # déterminer la dimension d'embedding
            sample = self.embedding_model.encode(["test"], convert_to_tensor=False)
            self.embedding_dim = len(sample[0])
            logger.info(f"Modèle d'embedding initialisé (dim={self.embedding_dim})")
        except Exception as e:
            logger.exception(f"Erreur lors de l'initialisation du modèle d'embedding: {str(e)}")
            # Fallback vers None (on utilisera embeddings aléatoires)
            self.embedding_model = None
            self.embedding_dim = 384
            logger.warning("Utilisation d'embeddings aléatoires comme fallback (dim=384)")
    
    def add_documents(self, documents: List[LegalDocument]) -> bool:
        """Ajoute des documents à la base vectorielle"""
        try:
            if not documents:
                logger.warning("Aucun document à ajouter")
                return False
            
            ids = []
            texts = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                doc_id = f"{doc.source_file}_{i}_{abs(hash(doc.contenu)) % 1000000}"
                ids.append(doc_id)
                text_to_vectorize = self._create_text_for_vectorization(doc)
                texts.append(text_to_vectorize)
                metadata = {
                    "doc": doc.doc or "",
                    "titre": doc.titre or "",
                    "chapitre": doc.chapitre or "",
                    "section": doc.section or "",
                    "article": doc.article or "",
                    "pages": doc.pages or "",
                    "source_file": doc.source_file,
                    "contenu": (doc.contenu[:500] + "...") if len(doc.contenu) > 500 else doc.contenu
                }
                metadatas.append(metadata)
            
            logger.info(f"Génération des embeddings pour {len(texts)} documents...")
            if self.embedding_model:
                embeddings = self.embedding_model.encode(texts, convert_to_tensor=False).tolist()
            else:
                embeddings = [np.random.rand(self.embedding_dim).tolist() for _ in texts]
            
            # Sanity check: embeddings length matches texts
            if len(embeddings) != len(texts):
                logger.error("Mismatch embeddings/texts length")
                return False
            
            # Add in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_documents = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                logger.info(f"Ajouté batch {i//batch_size + 1}/{(len(ids) + batch_size - 1)//batch_size}")
            
            # debug: afficher quelques metadatas stockées
            try:
                tmp = self.collection.peek(max_results=3)
                logger.debug(f"Exemple de documents dans la collection: {tmp.get('metadatas')}")
            except Exception:
                # peek peut ne pas exister selon version; ignore
                pass
            
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.exception(f"Erreur lors de l'ajout des documents: {str(e)}")
            return False
    
    def _create_text_for_vectorization(self, doc: LegalDocument) -> str:
        """Crée le texte optimisé pour la vectorisation"""
        parts = []
        
        if doc.doc:
            parts.append(f"Document: {doc.doc}")
        if doc.titre:
            parts.append(f"Titre: {doc.titre}")
        if doc.chapitre:
            parts.append(f"Chapitre: {doc.chapitre}")
        if doc.section:
            parts.append(f"Section: {doc.section}")
        if doc.article:
            parts.append(f"Article: {doc.article}")
        
        parts.append(f"Contenu: {doc.contenu}")
        
        return " | ".join(parts)
    
    def search_similar_documents(
        self, 
        query: str, 
        n_results: int = 5,
        min_score: float = 0.3
    ) -> List[Source]:
        """Recherche des documents similaires à la requête"""
        try:
            if self.embedding_model:
                query_embedding = self.embedding_model.encode([query], convert_to_tensor=False).tolist()[0]
            else:
                query_embedding = np.random.rand(self.embedding_dim).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"]
            )
            
            # Vérifier structure résultats
            if not results or "metadatas" not in results or not results["metadatas"]:
                logger.warning("Aucun résultat retourné par Chroma (metadatas manquants)")
                return []
            
            metadatas_list = results["metadatas"][0]
            documents_list = results["documents"][0]
            distances_list = results.get("distances", [[]])[0] if "distances" in results else [None]*len(metadatas_list)
            
            sources = []
            for metadata, document, distance in zip(metadatas_list, documents_list, distances_list):
                # Robust handling: if distance is None -> accept (score unknown)
                if distance is None:
                    similarity_score = 1.0
                else:
                    # distance returned by Chroma may être :
                    # - Cosine distance (0..2) or (0..1), ou L2 distance.
                    # On mappe robustement en [0,1]
                    try:
                        d = float(distance)
                        # si d est dans [0,1], traiter comme distance -> sim = 1-d
                        if 0.0 <= d <= 1.0:
                            similarity_score = max(0.0, 1.0 - d)
                        else:
                            # si d > 1 (ex : L2), on fait une conversion heuristique
                            similarity_score = 1.0 / (1.0 + d)
                    except Exception:
                        similarity_score = 0.0
                
                if similarity_score >= min_score:
                    source = Source(
                        doc=metadata.get("doc", ""),
                        titre=metadata.get("titre", ""),
                        chapitre=metadata.get("chapitre", ""),
                        article=metadata.get("article", ""),
                        contenu=metadata.get("contenu", ""),
                        pages=metadata.get("pages", ""),
                        source_file=metadata.get("source_file", ""),
                        relevance_score=similarity_score
                    )
                    sources.append(source)
            
            logger.info(f"Trouvé {len(sources)} documents pertinents pour la requête (min_score={min_score})")
            # debug: lister top 3
            for s in sources[:3]:
                logger.debug(f"Top source: {s.source_file} - score={s.relevance_score} - doc={s.doc}")
            return sources
            
        except Exception as e:
            logger.exception(f"Erreur lors de la recherche: {str(e)}")
            return []
    
    def search_documents_for_dataset(
        self,
        query: str,
        source_file: str,
        n_results: int = 3,
        min_score: float = 0.1
    ) -> List[Source]:
        """Recherche des documents similaires mais filtrés par fichier source"""
        try:
            if self.embedding_model:
                query_embedding = self.embedding_model.encode([query], convert_to_tensor=False).tolist()[0]
            else:
                query_embedding = np.random.rand(self.embedding_dim).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["metadatas", "documents", "distances"],
                where={"source_file": source_file}
            )
            
            if not results or "metadatas" not in results or not results["metadatas"]:
                return []
            
            metadatas_list = results["metadatas"][0]
            documents_list = results["documents"][0]
            distances_list = results.get("distances", [[]])[0] if "distances" in results else [None]*len(metadatas_list)
            
            sources: List[Source] = []
            for metadata, document, distance in zip(metadatas_list, documents_list, distances_list):
                if distance is None:
                    similarity_score = 1.0
                else:
                    try:
                        d = float(distance)
                        if 0.0 <= d <= 1.0:
                            similarity_score = max(0.0, 1.0 - d)
                        else:
                            similarity_score = 1.0 / (1.0 + d)
                    except Exception:
                        similarity_score = 0.0
                
                if similarity_score >= min_score:
                    sources.append(
                        Source(
                            doc=metadata.get("doc", ""),
                            titre=metadata.get("titre", ""),
                            chapitre=metadata.get("chapitre", ""),
                            article=metadata.get("article", ""),
                            contenu=metadata.get("contenu", ""),
                            pages=metadata.get("pages", ""),
                            source_file=metadata.get("source_file", ""),
                            relevance_score=similarity_score
                        )
                    )
            
            return sources
        except Exception as e:
            logger.exception(f"Erreur lors de la recherche filtrée ({source_file}): {str(e)}")
            return []
    
    def clear_collection(self) -> bool:
        """Vide la collection"""
        try:
            # supprime et recrée la collection
            try:
                self.client.delete_collection(settings.CHROMA_COLLECTION_NAME)
            except Exception:
                # certaines versions n'implémentent pas delete_collection de cette façon
                logger.debug("delete_collection non supporté via client, essai de reset via collection.reset()")
                try:
                    self.collection.reset()
                except Exception:
                    pass
            
            self.collection = self.client.create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "Documents juridiques marocains"}
            )
            logger.info("Collection vidée et recréée avec succès")
            return True
        except Exception as e:
            logger.exception(f"Erreur lors du vidage de la collection: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de la collection"""
        try:
            try:
                count = self.collection.count()
            except Exception:
                # fallback si count non disponible
                result = self.collection.peek(max_results=1)
                count = len(result.get("ids", []))
            return {
                "total_documents": count,
                "collection_name": settings.CHROMA_COLLECTION_NAME,
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2" if self.embedding_model else "random-fallback",
                "embedding_dim": self.embedding_dim
            }
        except Exception as e:
            logger.exception(f"Erreur lors de la récupération des statistiques: {str(e)}")
            return {"error": str(e)}
