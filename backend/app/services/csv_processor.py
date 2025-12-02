import pandas as pd
import os
from typing import List, Dict, Any
from pathlib import Path
import logging
from ..models import LegalDocument

logger = logging.getLogger(__name__)

class CSVProcessor:
    """Service pour traiter les fichiers CSV contenant les documents juridiques"""
    
    def __init__(self, data_directory: str):
        # Résoudre le chemin pour éviter les erreurs relatives
        self.data_directory = Path(data_directory).expanduser().resolve()
        self.documents: List[LegalDocument] = []
    
    def load_all_csv_files(self) -> List[LegalDocument]:
        """Charge tous les fichiers CSV du répertoire data"""
        self.documents = []
        
        if not self.data_directory.exists():
            raise FileNotFoundError(f"Le répertoire {self.data_directory} n'existe pas")
        
        csv_files = sorted(list(self.data_directory.glob("*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans {self.data_directory}")
        
        logger.info(f"Trouvé {len(csv_files)} fichiers CSV à traiter dans {self.data_directory}")
        
        for csv_file in csv_files:
            try:
                documents = self._process_csv_file(csv_file)
                self.documents.extend(documents)
                logger.info(f"Traité {len(documents)} documents depuis {csv_file.name}")
            except Exception as e:
                logger.exception(f"Erreur lors du traitement de {csv_file.name}: {str(e)}")
                continue
        
        logger.info(f"Total de {len(self.documents)} documents chargés")
        return self.documents
    
    def _process_csv_file(self, csv_file: Path) -> List[LegalDocument]:
        """Traite un fichier CSV spécifique"""
        try:
            # Forcer l'encodage 'utf-8' puis fallback
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except Exception:
                df = pd.read_csv(csv_file, encoding='latin-1')
        except Exception as e:
            logger.exception(f"Impossible de lire {csv_file.name}: {e}")
            return []
        
        documents = []
        
        # Normaliser les noms de colonnes
        df.columns = df.columns.str.strip().str.lower()
        
        # Parcourir les lignes
        for idx, row in df.iterrows():
            try:
                # Extraire le contenu principal
                contenu = self._extract_content(row)
                if not contenu or pd.isna(contenu) or not str(contenu).strip():
                    # ignorer les lignes vides
                    continue
                
                # Créer le document juridique
                doc = LegalDocument(
                    doc=self._safe_get(row, 'doc', ''),
                    titre=self._safe_get(row, 'titre', ''),
                    chapitre=self._safe_get(row, 'chapitre', ''),
                    section=self._safe_get(row, 'section', ''),
                    article=self._safe_get(row, 'article', ''),
                    contenu=str(contenu).strip(),
                    pages=self._safe_get(row, 'pages', ''),
                    index=self._safe_get(row, 'index', ''),
                    source_file=csv_file.name
                )
                
                documents.append(doc)
                
            except Exception as e:
                # logger warning with line index for trace
                logger.exception(f"Erreur lors du traitement de la ligne {idx} dans {csv_file.name}: {e}")
                continue
        
        return documents
    
    def _extract_content(self, row: pd.Series) -> str:
        """Extrait le contenu principal d'une ligne"""
        # Priorité des colonnes de contenu
        content_columns = ['contenu', 'content', 'texte', 'text', 'article', 'body']
        
        for col in content_columns:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                return str(row[col]).strip()
        
        # Si aucune colonne de contenu trouvée, utiliser la première colonne non vide (exclusion des métadonnées évidentes)
        for col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                # éviter de retourner des colonnes qui sont manifestement des IDs numériques sans texte
                text_candidate = str(val).strip()
                if len(text_candidate) > 3:
                    return text_candidate
        
        return ""
    
    def _safe_get(self, row: pd.Series, key: str, default: Any = None) -> Any:
        """Récupère une valeur de manière sécurisée"""
        if key in row and pd.notna(row[key]):
            return str(row[key]).strip()
        return default
    
    def get_documents_by_source(self, source_file: str) -> List[LegalDocument]:
        """Récupère tous les documents d'un fichier source spécifique"""
        return [doc for doc in self.documents if doc.source_file == source_file]
    
    def get_documents_by_doc_type(self, doc_type: str) -> List[LegalDocument]:
        """Récupère tous les documents d'un type spécifique"""
        return [doc for doc in self.documents if (doc.doc or "").lower() == doc_type.lower()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les documents chargés"""
        if not self.documents:
            return {"total_documents": 0}
        
        stats = {
            "total_documents": len(self.documents),
            "source_files": len(set(doc.source_file for doc in self.documents)),
            "doc_types": len(set(doc.doc for doc in self.documents if doc.doc)),
            "documents_with_articles": len([doc for doc in self.documents if doc.article]),
            "documents_with_chapters": len([doc for doc in self.documents if doc.chapitre]),
        }
        
        return stats
