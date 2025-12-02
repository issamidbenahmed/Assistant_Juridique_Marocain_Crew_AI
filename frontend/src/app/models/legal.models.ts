export interface Source {
  doc: string;
  titre?: string;
  chapitre?: string;
  article?: string;
  contenu: string;
  pages?: string;
  source_file: string;
  relevance_score: number;
}

export interface AnswerResponse {
  answer: string;
  sources: Source[];
  confidence_score: number;
  processing_time: number;
  timestamp: string;
}

export interface QuestionRequest {
  question: string;
  context_limit?: number;
}

export interface HistoryEntry {
  id: string;
  question: string;
  answer: string;
  sources: Source[];
  confidence_score: number;
  timestamp: string;
}

export interface ServiceStatus {
  is_initialized: boolean;
  ollama_available: boolean;
  gemini_available: boolean;
  vector_store_stats: {
    total_documents: number;
    collection_name: string;
    embedding_model: string;
  };
  csv_stats: {
    total_documents: number;
    source_files: number;
    doc_types: number;
    documents_with_articles: number;
    documents_with_chapters: number;
  };
  history_count: number;
}
