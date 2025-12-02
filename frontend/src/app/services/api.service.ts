import { Injectable } from '@angular/core';
import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Observable, throwError } from 'rxjs';
import { catchError, map } from 'rxjs/operators';
import { 
  AnswerResponse, 
  QuestionRequest, 
  HistoryEntry, 
  ServiceStatus 
} from '../models/legal.models';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private readonly baseUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  /**
   * Pose une question à l'assistant juridique
   */
  askQuestion(request: QuestionRequest): Observable<AnswerResponse> {
    return this.http.post<AnswerResponse>(`${this.baseUrl}/ask`, request)
      .pipe(
        catchError(this.handleError)
      );
  }

  /**
   * Récupère l'historique des conversations
   */
  getHistory(limit: number = 50): Observable<HistoryEntry[]> {
    return this.http.get<{history: HistoryEntry[], count: number}>(`${this.baseUrl}/history?limit=${limit}`)
      .pipe(
        map(response => response.history),
        catchError(this.handleError)
      );
  }

  /**
   * Vide l'historique des conversations
   */
  clearHistory(): Observable<{message: string}> {
    return this.http.delete<{message: string}>(`${this.baseUrl}/history`)
      .pipe(
        catchError(this.handleError)
      );
  }

  /**
   * Recharge les données CSV
   */
  reloadData(): Observable<{message: string, documents_processed: number, processing_time: number}> {
    return this.http.post<{message: string, documents_processed: number, processing_time: number}>(`${this.baseUrl}/reload-data`, {})
      .pipe(
        catchError(this.handleError)
      );
  }

  /**
   * Récupère le statut du service
   */
  getServiceStatus(): Observable<ServiceStatus> {
    return this.http.get<{status: ServiceStatus}>(`${this.baseUrl}/status`)
      .pipe(
        map(response => response.status),
        catchError(this.handleError)
      );
  }

  /**
   * Vérifie la santé de l'API
   */
  checkHealth(): Observable<{status: string, services: ServiceStatus}> {
    return this.http.get<{status: string, services: ServiceStatus}>(`${this.baseUrl}/health`)
      .pipe(
        catchError(this.handleError)
      );
  }

  /**
   * Gestionnaire d'erreurs HTTP
   */
  private handleError(error: HttpErrorResponse): Observable<never> {
    let errorMessage = 'Une erreur inattendue s\'est produite';
    
    if (error.error instanceof ErrorEvent) {
      // Erreur côté client
      errorMessage = `Erreur: ${error.error.message}`;
    } else {
      // Erreur côté serveur
      if (error.error && error.error.detail) {
        errorMessage = error.error.detail;
      } else if (error.error && error.error.error) {
        errorMessage = error.error.error;
      } else {
        errorMessage = `Erreur ${error.status}: ${error.statusText}`;
      }
    }
    
    console.error('Erreur API:', error);
    return throwError(() => new Error(errorMessage));
  }
}
