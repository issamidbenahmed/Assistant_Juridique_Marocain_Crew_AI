import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatSnackBarModule, MatSnackBar } from '@angular/material/snack-bar';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatChipsModule } from '@angular/material/chips';
import { MatSidenavModule } from '@angular/material/sidenav';
import { MatListModule } from '@angular/material/list';
import { MatMenuModule } from '@angular/material/menu';
import { MatDividerModule } from '@angular/material/divider';
import { MatBadgeModule } from '@angular/material/badge';
import { MatSelectModule } from '@angular/material/select';

import { ApiService } from './services/api.service';
import { 
  AnswerResponse, 
  QuestionRequest, 
  HistoryEntry, 
  ServiceStatus,
  Source 
} from './models/legal.models';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatToolbarModule,
    MatCardModule,
    MatInputModule,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatSnackBarModule,
    MatExpansionModule,
    MatChipsModule,
    MatSidenavModule,
    MatListModule,
    MatMenuModule,
    MatDividerModule,
    MatBadgeModule,
    MatSelectModule
  ],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'Assistant Juridique Marocain';
  
  // Landing page
  showLanding = true;
  
  // État de l'application
  isLoading = false;
  isServiceReady = false;
  serviceStatus: ServiceStatus | null = null;
  
  // Chat
  currentQuestion = '';
  messages: Array<{
    type: 'user' | 'assistant';
    content: string;
    timestamp: Date;
    sources?: Source[];
    confidenceScore?: number;
  }> = [];
  
  // Historique
  history: HistoryEntry[] = [];
  showHistory = false;
  
  // Configuration
  contextLimit = 5;

  constructor(
    private apiService: ApiService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.checkServiceHealth();
    this.loadHistory();
  }
  
  /**
   * Démarre l'application (passe de landing à chat)
   */
  startChat(): void {
    this.showLanding = false;
  }
  
  /**
   * Retourne à la landing page
   */
  goToLanding(): void {
    this.showLanding = true;
  }

  ngOnDestroy(): void {
    // Nettoyage si nécessaire
  }

  /**
   * Vérifie la santé du service
   */
  checkServiceHealth(): void {
    this.apiService.checkHealth().subscribe({
      next: (response) => {
        this.isServiceReady = response.status === 'healthy';
        this.serviceStatus = response.services;
        
        if (!this.isServiceReady) {
          this.showError('Le service juridique n\'est pas disponible');
        }
      },
      error: (error) => {
        this.isServiceReady = false;
        this.showError(`Erreur de connexion: ${error.message}`);
      }
    });
  }

  /**
   * Charge l'historique des conversations
   */
  loadHistory(): void {
    this.apiService.getHistory(50).subscribe({
      next: (history) => {
        this.history = history;
      },
      error: (error) => {
        console.error('Erreur lors du chargement de l\'historique:', error);
      }
    });
  }

  /**
   * Pose une question à l'assistant
   */
  askQuestion(): void {
    if (!this.currentQuestion.trim() || this.isLoading || !this.isServiceReady) {
      return;
    }

    const question = this.currentQuestion.trim();
    this.currentQuestion = '';
    this.isLoading = true;

    // Ajouter la question de l'utilisateur
    this.messages.push({
      type: 'user',
      content: question,
      timestamp: new Date()
    });

    // Préparer la requête
    const request: QuestionRequest = {
      question: question,
      context_limit: this.contextLimit
    };

    // Envoyer la requête
    this.apiService.askQuestion(request).subscribe({
      next: (response: AnswerResponse) => {
        // Ajouter la réponse de l'assistant
        this.messages.push({
          type: 'assistant',
          content: response.answer,
          timestamp: new Date(response.timestamp),
          sources: response.sources,
          confidenceScore: response.confidence_score
        });

        this.isLoading = false;
        
        // Recharger l'historique
        this.loadHistory();
      },
      error: (error) => {
        this.messages.push({
          type: 'assistant',
          content: `Erreur: ${error.message}`,
          timestamp: new Date()
        });
        this.isLoading = false;
        this.showError(`Erreur lors de la génération de la réponse: ${error.message}`);
      }
    });
  }

  /**
   * Gère la pression de la touche Entrée
   */
  onKeyPress(event: KeyboardEvent): void {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      this.askQuestion();
    }
  }

  /**
   * Charge une conversation depuis l'historique
   */
  loadFromHistory(entry: HistoryEntry): void {
    this.messages = [
      {
        type: 'user',
        content: entry.question,
        timestamp: new Date(entry.timestamp)
      },
      {
        type: 'assistant',
        content: entry.answer,
        timestamp: new Date(entry.timestamp),
        sources: entry.sources,
        confidenceScore: entry.confidence_score
      }
    ];
    this.showHistory = false;
  }

  /**
   * Vide l'historique
   */
  clearHistory(): void {
    this.apiService.clearHistory().subscribe({
      next: () => {
        this.history = [];
        this.showError('Historique vidé avec succès');
      },
      error: (error) => {
        this.showError(`Erreur lors du vidage de l'historique: ${error.message}`);
      }
    });
  }

  /**
   * Recharge les données
   */
  reloadData(): void {
    this.isLoading = true;
    this.apiService.reloadData().subscribe({
      next: (response) => {
        this.showError(`Données rechargées: ${response.documents_processed} documents traités`);
        this.isLoading = false;
        this.checkServiceHealth();
      },
      error: (error) => {
        this.showError(`Erreur lors du rechargement: ${error.message}`);
        this.isLoading = false;
      }
    });
  }

  /**
   * Obtient la classe CSS pour le score de confiance
   */
  getConfidenceClass(score: number): string {
    if (score >= 0.7) return 'confidence-high';
    if (score >= 0.4) return 'confidence-medium';
    return 'confidence-low';
  }

  /**
   * Obtient le texte du score de confiance
   */
  getConfidenceText(score: number): string {
    if (score >= 0.7) return 'Confiance élevée';
    if (score >= 0.4) return 'Confiance moyenne';
    return 'Confiance faible';
  }

  /**
   * Affiche un message d'erreur
   */
  private showError(message: string): void {
    this.snackBar.open(message, 'Fermer', {
      duration: 5000,
      horizontalPosition: 'center',
      verticalPosition: 'top'
    });
  }
}
