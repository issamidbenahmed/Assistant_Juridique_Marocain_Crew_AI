import json
import logging
import re
from typing import List, Dict, Any, Optional

from ..models import Source
from ..config import settings

try:
    from crewai import Agent, Task, Crew, Process, LLM
except ImportError:  # pragma: no cover - library optionnelle
    Agent = None
    Task = None
    Crew = None
    Process = None
    LLM = None

logger = logging.getLogger(__name__)


class CrewMultiAgentService:
    """Orchestre trois agents spécialisés + un superviseur via CrewAI."""

    def __init__(self, vector_store, dataset_files: List[str]):
        self.vector_store = vector_store
        self.dataset_files = sorted(dataset_files)
        self.dataset_agents: List[Dict[str, Any]] = []
        self.supervisor_agent: Optional[Agent] = None
        self._latest_sources: List[Source] = []
        self.is_available = bool(
            settings.ENABLE_CREW_AGENTS and Agent and Task and Crew and Process and LLM
        )

        if not self.is_available:
            logger.info("CrewAI désactivé ou non installé — pipeline multi-agent ignoré.")
            return

        try:
            self._build_agents()
        except Exception as exc:
            self.is_available = False
            logger.exception("Impossible d'initialiser CrewAI: %s", exc)

    def update_datasets(self, dataset_files: List[str]):
        """Met à jour la liste des CSV lorsque les données changent."""
        self.dataset_files = sorted(dataset_files)
        if not self.is_available:
            return
        self._build_agents()

    def run(self, question: str, context_limit: int = 5) -> Optional[Dict[str, Any]]:
        """Lance les agents spécialisés et retourne la synthèse du superviseur."""
        if not self.is_available or not self.dataset_agents or not self.supervisor_agent:
            return None

        logger.info("=" * 80)
        logger.info("🤖 CREWAI MULTI-AGENT ACTIVÉ")
        logger.info("=" * 80)
        logger.info(f"📝 Question: {question}")
        
        dataset_contexts = self._prepare_dataset_contexts(question, context_limit)
        active_agents = [
            entry for entry in self.dataset_agents if dataset_contexts.get(entry["dataset"])
        ]
        if not active_agents:
            logger.warning("Aucun contexte disponible pour les agents CrewAI.")
            return None

        logger.info(f"👥 {len(active_agents)} agents spécialisés activés:")
        for entry in active_agents:
            logger.info(f"   - Agent {entry['dataset']}")

        tasks = []
        for entry in active_agents:
            dataset = entry["dataset"]
            description = self._build_dataset_task_description(
                dataset=dataset,
                question=question,
                context_text=dataset_contexts[dataset],
            )
            tasks.append(
                Task(
                    description=description,
                    expected_output=(
                        "JSON strict: {\"dataset\": \"nom.csv\", "
                        "\"score_confiance\": 0-1, "
                        "\"points\": [{\"article\": \"\", \"resume\": \"\", \"source_file\": \"\"}]}"
                    ),
                    agent=entry["agent"],
                    async_execution=True,
                )
            )

        logger.info("🎯 Création de la tâche superviseur...")
        supervisor_task = Task(
            description=self._build_supervisor_task_description(question),
            expected_output=(
                "JSON strict: {\"answer\": \"réponse synthétique\", "
                "\"confidence\": 0-1, "
                "\"citations\": [\"dataset1.csv - Article 5\", ...]}"
            ),
            agent=self.supervisor_agent,
            context=tasks,
        )

        crew = Crew(
            agents=[entry["agent"] for entry in active_agents] + [self.supervisor_agent],
            tasks=tasks + [supervisor_task],
            process=Process.sequential,
            verbose=True,  # Activer les logs détaillés
        )

        logger.info("🚀 Lancement de l'équipe CrewAI (agents dataset en parallèle via async_execution)...")
        try:
            final_output = crew.kickoff()
            logger.info("✅ CrewAI a terminé avec succès!")
        except Exception as exc:
            logger.exception("❌ Échec de l'exécution CrewAI: %s", exc)
            return None

        parsed = self._parse_supervisor_output(final_output)
        if not parsed:
            logger.warning("⚠️  Impossible de parser la sortie du superviseur")
            return None

        # Limiter et injecter les sources récentes pour le pipeline principal
        parsed["sources"] = self._latest_sources[: max(context_limit, 3)]
        logger.info(f"📊 Réponse finale générée avec {len(parsed['sources'])} sources")
        logger.info("=" * 80)
        return parsed

    # --------- Internal helpers ----------

    def _build_agents(self):
        """(Re)construit les agents spécialisés + superviseur."""
        base_llm = self._create_llm(
            model_override=settings.CREW_MODEL,
            temperature=settings.CREW_TEMPERATURE,
        )
        supervisor_llm = self._create_llm(
            model_override=settings.CREW_MODEL,
            temperature=settings.CREW_SUPERVISOR_TEMPERATURE,
        )

        if not base_llm or not supervisor_llm:
            raise RuntimeError("Impossible d'initialiser les LLM CrewAI")

        self.dataset_agents = []
        for dataset in self.dataset_files:
            agent = Agent(
                role=f"Analyste juridique {dataset}",
                goal=(
                    "Fournir des faits juridiques fiables extraits uniquement du fichier "
                    f"{dataset} pour répondre rapidement aux questions."
                ),
                backstory=(
                    "Juriste spécialisé qui connaît la structure du fichier et sait repérer les "
                    "articles pertinents pour éclairer la question."
                ),
                allow_delegation=False,
                verbose=False,
                llm=base_llm,
            )
            self.dataset_agents.append({"dataset": dataset, "agent": agent})

        self.supervisor_agent = Agent(
            role="Superviseur juridique",
            goal=(
                "Croiser les synthèses des trois analystes CSV pour produire une réponse finale "
                "cohérente, sourcée et concise."
            ),
            backstory="Avocat senior chargé de valider les réponses avant envoi à l'utilisateur.",
            allow_delegation=False,
            verbose=False,
            llm=supervisor_llm,
        )

    def _create_llm(self, model_override: Optional[str], temperature: float) -> Optional[LLM]:
        """Construit un objet LLM CrewAI basé sur Ollama (par défaut)."""
        if not LLM:
            return None
        model_name = model_override or f"ollama/{settings.OLLAMA_MODEL}"
        # Timeout très élevé pour permettre à Qwen de répondre même lentement
        timeout_seconds = settings.CREW_TIMEOUT
        kwargs: Dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": settings.CREW_MAX_TOKENS,
            "timeout": timeout_seconds,
            "request_timeout": timeout_seconds,  # Timeout pour litellm
            "api_base": settings.OLLAMA_BASE_URL if model_name.startswith("ollama/") else None,
        }
        if model_name.startswith("ollama/"):
            kwargs["base_url"] = settings.OLLAMA_BASE_URL
        try:
            return LLM(**kwargs)
        except Exception as exc:
            logger.exception("Erreur lors de la création du LLM CrewAI (%s): %s", model_name, exc)
            return None

    def _prepare_dataset_contexts(self, question: str, context_limit: int) -> Dict[str, str]:
        """Récupère les extraits pertinents pour chaque dataset."""
        contexts: Dict[str, str] = {}
        aggregated_sources: List[Source] = []
        per_dataset = max(context_limit, settings.CREW_AGENT_TOP_K)

        for dataset in self.dataset_files:
            sources = self.vector_store.search_documents_for_dataset(
                query=question,
                source_file=dataset,
                n_results=per_dataset,
                min_score=settings.CREW_MIN_SCORE,
            )
            if not sources:
                continue
            aggregated_sources.extend(sources)
            contexts[dataset] = self._format_sources_for_prompt(dataset, sources)

        self._latest_sources = aggregated_sources
        return contexts

    def _format_sources_for_prompt(self, dataset: str, sources: List[Source]) -> str:
        """Formate les sources en texte structuré pour les prompts agents."""
        parts = [f"=== Extraits issus de {dataset} ==="]
        for idx, source in enumerate(sources, 1):
            snippet = source.contenu.replace("\n", " ").strip()
            if len(snippet) > 600:
                snippet = snippet[:600] + "..."
            parts.append(
                f"[{idx}] Document: {source.doc or 'Inconnu'} | Article: {source.article or 'N/A'} | "
                f"Chapitre: {source.chapitre or 'N/A'} | Score: {source.relevance_score:.2f}\n"
                f"Texte: {snippet}"
            )
        return "\n".join(parts)

    def _build_dataset_task_description(self, dataset: str, question: str, context_text: str) -> str:
        """Crée la consigne spécifique à un agent dataset."""
        return (
            f"Tu es l'expert assigné au fichier {dataset}. "
            f"Analyse la question suivante en te basant EXCLUSIVEMENT sur les extraits fournis.\n"
            f"QUESTION UTILISATEUR: {question}\n\n"
            f"{context_text}\n\n"
            "Consignes:\n"
            "- Ne pas inventer de nouvelles sources.\n"
            "- Résume les passages utiles et précise l'article ou la section quand c'est possible.\n"
            "- Retourne UNIQUEMENT du JSON valide."
        )

    def _build_supervisor_task_description(self, question: str) -> str:
        """Consigne pour le superviseur."""
        return (
            "Tu es le superviseur juridique. Utilise les sorties JSON des analystes pour répondre "
            "de manière synthétique à l'utilisateur.\n"
            f"QUESTION: {question}\n"
            "Étapes:\n"
            "1. Croise les points clés fournis par chaque dataset.\n"
            "2. Priorise les passages avec le meilleur score de confiance.\n"
            "3. Rédige la réponse finale en français professionnel avec références explicites "
            "(doc + article).\n"
            "4. Retourne uniquement du JSON valide."
        )

    def _parse_supervisor_output(self, output: Any) -> Optional[Dict[str, Any]]:
        """Tente de parser la sortie du superviseur en JSON."""
        # Si c'est déjà un dict, on le retourne
        if isinstance(output, dict):
            return output
        
        # Si c'est un objet CrewAI avec attribut 'raw', on l'extrait
        if hasattr(output, 'raw'):
            output = output.raw
        
        # Si c'est un objet CrewAI avec attribut 'output', on l'extrait
        if hasattr(output, 'output'):
            output = output.output
        
        # Convertir en string si nécessaire
        output_str = str(output) if not isinstance(output, str) else output
        
        # Essayer de parser directement
        try:
            return json.loads(output_str)
        except json.JSONDecodeError:
            pass
        
        # Essayer d'extraire le JSON avec regex
        match = re.search(r"\{.*\}", output_str, re.DOTALL)
        if not match:
            logger.error("Sortie superviseur non parsable: %s", output_str[:500])
            return None
        
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.error("JSON invalide renvoyé par le superviseur: %s", match.group(0)[:500])
            return None


