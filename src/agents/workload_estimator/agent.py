"""Starter Code Générique d'Agent Étudiant (LangGraph + RAG + SLM).

Ce fichier sert de point de départ universel pour TOUS les archétypes :
- advisor, auditor, quiz_gen, internship_matcher, workload_estimator, exam_planner.

Directives pour votre équipe :
1. Personnalisez les propriétés 'name' et 'description' ci-dessous.
2. Branchez vos filtres RAG ou vos outils Python déterministes (tools.py).
3. Soignez le prompt système envoyé au SLM local (Ollama 3B).
"""

import os
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.core.base_agent import BaseAgentPlugin
from src.core.ingestion import get_vector_store
from src.core.state import AgentState

from .tools import format_domain_response, process_domain_logic


class WorkloadEstimatorPlugin(BaseAgentPlugin):
    """Agent spécialisé conçu et développé par l'équipe d'étudiants.

    Hérite obligatoirement du contrat d'interface BaseAgentPlugin.
    """

    @property
    def name(self) -> str:
        """Identifiant unique de l'agent (archétype 5 : estimation de charge et volume horaire)."""
        return "workload_estimator"

    @property
    def description(self) -> str:
        """Description sémantique pour le Superviseur LangGraph.

        Mots-clés d'intention : charge de travail, volume d'heures, ECTS, CM, TD, TP,
        travail personnel prévisionnel.
        """
        return (
            "Estimateur de charge de travail académique et volume d'heures. "
            "Analyse le nombre d'ECTS, les heures de cours magistraux, TD et TP, "
            "et évalue le temps de travail personnel prévisionnel requis."

        )

    def create_node(self) -> Callable[[AgentState], dict[str, Any]]:
        """Retourne la fonction exécutable (nœud LangGraph) de l'agent."""
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b")

        llm = ChatOllama(base_url=ollama_url, model=model_name, temperature=0.1, num_predict=400)
        vector_store = None

        def node(state: AgentState) -> dict[str, Any]:
            nonlocal vector_store
            if vector_store is None:
                vector_store = get_vector_store(reindex_if_empty=False)

            # Extraction du dernier message reçu
            last_message = state["messages"][-1]
            user_query = str(last_message.content)

            # Étape 1 : RAG ciblé avec filtres de métadonnées (k <= 2 pour SLM 3B)
            """Le volume horaire (CM/TD/TP) et les ECTS ne figurent que dans l'en-tête du
             syllabus : on filtre toujours sur cette section, puis sur le code UE ou,
             à défaut, le semestre s'ils sont cités dans la question."""
            import re

            codes_ue = list(
                dict.fromkeys(
                    f"{prefixe.upper()}_{numero}"
                    for prefixe, numero in re.findall(r"\b([A-Za-z]{2,4})[_-]?(\d{3})\b", user_query)
                )
            )
            semestres = list(
                dict.fromkeys(
                    s.upper() for s in re.findall(r"\b(S(?:10|[5-9])|M2)\b", user_query, re.IGNORECASE)
                )
            )

            conditions: list[dict[str, Any]] = [{"section_title": "En-tête & Informations Générales"}]
            if codes_ue:
                conditions.append({"code_ue": {"$in": codes_ue}})
            elif semestres:
                conditions.append({"semestre": {"$in": semestres}})
            metadata_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]

            relevant_docs = []
            try:
                retriever = vector_store.as_retriever(
                    search_kwargs={"k": 2, "filter": metadata_filter}
                )
                relevant_docs = retriever.invoke(user_query)
            except Exception:
                relevant_docs = []

            rag_context = (
                "\n---\n".join([doc.page_content for doc in relevant_docs])
                if relevant_docs
                else "Aucun document spécifique trouvé."
            )

            # Étape 2 : Exécution des outils Python déterministes (Règle d'or SLM)
            # tools.py fait TOUT le calcul : syllabus du S8 (ou UE citées), regex CM/TD/TP,
            # convention 1 ECTS = 25 h, travail personnel et charge totale. Le SLM ne fait
            # que commenter ces chiffres (voir la clé 'consigne_slm').
            tool_results = process_domain_logic(user_query, relevant_docs)

            # Étape 3 : Synthèse par le SLM avec Prompt Système défensif
            system_prompt = (
                f"Tu es l'agent spécialisé '{self.name}'.\n"
                "Ton rôle est d'apporter une réponse rigoureuse, concise et factuelle à l'étudiant.\n\n"
                "RÈGLES STRICTES :\n"
                "1. Base-toi EXCLUSIVEMENT sur les éléments du Contexte RAG et les Résultats des Outils fournis ci-dessous.\n"
                "2. N'invente aucun cours, code d'UE ou prérequis non présent dans les données.\n"
                "3. Si l'information est absente ou insuffisante, reconnais-le explicitement.\n"
                "4. Reste concis (maximum 3 à 5 phrases claires).\n\n"
                f"CONTEXTE RAG :\n{rag_context}\n\n"
                f"RÉSULTATS DES OUTILS PYTHON :\n{tool_results}"
            )

            messages = [
                SystemMessage(content=system_prompt),
                last_message,
            ]

            try:
                response = llm.invoke(messages)
                response_text = str(response.content)
            except Exception:
                # Fallback défensif en cas d'indisponibilité temporaire du SLM
                response_text = format_domain_response(tool_results)

            # Préfixage obligatoire avec l'identifiant pour la traçabilité
            final_message = AIMessage(content=f"[Agent{self.name}]\n{response_text}")

            return {
                "messages": [final_message],
                "current_agent": self.name,
            }

        return node
