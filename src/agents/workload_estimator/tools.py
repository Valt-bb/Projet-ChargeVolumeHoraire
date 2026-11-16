"""Outils Python Déterministes pour l'Agent Étudiant.

RÈGLE D'OR DE L'INGÉNIERIE SLM :
Tout calcul arithmétique, manipulation d'ensembles, algorithme de graphe,
validation de format JSON ou parsing par regex DOIT être exécuté en pur Python ici.
Ne laissez jamais le SLM faire des additions ou des détections de cycles.
"""

from typing import Any


def parse_workload_hours(raw_text: str) -> dict[str, int]:
    """Extrait les heures de CM, TD et TP à l'aide d'expressions régulières.

    Seule la première occurrence de chaque type est retenue (ligne « Volume horaire »
    de l'en-tête), pour ne pas compter deux fois un même volume.
    """
    import re

    hours = dict.fromkeys(_HOUR_KEYS, 0)
    seen: set[str] = set()
    for value, kind in re.findall(r"(\d+)\s*h\s*(CM|TD|TP)\b", raw_text, re.IGNORECASE):
        key = kind.upper()
        if key not in seen:
            hours[key] = int(value)
            seen.add(key)
    return hours


def estimate_personal_study_time(ects: int, contact_hours: int) -> int:
    """Calcule le volume de travail personnel prévisionnel selon la formule officielle ECTS.

    travail personnel = ECTS × 25 h − heures de présentiel (jamais négatif).
    """
    return max(ects * ECTS_HOURS - contact_hours, 0)


# --- Paramètres de l'estimateur de charge (workload_estimator) ---------------
ECTS_HOURS = 25  # Convention ECTS : 1 crédit ≈ 25 h de travail total
DEFAULT_SEMESTER = "S8"  # Scénario : choix de cours du semestre 8
HEADER_SECTION = "En-tête & Informations Générales"
_HOUR_KEYS = ("CM", "TD", "TP")
_TOTAL_KEYS = ("ects", *_HOUR_KEYS, "contact_hours", "total_workload", "personal_study")
_syllabus_headers_cache: list[Any] = []


def _load_syllabus_headers() -> list[Any]:
    """Charge (une seule fois) les en-têtes des fiches de data/syllabus."""
    if not _syllabus_headers_cache:
        from pathlib import Path

        from src.core.ingestion import load_and_chunk_syllabus

        syllabus_dir = Path(__file__).resolve().parents[3] / "data" / "syllabus"
        try:
            docs = load_and_chunk_syllabus(str(syllabus_dir))
        except FileNotFoundError:
            docs = []
        _syllabus_headers_cache.extend(
            d for d in docs if d.metadata.get("section_title") == HEADER_SECTION
        )
    return _syllabus_headers_cache


def _query_scope(query: str) -> tuple[list[str], str]:
    """Codes UE cités dans la question, et semestre visé (S8 par défaut)."""
    import re

    codes = list(
        dict.fromkeys(
            f"{prefix.upper()}_{number}"
            for prefix, number in re.findall(r"\b([A-Za-z]{2,4})[_-]?(\d{3})\b", query)
        )
    )
    match = re.search(r"\bS(10|[5-9])\b|\bsemestre\s*(10|[5-9])\b", query, re.IGNORECASE)
    semestre = f"S{match.group(1) or match.group(2)}" if match else DEFAULT_SEMESTER
    return codes, semestre


def _course_workload(doc: Any) -> dict[str, Any] | None:
    """Calcule la charge d'une UE à partir de son en-tête (None si pas de volume horaire)."""
    import re

    metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
    content = str(getattr(doc, "page_content", ""))
    hours = parse_workload_hours(content)
    contact = sum(hours.values())
    if not metadata or contact == 0:
        return None

    ects_match = re.search(r"(\d+)\s*ECTS", content, re.IGNORECASE)
    ects = int(ects_match.group(1)) if ects_match else int(metadata.get("ects", 0) or 0)
    return {
        "code_ue": metadata.get("code_ue", "INCONNU"),
        "semestre": metadata.get("semestre", "?"),
        "ects": ects,
        **hours,
        "contact_hours": contact,
        "total_workload": ects * ECTS_HOURS,
        "personal_study": estimate_personal_study_time(ects, contact),
    }


def process_domain_logic(query: str, retrieved_docs: list[Any]) -> dict[str, Any]:
    """Calcule la charge de travail (workload_estimator) en pur Python.

    1. Ne garde que les syllabus nécessaires : les UE citées dans la question, sinon
       celles du semestre visé (S8 par défaut). Les passages ChromaDB sont complétés
       par les en-têtes de data/syllabus, pour ne manquer aucune UE du semestre.
    2. Extrait CM/TD/TP par regex, applique 1 ECTS = 25 h et calcule le travail
       personnel par UE puis la charge totale.

    Args:
        query: Texte de la question de l'étudiant
        retrieved_docs: Chunks documentaires extraits par ChromaDB

    Returns:
        Détail par UE, totaux, résumé chiffré et consigne pour le SLM.
    """
    codes, semestre = _query_scope(query)

    courses: list[dict[str, Any]] = []
    seen: set[str] = set()
    for doc in [*retrieved_docs, *_load_syllabus_headers()]:
        course = _course_workload(doc)
        if course is None or course["code_ue"] in seen:
            continue
        if codes and course["code_ue"] not in codes:
            continue
        if not codes and course["semestre"] != semestre:
            continue
        seen.add(course["code_ue"])
        courses.append(course)
    courses.sort(key=lambda c: c["code_ue"])

    totals = {key: sum(c[key] for c in courses) for key in _TOTAL_KEYS}

    if courses:
        lines = [
            f"{c['code_ue']} ({c['semestre']}) : {c['ects']} ECTS, "
            f"{c['CM']} h CM + {c['TD']} h TD + {c['TP']} h TP = {c['contact_hours']} h de présentiel, "
            f"charge totale {c['total_workload']} h dont {c['personal_study']} h de travail personnel"
            for c in courses
        ]
        lines.append(
            f"TOTAL : {totals['ects']} ECTS, {totals['contact_hours']} h de présentiel, "
            f"charge totale {totals['total_workload']} h dont {totals['personal_study']} h "
            f"de travail personnel (1 ECTS = {ECTS_HOURS} h)"
        )
        summary = " ; ".join(lines)
    else:
        summary = "Aucun volume horaire trouvé dans les syllabus pour cette demande"

    return {
        "status": "success",
        "documents_analyzed": len(retrieved_docs),
        "semestre": semestre,
        "courses": courses,
        "totals": totals,
        "summary": summary,
        "consigne_slm": (
            "Chiffres déjà calculés par Python : reprends-les tels quels et ne refais aucun "
            "calcul. Résume les points de vigilance méthodologiques pour organiser le travail : "
            "UE les plus lourdes en présentiel ou en travail personnel, et risque de concentrer "
            "plusieurs livrables de projet sur la même période."
        ),
    }


def format_domain_response(tool_results: dict[str, Any]) -> str:
    """Génère une réponse formatée de secours si le SLM n'est pas disponible."""
    summary = tool_results.get("summary", "")
    analyzed = tool_results.get("documents_analyzed", 0)
    return f"Traitement effectué : {summary} ({analyzed} document(s) analysé(s))."
