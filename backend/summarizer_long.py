# backend/summarizer_long.py
# Synthèse longue hiérarchique d'une affaire
# Auteur: Jax

"""
summarizer_long.py — Synthèse longue hiérarchique d'une affaire

Objectif :
- Produire une synthèse LONGUE et détaillée d'un dossier (plusieurs milliers de pages),
  en plusieurs passes hiérarchiques (map → reduce → refine → expand),
  sans exploser le contexte du LLM.

Entrées principales :
- case_id : identifiant de l'affaire (pour les logs / callbacks)
- chunks  : liste de dicts {"text": "...", "meta": {...}} (même format que pour summarizer.py)
- model   : nom du modèle Ollama (par ex. "deepseek-v3.1:671b-cloud")

Fonction publique principale :
- summarize_case_long(case_id, chunks, model=None, progress_cb=None, ...)

Le callback progress_cb reçoit des dicts de la forme :
- {"type": "stage", "message": "..."}
- {"type": "progress", "value": 0.42, "stage": "map-blocks"}
- {"type": "end", "case_id": case_id}

Ce module NE touche pas au RAG : il suppose qu'on lui fournit déjà les chunks pertinents
(ex: tous les documents d'une affaire via bm25.json).
"""

from __future__ import annotations

import os
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import prompts               # backend/prompts.py
import ollama_client         # backend/ollama_client.py
from llm_profiles import get_profile
from identity_mapper import anonymize_prompt_for_model, deanonymize_text_for_case

logger = logging.getLogger(__name__)

# Type du callback de progression
ProgressCB = Optional[Callable[[Dict[str, Any]], None]]

# ==========================
#  Config depuis .env
# ==========================

# Taille max (en caractères) d'un bloc de texte brut pour la 1ère passe (MAP)
# Valeur par défaut ajustée pour être plus robuste (8000 au lieu de 12000)
LONG_SUM_BLOCK_CHARS = int(os.getenv("LONG_SUM_BLOCK_CHARS", "8000"))

# Nombre de résumés locaux à regrouper pour une synthèse intermédiaire (REDUCE)
LONG_SUM_SECTION_GROUP = int(os.getenv("LONG_SUM_SECTION_GROUP", "5"))

# Nombre max de tokens générés pour les résumés locaux
# Par défaut 768 (plus léger que 1024 pour ménager le modèle en gros volume)
LONG_SUM_LOCAL_MAX_TOKENS = int(os.getenv("LONG_SUM_LOCAL_MAX_TOKENS", "768"))

# Nombre max de tokens générés pour les synthèses intermédiaires
LONG_SUM_SECTION_MAX_TOKENS = int(os.getenv("LONG_SUM_SECTION_MAX_TOKENS", "2048"))

# Nombre max de tokens générés pour les raffinements + rapport final
# Par défaut 2048 (au lieu de 4096) pour limiter les sorties très massives
LONG_SUM_FINAL_MAX_TOKENS = int(os.getenv("LONG_SUM_FINAL_MAX_TOKENS", "2048"))

# Titre de l’affaire dans les prompts (facultatif, peut être renseigné par la route)
DEFAULT_CASE_TITLE = os.getenv("LONG_SUM_DEFAULT_CASE_TITLE", "")


# ==========================
#  Helpers callback & texte
# ==========================

def _safe_cb(cb: ProgressCB, payload: Dict[str, Any]) -> None:
    if cb is None:
        return
    try:
        cb(payload)
    except Exception:
        logger.exception("summarizer_long: progress callback failed: %r", payload)


def _normalize(text: str) -> str:
    """Nettoyage minimal de la sortie LLM."""
    if not text:
        return ""
    text = text.replace("\ufeff", "").strip()
    # parfois le modèle renvoie ```markdown ...```
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    return text


def _concat_chunks(chunks: Sequence[Dict[str, Any]]) -> str:
    """Concatène tous les 'text' des chunks en un seul gros texte."""
    parts: List[str] = []
    for ch in chunks:
        txt = ch.get("text") or ch.get("content") or ""
        if txt:
            parts.append(txt)
    return "\n\n".join(parts)


# ==========================
#  Wrappers LLM
# ==========================

def _llm_short(
    model: str,
    prompt: str,
    *,
    case_id: Optional[str],
    max_tokens: int = LONG_SUM_LOCAL_MAX_TOKENS,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    """
    Appel LLM pour les résumés locaux (MAP), sans pseudonymisation.
    """
    txt = ollama_client.generate_long(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return _normalize(txt)


def _llm_long(
    model: str,
    prompt: str,
    *,
    case_id: Optional[str],
    max_tokens: int = LONG_SUM_FINAL_MAX_TOKENS,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> str:
    """
    Appel LLM pour les synthèses intermédiaires / rapport long / polish,
    sans pseudonymisation.
    """
    txt = ollama_client.generate_long(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return _normalize(txt)


# ==========================
#  Étape 1 — MAP : résumés de blocs
# ==========================

def _split_into_blocks(
    chunks: Sequence[Dict[str, Any]],
    max_chars: int = LONG_SUM_BLOCK_CHARS,
) -> List[str]:
    """
    Regroupe les chunks successifs en blocs de taille raisonnable (max_chars).
    Chaque bloc sera résumé individuellement (MAP).
    """
    blocks: List[str] = []
    current: List[str] = []
    current_len = 0

    for ch in chunks:
        txt = (ch.get("text") or ch.get("content") or "").strip()
        if not txt:
            continue
        l = len(txt) + 2
        if current and (current_len + l > max_chars):
            blocks.append("\n\n".join(current))
            current = [txt]
            current_len = l
        else:
            current.append(txt)
            current_len += l

    if current:
        blocks.append("\n\n".join(current))

    return blocks


def _prompt_block_summary(case_title: str, block_text: str) -> str:
    """
    Prompt pour résumer un bloc (~10-30 pages) de procédure.
    Objectif : produire un résumé local détaillé mais synthétique.
    """
    head = prompts.SYSTEM_FR
    title_part = f"Titre de l'affaire : {case_title}\n\n" if case_title else ""
    return (
        f"{head}\n"
        "Tu es un magistrat instructeur français.\n"
        "Tu reçois ci-dessous un large extrait de pièces de procédure (PV, rapports, auditions, écoutes, actes divers).\n\n"
        f"{title_part}"
        "TÂCHE (niveau bloc) :\n"
        "- Produire un résumé détaillé de cet extrait, en 25–40 lignes.\n"
        "- Mettre en évidence :\n"
        "  * les faits importants (qui, quoi, où, quand, comment),\n"
        "  * les principaux actes d'enquête,\n"
        "  * les protagonistes (suspects, témoins, victimes) et leurs rôles,\n"
        "  * les éléments techniques (téléphonie, financier, scientifique, etc.),\n"
        "  * les contradictions et évolutions de versions.\n"
        "- Rester STRICTEMENT factuel, sans spéculations.\n"
        "- Utiliser un style de rapport, en paragraphes courts (pas de roman, pas de bullet points\n"
        "  interminables, mais tu peux utiliser des listes si utile).\n\n"
        "Extrait à résumer :\n"
        "-------------------\n"
        f"{block_text}\n"
    )


def _map_block_summaries(
    model: str,
    case_id: str,
    case_title: str,
    blocks: Sequence[str],
    progress_cb: ProgressCB,
    *,
    temperature: Optional[float],
    top_p: Optional[float],
) -> List[str]:
    """
    MAP : génère un résumé par bloc de texte.
    """
    _safe_cb(progress_cb, {"type": "stage", "stage": "map-blocks", "message": "Résumés par blocs..."})
    summaries: List[str] = []
    total = max(len(blocks), 1)

    for i, block in enumerate(blocks, 1):
        progress_val = 0.05 + 0.25 * (i / total)
        _safe_cb(
            progress_cb,
            {
                "type": "progress",
                "value": progress_val,
                "stage": "map-blocks",
                "block_index": i,
                "block_total": total,
            },
        )
        prompt = _prompt_block_summary(case_title, block)
        s = _llm_short(
            model,
            prompt,
            case_id=case_id,
            max_tokens=LONG_SUM_LOCAL_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
        )
        summaries.append(s)

    return summaries


# ==========================
#  Étape 2 — REDUCE : synthèses intermédiaires
# ==========================

def _group_list(seq: Sequence[Any], n: int) -> List[List[Any]]:
    return [list(seq[i:i + n]) for i in range(0, len(seq), n)]


def _prompt_section_reduce(case_title: str, mini_summaries: Sequence[str]) -> str:
    """
    Prompt pour regrouper un paquet de résumés locaux en une synthèse intermédiaire.
    """
    head = prompts.SYSTEM_FR
    title_part = f"Titre de l'affaire : {case_title}\n\n" if case_title else ""
    joined = "\n\n---\n\n".join(mini_summaries)
    return (
        f"{head}\n"
        "Tu es un analyste judiciaire français.\n"
        "On te fournit ci-dessous plusieurs résumés locaux d'extraits de procédure (PV, rapports, auditions, etc.).\n\n"
        f"{title_part}"
        "TÂCHE (synthèse intermédiaire) :\n"
        "- Fusionner ces résumés locaux en une synthèse structurée de 30–50 lignes.\n"
        "- Organiser ton texte par grands axes (faits, enquête, protagonistes, actes techniques, etc.).\n"
        "- Simplifier les redondances, mais sans perdre les informations importantes.\n"
        "- Mentionner les principales chronologies et les évolutions (nouvelles pistes, revirements, décisions clés).\n"
        "- Style : rapport, neutre, factuel, en français, paragraphes courts.\n\n"
        "Résumés locaux à fusionner :\n"
        "----------------------------\n"
        f"{joined}\n"
    )


def _reduce_sections(
    model: str,
    case_id: str,
    case_title: str,
    block_summaries: Sequence[str],
    progress_cb: ProgressCB,
    *,
    temperature: Optional[float],
    top_p: Optional[float],
) -> List[str]:
    """
    REDUCE : regroupe les résumés de blocs en synthèses intermédiaires.
    """
    if not block_summaries:
        return []

    _safe_cb(progress_cb, {"type": "stage", "stage": "reduce-sections", "message": "Synthèses intermédiaires..."})
    groups = _group_list(block_summaries, max(1, LONG_SUM_SECTION_GROUP))
    total = max(len(groups), 1)
    section_summaries: List[str] = []

    for i, g in enumerate(groups, 1):
        progress_val = 0.30 + 0.25 * (i / total)
        _safe_cb(
            progress_cb,
            {
                "type": "progress",
                "value": progress_val,
                "stage": "reduce-sections",
                "group_index": i,
                "group_total": total,
            },
        )
        prompt = _prompt_section_reduce(case_title, g)
        s = _llm_long(
            model,
            prompt,
            case_id=case_id,
            max_tokens=LONG_SUM_SECTION_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
        )
        section_summaries.append(s)

    return section_summaries


# ==========================
#  Étape 3 — REFINE : rapport long incrémental
# ==========================

def _prompt_initial_long_report(case_title: str, section_text: str) -> str:
    head = prompts.SYSTEM_FR
    title_part = f"Titre de l'affaire : {case_title}\n\n" if case_title else ""
    return (
        f"{head}\n"
        "Tu es un magistrat instructeur rédigeant une synthèse judiciaire longue sur une affaire pénale.\n\n"
        f"{title_part}"
        "On te fournit ci-dessous une première synthèse intermédiaire de l'affaire (déjà structurée).\n"
        "À partir de ce texte, rédige un RAPPORT LONG, très détaillé (au moins 8 à 12 pages A4 si on l'imprime), structuré en parties et sous-parties.\n\n"
        "Plan recommandé (tu peux l'adapter légèrement si besoin) :\n"
        "1. Contexte et faits initiaux\n"
        "2. Déroulement de l'enquête (actes, investigations, décisions)\n"
        "3. Protagonistes (victimes, suspects, témoins) et rôles\n"
        "4. Éléments techniques (téléphonie, financier, scientifique, etc.)\n"
        "5. Garde à vue, déclarations et évolutions de versions\n"
        "6. Synthèse finale (bilan factuel, zones d'ombre, axes forts de l'enquête)\n\n"
        "RÈGLES :\n"
        "- Style : rapport, neutre, professionnel, en français.\n"
        "- Paragraphe courts, clairement structurés.\n"
        "- Pas de digressions théoriques, pas d'explications sur le droit.\n"
        "- Tu n'inventes rien : tu développes uniquement ce qui ressort des synthèses fournies.\n\n"
        "Synthèse intermédiaire de départ :\n"
        "---------------------------------\n"
        f"{section_text}\n"
    )


def _prompt_refine_long_report(case_title: str, current_report: str, new_section: str) -> str:
    """
    Prompt pour raffiner / enrichir un rapport long existant avec une nouvelle synthèse.
    Corrigé pour éviter les réponses du type « envoyez-moi le rapport provisoire » :
    même si current_report est vide ou très court, le modèle DOIT produire un rapport complet.
    """
    head = prompts.SYSTEM_FR
    title_part = f"Titre de l'affaire : {case_title}\n\n" if case_title else ""

    return (
        f"{head}\n"
        "Tu améliores et complètes un rapport judiciaire long à partir de nouvelles informations.\n"
        "On te donne :\n"
        "1) un RAPPORT PROVISOIRE (déjà long, structuré, ou parfois très court voire vide),\n"
        "2) une NOUVELLE SYNTHÈSE INTERMÉDIAIRE apportant des éléments complémentaires.\n\n"
        f"{title_part}"
        "TÂCHE :\n"
        "- Réécrire un rapport complet, en intégrant TOUTES les informations du rapport provisoire ET de la nouvelle synthèse.\n"
        "- Si le RAPPORT PROVISOIRE ACTUEL est très court ou vide, tu considères qu'il n'y en a pas\n"
        "  et tu construis le rapport à partir de la seule NOUVELLE SYNTHÈSE INTERMÉDIAIRE.\n"
        "- NE DEMANDE JAMAIS à l'utilisateur de renvoyer un document, un rapport ou un texte :\n"
        "  tu dois travailler UNIQUEMENT à partir des textes ci-dessous.\n"
        "- NE DIS JAMAIS que tu ne peux pas réaliser la tâche faute de rapport provisoire :\n"
        "  même partiels, les textes fournis suffisent.\n"
        "- Conserver un plan clair similaire :\n"
        "  1. Contexte et faits initiaux\n"
        "  2. Déroulement de l'enquête\n"
        "  3. Protagonistes\n"
        "  4. Éléments techniques\n"
        "  5. Garde à vue et déclarations\n"
        "  6. Synthèse finale\n"
        "- Allonger et détailler lorsqu'il y a de nouveaux éléments (sans paraphraser inutilement).\n"
        "- Si des contradictions apparaissent, les signaler clairement.\n"
        "- NE PAS réduire la richesse du rapport : le nouveau rapport doit être au moins aussi détaillé,\n"
        "  souvent plus, que le précédent.\n\n"
        "RAPPORT PROVISOIRE ACTUEL :\n"
        "---------------------------\n"
        f"{current_report}\n\n"
        "NOUVELLE SYNTHÈSE INTERMÉDIAIRE :\n"
        "---------------------------\n"
        f"{new_section}\n"
    )


def _refine_long_report(
    model: str,
    case_id: str,
    case_title: str,
    section_summaries: Sequence[str],
    progress_cb: ProgressCB,
    *,
    temperature: Optional[float],
    top_p: Optional[float],
) -> str:
    """
    REFINE : construit un rapport long incrémentalement à partir des synthèses intermédiaires.
    """
    if not section_summaries:
        return ""

    _safe_cb(progress_cb, {"type": "stage", "stage": "refine-long", "message": "Construction du rapport long..."})

    # 1er passage : on part d'une section pour initialiser le rapport
    first = section_summaries[0]
    prompt0 = _prompt_initial_long_report(case_title, first)
    current_report = _llm_long(
        model,
        prompt0,
        case_id=case_id,
        max_tokens=LONG_SUM_FINAL_MAX_TOKENS,
        temperature=temperature,
        top_p=top_p,
    )

    total = max(len(section_summaries), 1)
    for i, sec in enumerate(section_summaries[1:], start=2):
        progress_val = 0.60 + 0.30 * (i / total)
        _safe_cb(
            progress_cb,
            {
                "type": "progress",
                "value": progress_val,
                "stage": "refine-long",
                "section_index": i,
                "section_total": total,
            },
        )
        prompt = _prompt_refine_long_report(case_title, current_report, sec)
        current_report = _llm_long(
            model,
            prompt,
            case_id=case_id,
            max_tokens=LONG_SUM_FINAL_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
        )

    return current_report


# ==========================
#  Étape 4 — Finalisation légère
# ==========================

def _final_polish(
    model: str,
    case_id: str,
    case_title: str,
    report: str,
    progress_cb: ProgressCB,
    *,
    temperature: Optional[float],
    top_p: Optional[float],
) -> str:
    """
    Dernier passage : harmonisation légère du style, sans tout réécrire.
    """
    _safe_cb(progress_cb, {"type": "stage", "stage": "final-polish", "message": "Harmonisation finale du rapport..."})
    head = prompts.SYSTEM_FR
    title_part = f"Titre de l'affaire : {case_title}\n\n" if case_title else ""
    prompt = (
        f"{head}\n"
        "Tu vas relire le rapport judiciaire long suivant et :\n"
        "- harmoniser légèrement le style (phrases trop longues, petites répétitions),\n"
        "- améliorer la lisibilité (paragraphe, transitions),\n"
        "- corriger les petites fautes, sans changer le fond ni la structure générale.\n"
        "- ne PAS condenser ni raccourcir fortement : la longueur globale doit rester comparable.\n\n"
        f"{title_part}"
        "Rapport à relire :\n"
        "------------------\n"
        f"{report}\n"
    )
    fixed = _llm_long(
        model,
        prompt,
        case_id=case_id,
        max_tokens=LONG_SUM_FINAL_MAX_TOKENS,
        temperature=temperature,
        top_p=top_p,
    )
    _safe_cb(progress_cb, {"type": "progress", "value": 0.98, "stage": "final-polish"})
    # on garde le plus long des deux si la version finale est trop courte
    if fixed and len(fixed) > len(report) * 0.6:
        return fixed
    return report


# ==========================
#  FONCTION PUBLIQUE
# ==========================

def summarize_case_long(
    case_id: str,
    chunks: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
    progress_cb: ProgressCB = None,
    *,
    case_title: Optional[str] = None,
) -> str:
    """
    Synthèse longue hiérarchique d'une affaire.

    - Utilise une stratégie en plusieurs passes :
      1) MAP    : résumés par blocs (~10–30 pages)
      2) REDUCE : synthèses intermédiaires par groupes de blocs
      3) REFINE : rapport long incrémental
      4) POLISH : harmonisation légère

    - Le modèle par défaut est OLLAMA_DEFAULT_MODEL (ou deepseek si configuré).
    - chunks : liste de dicts {"text": "...", "meta": {...}} (format bm25.json -> summarizer).
    """
    if not model:
        # modèle par défaut : DeepSeek cloud (si OLLAMA_DEFAULT_MODEL n'est pas défini)
        model = os.getenv("OLLAMA_DEFAULT_MODEL", "deepseek-v3.1:671b-cloud")

    if case_title is None:
        case_title = DEFAULT_CASE_TITLE or case_id

    # Profil LLM spécifique pour la synthèse longue
    profile = get_profile("long_summary")
    temperature = profile.get("temperature")
    top_p = profile.get("top_p")

    logger.info(
        "summarizer_long: config LONG_SUM_BLOCK_CHARS=%d, LOCAL_MAX=%d, SECTION_MAX=%d, FINAL_MAX=%d, temp=%.3f, top_p=%.3f",
        LONG_SUM_BLOCK_CHARS,
        LONG_SUM_LOCAL_MAX_TOKENS,
        LONG_SUM_SECTION_MAX_TOKENS,
        LONG_SUM_FINAL_MAX_TOKENS,
        temperature if temperature is not None else -1.0,
        top_p if top_p is not None else -1.0,
    )

    _safe_cb(progress_cb, {"type": "start", "case_id": case_id, "stage": "start-long"})

    # Sécurité : si aucun chunk, on répond vite
    if not chunks:
        msg = f"(Synthèse longue) Aucun texte disponible pour l'affaire {case_id}."
        _safe_cb(progress_cb, {"type": "progress", "value": 1.0, "stage": "no-data"})
        _safe_cb(progress_cb, {"type": "end", "case_id": case_id})
        return msg

    # 1) MAP : blocs -> résumés locaux
    blocks = _split_into_blocks(chunks, max_chars=LONG_SUM_BLOCK_CHARS)
    logger.info("summarizer_long: case %s -> %d blocs pour MAP", case_id, len(blocks))
    block_summaries = _map_block_summaries(
        model,
        case_id,
        case_title,
        blocks,
        progress_cb,
        temperature=temperature,
        top_p=top_p,
    )

    # 2) REDUCE : résumés locaux -> synthèses intermédiaires
    section_summaries = _reduce_sections(
        model,
        case_id,
        case_title,
        block_summaries,
        progress_cb,
        temperature=temperature,
        top_p=top_p,
    )

    # 3) REFINE : synthèses intermédiaires -> rapport long
    long_report = _refine_long_report(
        model,
        case_id,
        case_title,
        section_summaries,
        progress_cb,
        temperature=temperature,
        top_p=top_p,
    )

    # 4) POLISH : harmonisation
    final_report = _final_polish(
        model,
        case_id,
        case_title,
        long_report,
        progress_cb,
        temperature=temperature,
        top_p=top_p,
    )

    _safe_cb(progress_cb, {"type": "progress", "value": 1.0, "stage": "done"})
    _safe_cb(progress_cb, {"type": "end", "case_id": case_id})

    return final_report


# Alias pratique si tu veux importer sous un autre nom
summarize_enquete_long = summarize_case_long
