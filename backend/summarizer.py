# backend/summarizer.py
# Outils communs de synthèse
# Auteur: Jax

import os
import re
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import prompts  # backend/prompts.py
import ollama_client  # backend/ollama_client.py

logger = logging.getLogger(__name__)

# type du callback de progression (SSE côté routes_leaf)
ProgressCB = Optional[Callable[[Dict[str, Any]], None]]

# taille max qu'on laisse passer dans un prompt de section
MAX_SECTION_CTX_CHARS = int(os.getenv("SUMMARY_MAX_SECTION_CTX", "12000"))


# ---------------------------------------------------------------------------
# helpers callback
# ---------------------------------------------------------------------------

def _safe_cb(cb: ProgressCB, payload: Dict[str, Any]) -> None:
    if cb is None:
        return
    try:
        cb(payload)
    except Exception:
        logger.exception("progress callback failed: %r", payload)


# ---------------------------------------------------------------------------
# helpers texte
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufeff", "").strip()
    # parfois le modèle renvoie ```markdown ...```
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    return text


def _looks_truncated(text: str) -> bool:
    if not text:
        return True
    if len(text) < 200:
        return True
    t = text.rstrip()
    if t.endswith("##") or t.endswith("###"):
        return True
    if t.endswith(":") or t.endswith("-"):
        return True
    if not t.endswith((".", "!", "?", "»")):
        return True
    return False


# ---------------------------------------------------------------------------
# sélecteur de contexte par section
# ---------------------------------------------------------------------------

def _select_relevant_chunks_for_section(
    section_title: str,
    chunks: Sequence[Dict[str, Any]],
) -> str:
    """
    On prend les mots-clés définis dans prompts.LEAF_SECTION_QUERIES pour cette section
    et on assemble seulement les passages pertinents.
    """
    queries = getattr(prompts, "LEAF_SECTION_QUERIES", {}).get(section_title, [])
    if not queries:
        texts = [c.get("text") or c.get("content") or "" for c in chunks]
        joined = "\n\n".join(t for t in texts if t)
        return joined[:MAX_SECTION_CTX_CHARS]

    patterns = [re.compile(re.escape(q), re.IGNORECASE) for q in queries]
    selected: List[str] = []
    for ch in chunks:
        txt = ch.get("text") or ch.get("content") or ""
        if not txt:
            continue
        if any(p.search(txt) for p in patterns):
            selected.append(txt)

    if not selected:
        texts = [c.get("text") or c.get("content") or "" for c in chunks]
        joined = "\n\n".join(t for t in texts if t)
        return joined[:MAX_SECTION_CTX_CHARS]

    joined = "\n\n".join(selected)
    return joined[:MAX_SECTION_CTX_CHARS]


# ---------------------------------------------------------------------------
# consignes spécifiques par section du plan LEAF
# ---------------------------------------------------------------------------

SECTION_HINTS: Dict[str, str] = {
    "I. Les faits": (
        "Expose de manière claire les faits initiaux : ce qui est reproché, le contexte, le lieu, la date, "
        "les victimes, la nature de l’infraction. Mets en avant les éléments matériels (objets, montants, "
        "armes, véhicules, lieux) et le déclenchement de l’enquête (plainte, signalement, flagrant délit, etc.)."
    ),
    "II. Constatations": (
        "Rassemble les constatations matérielles et procédurales : PV de constatations, premières observations sur place, "
        "description de la scène, scellés, relevés, photos, traces et indices. Reste très factuel."
    ),
    "III. Recherches": (
        "Décris les recherches entreprises : réquisitions, vérifications, exploitation de vidéosurveillance, recherches de personnes, "
        "exploitation de fichiers (police, téléphonie, bancaire…), recherches infructueuses si elles sont notables."
    ),
    "IV. Témoignages": (
        "Synthétise les auditions de témoins : qui parle, sur quoi, quelles informations clés ils apportent, "
        "convergences ou contradictions importantes entre déclarations."
    ),
    "V. Investigations téléphoniques": (
        "Expose les investigations téléphoniques : fadettes, relevés détaillés, bornages, analyses de trafic (MSISDN/IMEI/IMSI), "
        "liens entre numéros, localisations, volumes d’appels, éventuelles mises en évidence de réseaux."
    ),
    "VI. Écoutes téléphoniques": (
        "Résume les écoutes/interceptions téléphoniques : principaux échanges significatifs, codes de langage, "
        "éléments confirmant ou infirmant les hypothèses, passages clés utiles à l’enquête."
    ),
    "VII. Investigations scientifiques (ADN, empreintes, etc.)": (
        "Expose les éléments de police scientifique : traces ADN, empreintes papillaires, analyses balistiques, "
        "rapports de laboratoire, résultats d’examens techniques. Précise à quoi ces analyses sont reliées (scellés, lieux, personnes)."
    ),
    "VIII. Suspects": (
        "Présente les principaux suspects / mis en cause : identité, rôle supposé, éléments à charge (et à décharge si clairs), "
        "liens entre eux et avec les victimes. Reste factuel : pas de qualification juridique ni d’appréciation de culpabilité."
    ),
    "IX. Gardes à vue": (
        "Dresse la liste des gardes à vue : qui, quand, pour quels faits, durée, prolongations. "
        "Mentionne les actes procéduraux associés (notifications de droits, désignation d’avocat, etc.)."
    ),
    "X. Auditions des suspects": (
        "Résume les auditions des suspects : contenu essentiel des déclarations, éléments nouveaux, aveux ou dénégations, "
        "évolutions de versions. Précise si la personne reconnaît les faits entièrement, partiellement, ou les nie."
    ),
    "XI. Information judiciaire": (
        "Expose les actes de l’information judiciaire : mise en examen, constitution de partie civile, réquisitoires, ordonnances, "
        "actes du juge d’instruction ayant un impact sur le dossier."
    ),
    "XII. Expertises techniques et scientifiques": (
        "Présente les expertises (techniques, scientifiques, comptables, psychologiques, etc.) : objet, conclusions principales, "
        "impact sur la compréhension des faits ou sur la position des protagonistes."
    ),
    "XIII. Interrogatoires": (
        "Résume les interrogatoires (notamment devant juge d’instruction) : posture des mis en examen, "
        "évolutions de leurs déclarations par rapport aux gardes à vue, points d’achoppement ou éléments nouveaux."
    ),
    "XIV. CRT (balises, sonorisations, écoutes, interception DATA)": (
        "Expose les mesures techniques spéciales (balises, sonorisations, interceptions de données) : mise en place, "
        "principaux enseignements, déplacements importants, rencontres, conversations clés. "
        "Reste factuel et centré sur ce qui fait avancer la compréhension de l’affaire."
    ),
}


def _build_section_prompt(
    *,
    section_title: str,
    case_summary: str,
    section_ctx: str,
) -> str:
    """
    Construit le prompt de rédaction pour UNE section du plan LEAF.

    On rappelle au modèle :
    - le plan complet (I → XIV),
    - la section courante à rédiger,
    - un résumé global de l’affaire,
    - le contexte filtré pour cette section,
    - des consignes spécifiques à la section si disponibles.
    """
    head = prompts.SYSTEM_FR

    # plan complet LEAF (celui que tu as défini dans prompts.LEAF_FIXED_OUTLINE)
    outline = getattr(prompts, "LEAF_FIXED_OUTLINE", [])
    if outline:
        plan_str = "Plan complet de la synthèse (plan LEAF) :\n"
        for t in outline:
            plan_str += f"- {t}\n"
        plan_str += "\n"
    else:
        plan_str = ""

    # consignes spécifiques selon la section
    section_hint = SECTION_HINTS.get(section_title, "").strip()
    if section_hint:
        specific_block = (
            "Consigne spécifique pour cette section :\n"
            f"- {section_hint}\n\n"
        )
    else:
        specific_block = ""

    return (
        f"{head}\n"
        f"{plan_str}"
        f"Tu rédiges maintenant UNIQUEMENT la section suivante d'une synthèse judiciaire complète : **{section_title}**.\n"
        "La synthèse suit le plan LEAF ci-dessus ; tu n'écris que cette section, pas les autres.\n"
        "Ta rédaction doit être structurée, factuelle, en français, avec un ton neutre et professionnel.\n"
        "Si des éléments manquent dans le contexte, signale-le brièvement en fin de section.\n\n"
        f"{specific_block}"
        "Rappel synthétique de l'affaire :\n"
        "-----------------------------\n"
        f"{case_summary}\n"
        "-----------------------------\n\n"
        "Éléments du dossier potentiellement liés à cette section :\n"
        "-----------------------------\n"
        f"{section_ctx}\n"
        "-----------------------------\n\n"
        "Règles de rédaction :\n"
        "- commence directement par le contenu de la section (pas d’introduction générale sur l’ensemble du dossier),\n"
        "- utilise des sous-parties ou des paragraphes courts si nécessaire,\n"
        "- cite les pièces ou sources quand tu peux les déduire : [source: nom_fichier, chunk N],\n"
        "- n'invente pas de pièces ni de faits,\n"
        "- n'écris pas les autres sections du plan LEAF.\n"
    )


def _build_case_summary_prompt(chunks: Sequence[Dict[str, Any]]) -> str:
    all_texts = [c.get("text") or c.get("content") or "" for c in chunks]
    joined = "\n\n".join(t for t in all_texts if t)
    joined = joined[: min(len(joined), prompts.MAX_CTX_CHARS)]
    return (
        f"{prompts.SYSTEM_FR}\n"
        "Voici un dossier d'enquête découpé en extraits. Fais un résumé synthétique (10–15 lignes) "
        "qui rappelle l'affaire, les acteurs, les faits saillants et les éléments techniques majeurs. "
        "Ce résumé sera utilisé ensuite pour écrire chaque section du plan LEAF (I à XIV).\n\n"
        "Dossier (extraits) :\n"
        "---------------------\n"
        f"{joined}\n"
    )


# ---------------------------------------------------------------------------
# helpers anonymisation / dé-pseudonymisation
# ---------------------------------------------------------------------------

try:
    from identity_mapper import anonymize_prompt_for_model, deanonymize_text_for_case
except Exception:
    # fallback silencieux si le module est absent
    def anonymize_prompt_for_model(model, prompt, case_id=None):
        return prompt, {"sanitized": False}

    def deanonymize_text_for_case(text, case_id=None):
        return text


# ---------------------------------------------------------------------------
# LLM wrappers (anonymisation + dé-pseudonymisation)
# ---------------------------------------------------------------------------

def _llm_small(model: str, prompt: str, case_id: Optional[str]) -> str:
    """
    Appel LLM pour les petits résumés (global_summary),
    SANS anonymisation/dépseudonymisation.
    """
    txt = ollama_client.generate(model, prompt)
    return _normalize(txt)


def _llm_long(
    model: str,
    prompt: str,
    *,
    case_id: Optional[str],
    max_tokens: int = 4096,
) -> str:
    """
    Appel LLM pour les sections / finalisation,
    SANS anonymisation/dépseudonymisation.
    """
    txt = ollama_client.generate_long(model, prompt, max_tokens=max_tokens)
    return _normalize(txt)


# ---------------------------------------------------------------------------
# étapes de la synthèse
# ---------------------------------------------------------------------------

def _make_global_summary(
    model: str,
    case_id: str,
    chunks: Sequence[Dict[str, Any]],
    progress_cb: ProgressCB,
) -> str:
    _safe_cb(progress_cb, {"type": "stage", "message": "Résumé global du dossier..."})
    prompt = _build_case_summary_prompt(chunks)
    summary = _llm_small(model, prompt, case_id)
    _safe_cb(progress_cb, {"type": "progress", "value": 0.08, "stage": "global-summary"})
    return summary


def _get_outline() -> List[str]:
    outline = getattr(prompts, "LEAF_FIXED_OUTLINE", None)
    if outline is None:
        logger.warning("prompts.LEAF_FIXED_OUTLINE absent — utilisation d'un outline par défaut")
        outline = [
            "1. Résumé exécutif",
            "2. Contexte et pièces",
            "3. Chronologie",
            "4. Personnes impliquées",
            "5. Éléments matériels",
            "6. Sources",
        ]
    return list(outline)


def _generate_section(
    model: str,
    case_id: str,
    section_title: str,
    global_summary: str,
    chunks: Sequence[Dict[str, Any]],
    idx: int,
    total: int,
    progress_cb: ProgressCB,
) -> str:
    _safe_cb(
        progress_cb,
        {
            "type": "stage",
            "message": f"Section {idx}/{total} : {section_title}",
        },
    )

    section_ctx = _select_relevant_chunks_for_section(section_title, chunks)
    prompt = _build_section_prompt(
        section_title=section_title,
        case_summary=global_summary,
        section_ctx=section_ctx,
    )

    text = _llm_long(model, prompt, case_id=case_id, max_tokens=4096)

    if _looks_truncated(text):
        ext_prompt = (
            f"{prompts.SYSTEM_FR}\n"
            f"Le texte suivant est la section '{section_title}' d'une synthèse judiciaire. "
            "Complète-la en restant dans le même style, sans ouvrir de nouvelle section et sans réécrire le début.\n\n"
            "Texte à compléter :\n"
            "-------------------\n"
            f"{text}\n"
        )
        ext = _llm_long(model, ext_prompt, case_id=case_id, max_tokens=2048)
        text = (text + "\n" + ext).strip()

    base = 0.10
    ratio = base + 0.80 * (idx / max(total, 1))
    _safe_cb(
        progress_cb,
        {
            "type": "progress",
            "value": ratio,
            "stage": "sections",
        },
    )

    return f"## {section_title}\n\n{text}\n"


def _expand_final(model: str, case_id: str, doc: str, progress_cb: ProgressCB) -> str:
    _safe_cb(progress_cb, {"type": "stage", "message": "Finalisation de la synthèse..."})
    prompt = (
        f"{prompts.SYSTEM_FR}\n"
        "Tu vas relire la synthèse judiciaire suivante et :\n"
        "- harmoniser le style (même ton sur toutes les sections),\n"
        "- corriger les petites ruptures,\n"
        "- NE PAS supprimer de sections ni de titres existants,\n"
        "- NE PAS traduire, rester en français,\n"
        "- NE PAS réorganiser le plan LEAF.\n\n"
        "Synthèse :\n"
        "----------\n"
        f"{doc}\n"
    )
    fixed = _llm_long(model, prompt, case_id=case_id, max_tokens=2048)
    _safe_cb(progress_cb, {"type": "progress", "value": 0.98, "stage": "final"})
    return fixed


# ---------------------------------------------------------------------------
# FONCTION PUBLIQUE
# ---------------------------------------------------------------------------

def summarize_leaf_long(
    case_id: str,
    chunks: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
    progress_cb: ProgressCB = None,
    *,
    do_expand: bool = True,
) -> str:
    if not model:
        # modèle par défaut : DeepSeek cloud si non défini dans l'environnement
        model = os.getenv("OLLAMA_DEFAULT_MODEL", "deepseek-v3.1:671b-cloud")

    _safe_cb(progress_cb, {"type": "start", "case_id": case_id})

    # Résumé global avec pseudonymisation basée sur case_id
    global_summary = _make_global_summary(model, case_id, chunks, progress_cb)
    outline = _get_outline()

    sections_md: List[str] = []
    total = len(outline)
    for i, title in enumerate(outline, 1):
        sec_md = _generate_section(
            model,
            case_id,
            title,
            global_summary,
            chunks,
            i,
            total,
            progress_cb,
        )
        sections_md.append(sec_md)

    doc = "# Synthèse judiciaire (plan LEAF)\n\n" + "\n".join(sections_md)

    if do_expand:
        improved = _expand_final(model, case_id, doc, progress_cb)
        if improved and len(improved) > len(doc) * 0.6:
            doc = improved

    _safe_cb(progress_cb, {"type": "progress", "value": 1.0, "stage": "done"})
    _safe_cb(progress_cb, {"type": "end", "case_id": case_id})

    return doc
