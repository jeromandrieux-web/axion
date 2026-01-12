# backend/summarizer_chrono.py
# Moteur de synthèse chronologique (PV & rapports)
# Auteur: Jax

"""
summarizer_chrono.py — moteur de synthèse chronologique (PV & rapports)

Expose :
- summarize_chrono_long(...): str   → génère la synthèse chronologique détaillée (Markdown)
- summarize_chrono_short(...): str  → génère un rapport court mais costaud (≈ 5 parties)

Principe (version simplifiée et robuste) :
- On lit meta.json + bm25.json de l’affaire.
- On regroupe les chunks par document (PV, rapports, expertises, etc.).
- On trie les documents par date (meta.date).
- Pour chaque document :
    → on fabrique un extrait (concat chunks, tronqués),
    → on appelle directement le LLM via ollama_client,
    → on produit un résumé de l’acte.
- On EMPILE tous ces résumés dans un seul Markdown chronologique, sans autres raffinements.

La synthèse courte (“rapport de synthèse”) se fait ensuite à partir de ce Markdown.
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

from collections import defaultdict

# Prompts chrono
try:
    from backend.prompts_chrono import (
        CHRONO_DOC_SYSTEM,
        CHRONO_SHORT_REPORT_SYSTEM,
        CHRONO_DOC_STYLE,
        build_doc_user_prompt,
        build_short_report_user_prompt,
    )
except Exception:
    from prompts_chrono import (  # type: ignore
        CHRONO_DOC_SYSTEM,
        CHRONO_SHORT_REPORT_SYSTEM,
        CHRONO_DOC_STYLE,
        build_doc_user_prompt,
        build_short_report_user_prompt,
    )

# Client LLM direct (Ollama)
try:
    from backend import ollama_client
except Exception:
    import ollama_client  # type: ignore

# Ajout explicite (sécurise l'accès direct plus bas)
import ollama_client

logger = logging.getLogger(__name__)

STORAGE_ROOT = Path("backend/storage").resolve()

# ---------------------------------------------------------------------------
# Paramètres globaux (pilotables depuis .env)
# ---------------------------------------------------------------------------

# Limite par défaut de docs pour la chrono longue (0 = pas de limite automatique)
CHRONO_LONG_DEFAULT_MAX_DOCS = int(os.getenv("CHRONO_LONG_MAX_DOCS", "0"))
CHRONO_SHORT_DEFAULT_MAX_DOCS = int(os.getenv("CHRONO_SHORT_MAX_DOCS", "0"))

# Taille max de l'extrait par document pour la synthèse chrono
CHRONO_EXCERPT_CHARS = int(os.getenv("CHRONO_EXCERPT_CHARS", "8000"))

# Budget tokens pour chaque résumé d'acte
CHRONO_DOC_MAX_TOKENS = int(os.getenv("CHRONO_DOC_MAX_TOKENS", "420"))

# Budgets pour le rapport court
CHRONO_SHORT_BASE_MAX_TOKENS = int(os.getenv("CHRONO_SHORT_BASE_MAX_TOKENS", "2048"))
CHRONO_SHORT_MINI_MAX_TOKENS = int(os.getenv("CHRONO_SHORT_MINI_MAX_TOKENS", "900"))
CHRONO_SHORT_FINAL_MAX_TOKENS = int(os.getenv("CHRONO_SHORT_FINAL_MAX_TOKENS", "3072"))

# Segmentation du gros Markdown chrono pour le rapport court
CHRONO_TIMELINE_SEGMENT_CHARS = int(os.getenv("CHRONO_TIMELINE_SEGMENT_CHARS", "12000"))

# Modèle par défaut global (utilisé par le wrapper local)
DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "deepseek-v3.1:671b-cloud")

def _looks_truncated(text: str) -> bool:
    """
    Heuristique simple : détecte si un texte semble coupé en plein milieu.
    (inspiré de summarizer.py)
    """
    if not text:
        return True
    if len(text) < 200:
        # texte très court : on le laisse tranquille
        return False
    t = text.rstrip()
    # se termine par un titre ou un signe qui laisse penser qu'il manque la suite
    if t.endswith("##") or t.endswith("###"):
        return True
    if t.endswith(":") or t.endswith("-"):
        return True
    # pas de ponctuation finale "forte"
    if not t.endswith((".", "!", "?", "»")):
        return True
    return False


@dataclass
class DocMeta:
    name: str
    date: Optional[str]
    doc_type: Optional[str]
    pages: int
    chunks: int
    added: float


@dataclass
class DocBundle:
    meta: DocMeta
    chunks: List[str]              # texte des chunks ordonnés
    refs: List[Tuple[int, int]]    # (page, local_chunk_id)


# Label lisible pour les types
DOC_TYPE_LABEL = {
    "pv_audition": "PV d’audition",
    "pv_gav": "PV de GAV",
    "pv": "Procès-verbal",
    "rapport": "Rapport",
    "expertise": "Expertise",
    "ecoutes": "Transcription/Écoutes",
    "constatations": "Constatations",
}


# ---------------------------------------------------------------------------
# Helpers cas / dates / bundles
# ---------------------------------------------------------------------------

def _load_case_files(case_id: str) -> Tuple[Dict, Dict]:
    case_dir = STORAGE_ROOT / "cases" / case_id
    meta = json.loads((case_dir / "meta.json").read_text(encoding="utf-8"))
    bm25 = json.loads((case_dir / "bm25.json").read_text(encoding="utf-8"))
    return meta, bm25


def _iso_to_dt(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str)
    except Exception:
        return None


def _within(dt: Optional[datetime], date_from: Optional[str], date_to: Optional[str]) -> bool:
    if dt is None:
        return True
    if date_from:
        try:
            if dt < datetime.fromisoformat(date_from):
                return False
        except Exception:
            pass
    if date_to:
        try:
            if dt > datetime.fromisoformat(date_to):
                return False
        except Exception:
            pass
    return True


def _bundle_docs(meta: Dict, bm25: Dict) -> List[DocBundle]:
    """
    Regroupe les chunks par document (source = nom de fichier).
    """
    chunks: List[str] = bm25.get("chunks", [])
    sources: List[Dict] = bm25.get("sources", [])

    by_name: Dict[str, DocMeta] = {}
    for d in meta.get("docs", []):
        by_name[d["name"]] = DocMeta(
            name=d["name"],
            date=d.get("date"),
            doc_type=d.get("doc_type"),
            pages=d.get("pages", 0),
            chunks=d.get("chunks", 0),
            added=d.get("added", 0.0),
        )

    tmp: Dict[str, DocBundle] = {}

    for idx, text in enumerate(chunks):
        src = sources[idx] if idx < len(sources) else {}
        name = src.get("source") or "__unknown__"
        page = int(src.get("page", 1))
        lcid = int(src.get("local_chunk_id", idx))

        if name not in by_name:
            by_name[name] = DocMeta(
                name=name,
                date=src.get("date"),
                doc_type=src.get("doc_type"),
                pages=0,
                chunks=0,
                added=0.0,
            )
        if name not in tmp:
            tmp[name] = DocBundle(meta=by_name[name], chunks=[], refs=[])
        tmp[name].chunks.append(text)
        tmp[name].refs.append((page, lcid))

    bundles: List[DocBundle] = []
    for b in tmp.values():
        order = sorted(range(len(b.chunks)), key=lambda i: (b.refs[i][0], b.refs[i][1]))
        b.chunks = [b.chunks[i] for i in order]
        b.refs = [b.refs[i] for i in order]
        bundles.append(b)

    return bundles


def _split_timeline_md(timeline_md: str, max_chars: int = CHRONO_TIMELINE_SEGMENT_CHARS) -> List[str]:
    """
    Découpe un gros markdown de chronologie en plusieurs segments
    (double saut de ligne comme séparateur “souple”).
    """
    blocks = timeline_md.split("\n\n")
    chunks: List[str] = []
    current: List[str] = []
    cur_len = 0

    for block in blocks:
        piece = block + "\n\n"
        if cur_len + len(piece) > max_chars and current:
            chunks.append("".join(current).strip())
            current = [piece]
            cur_len = len(piece)
        else:
            current.append(piece)
            cur_len += len(piece)

    if current:
        chunks.append("".join(current).strip())

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Wrapper LLM simple (pas de pseudonymisation ici)
# ---------------------------------------------------------------------------

def _call_llm(
    system: str,
    user: str,
    *,
    model: Optional[str],
    max_tokens: int,
    temperature: float = 0.2,
) -> str:
    """
    Appel direct à Ollama via ollama_client.generate_long.
    """
    if not model:
        model = os.getenv("OLLAMA_DEFAULT_MODEL", "deepseek-v3.1:671b-cloud")

    prompt = (system or "").rstrip() + "\n\n" + (user or "")
    txt = ollama_client.generate_long(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (txt or "").strip()


# ---------------------------------------------------------------------------
# Wrapper local pour le rapport court (Ollama direct)
# ---------------------------------------------------------------------------
def _local_short_report_llm(system: str, user: str, **kw) -> str:
    """
    Wrapper local pour le rapport de synthèse court :
    on bypass /api/chat et on appelle directement Ollama.
    """
    model = kw.get("model") or DEFAULT_MODEL
    case_id = kw.get("case_id")

    # On concatène system + user comme pour les autres prompts
    prompt = f"{system}\n\n{user}" if system else user

    # Tentative d'anonymisation si disponible (ne plante pas si absent)
    try:
        sanitized, _meta = anonymize_prompt_for_model(model, prompt, case_id)
    except Exception:
        sanitized = prompt

    txt = ollama_client.generate_long(
        model,
        sanitized,
        max_tokens=kw.get("max_tokens", CHRONO_SHORT_BASE_MAX_TOKENS),
        temperature=kw.get("temperature", 0.2),
    )

    # Tentative de dé-anonymisation si disponible (silencieuse en échec)
    try:
        txt = deanonymize_text_for_case(txt or "", case_id)
        # fallback : parfois on veut dé-anonymiser sans case_id
        if txt == (txt or "") and case_id:
            txt = deanonymize_text_for_case(txt or "", None)
    except Exception:
        pass

    return (txt or "").strip()


# ---------------------------------------------------------------------------
# Synthèse chronologique LONGUE (empilage doc par doc)
# ---------------------------------------------------------------------------

def summarize_chrono_long(
    case_id: str,
    *,
    case_title: str = "",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    include_monthly: bool = False,  # désormais ignoré : empilement pur
    max_docs: Optional[int] = None,
    excerpt_chars: int = CHRONO_EXCERPT_CHARS,
    llm: Optional[Callable[[str, str], str]] = None,
    llm_kwargs: Optional[Dict] = None,
    progress_cb: Optional[Callable[[str, Dict], None]] = None,
) -> str:
    """
    Génère une synthèse chronologique Markdown pour `case_id` en mode EMPILEMENT :

    - Chaque document (PV, rapport, etc.) est résumé individuellement par le LLM.
    - Les résumés sont ensuite triés par date croissante.
    - On empile tous ces résumés dans un seul document Markdown, un bloc par acte/document.

    ⚠️ Aucun passage de "méga-synthèse" ou de synthèse mensuelle :
       il s'agit d'une chronologie détaillée, acte par acte.
    """
    if llm is None:
        raise RuntimeError("summarize_chrono_long nécessite un callable `llm(system, user, **kwargs)`")

    llm_kwargs = llm_kwargs or {}

    def emit(stage: str, payload: Dict | None = None):
        """
        Wrapper local pour la progression :
        - si progress_cb est fourni (routes_chrono), on l'appelle avec (stage, payload).
        """
        if progress_cb:
            try:
                progress_cb(stage, payload or {})
            except Exception:
                # On ne casse pas la synthèse si le callback plante
                pass

    # Chargement des fichiers d'affaire
    meta, bm25 = _load_case_files(case_id)
    bundles = _bundle_docs(meta, bm25)

    # Filtrage temporel & tri
    def key_date(b: DocBundle):
        dt = _iso_to_dt(b.meta.date)
        # None va en bas
        return (dt or datetime.max, b.meta.added or 0.0, b.meta.name)

    bundles.sort(key=key_date)
    bundles = [b for b in bundles if _within(_iso_to_dt(b.meta.date), date_from, date_to)]

    # Plafond éventuel via .env si max_docs non fourni
    if max_docs is None and CHRONO_LONG_DEFAULT_MAX_DOCS > 0:
        max_docs = CHRONO_LONG_DEFAULT_MAX_DOCS
    if max_docs:
        bundles = bundles[:max_docs]

    emit("index", {"total_docs": len(bundles)})

    # ───────────────────────────────────────────────
    #  1) Résumés LLM par document (acte)
    # ───────────────────────────────────────────────
    doc_summaries: List[Dict] = []

    for i, b in enumerate(bundles, start=1):
        display_name = b.meta.name
        label = DOC_TYPE_LABEL.get(b.meta.doc_type or "", "Document")
        excerpt = ("\n\n".join(b.chunks))[:excerpt_chars]

        # Construire le prompt utilisateur pour un seul document
        user = build_doc_user_prompt(
            case_title=case_title or meta.get("title") or f"Affaire {case_id}",
            display_name=display_name,
            doc_type_label=label,
            doc_date=b.meta.date,
            pages=b.meta.pages,
            doc_excerpt=excerpt,
        )

        # Appel LLM : on passe directement par le callable fourni (llm_call)
        summary = llm(
            CHRONO_DOC_SYSTEM,
            user,
            **(
                {
                    "max_tokens": CHRONO_DOC_STYLE.get("max_tokens", 420),
                    "temperature": CHRONO_DOC_STYLE.get("temperature", 0.2),
                }
                | llm_kwargs
            ),
        )

        doc_summaries.append(
            {
                "year": (b.meta.date or "9999")[:4],
                "date": b.meta.date,
                "label": label,
                "name": display_name,
                "summary": (summary or "").strip(),
            }
        )

        emit(
            "doc_summary_progress",
            {"done": i, "total": len(bundles), "doc": display_name},
        )

    # ───────────────────────────────────────────────
    #  2) Tri strict par date, puis par nom
    # ───────────────────────────────────────────────
    items = sorted(
        doc_summaries,
        key=lambda s: (s["date"] or "9999-12-31", s["name"]),
    )

    # ------------------------------------------------------------------
    # Construction du Markdown — version simple "empilement chronologique"
    # ------------------------------------------------------------------
    # On trie tous les documents par date (puis par nom) et on les enchaîne
    # dans un seul document, sans synthèses mensuelles ni regroupements.
    all_items = sorted(
        doc_summaries,
        key=lambda s: (s["date"] or f"{s['year']}-12-31", s["name"]),
    )

    lines: List[str] = []
    title = case_title or meta.get("title") or f"Affaire {case_id}"

    lines.append(f"# Synthèse chronologique de l'affaire {title}")
    lines.append("")
    lines.append(
        "_Chaque document horodaté est résumé séparément et listé dans l'ordre "
        "chronologique des actes de procédure._"
    )
    lines.append("")

    for it in all_items:
        date_str = it["date"] or "(date inconnue)"
        label = it["label"]
        name = it["name"]
        summary = it["summary"]

        # Titre de bloc pour ce document
        lines.append(f"## {date_str} — {label} — {name}")
        lines.append("")
        # Résumé du document
        lines.append(summary)
        lines.append("")
        # Référence simple
        lines.append(f"*Référence : {name}*")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Rapport court (“rapport de synthèse”) basé sur la chronologie
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# build_short_report_from_timeline (avec gestion de complétion si tronqué)
# ---------------------------------------------------------------------------
def build_short_report_from_timeline(
    *,
    case_title: str,
    timeline_md: str,
    llm: Callable[[str, str], str],
    llm_kwargs: Optional[Dict] = None,
) -> str:
    """
    Prend un markdown de chronologie détaillée (résultat de summarize_chrono_long avec include_monthly=False)
    et produit un rapport court structuré en 5 parties :

      1. Les faits
      2. L’enquête
      3. Les suspects
      4. Les arrestations et garde à vue
      5. Conclusion

    On utilise le budget de tokens MAX pour limiter les coupures,
    puis on ajoute, si besoin, un SECOND PASS de complétion pour finir le texte
    sans réécrire le début.
    """
    kw = dict(llm_kwargs or {})

    # On utilise le budget maximal dispo pour le rapport complet
    kw.setdefault(
        "max_tokens",
        CHRONO_SHORT_FINAL_MAX_TOKENS or CHRONO_SHORT_BASE_MAX_TOKENS,
    )
    kw.setdefault("temperature", 0.2)

    user = build_short_report_user_prompt(
        case_title=case_title,
        timeline_md=timeline_md,
    )

    # 1) Rapport principal
    report = llm(CHRONO_SHORT_REPORT_SYSTEM, user, **kw)
    text = (report or "").strip()

    # 2) Si le rapport a l'air tronqué, on demande une complétion
    if _looks_truncated(text):
        # petit budget pour la complétion, mais on peut encore être généreux
        ext_kw = dict(kw)
        ext_kw["max_tokens"] = min(
            CHRONO_SHORT_FINAL_MAX_TOKENS or 2048,
            1024,
        )

        ext_user = (
            "Le texte suivant est un rapport de synthèse judiciaire structuré en 5 parties "
            "(Les faits, L’enquête, Les suspects, Arrestations et garde à vue, Conclusion).\n"
            "Il est coupé avant la fin. Complète-le en reprenant exactement le même plan et le même style, "
            "sans réécrire le début, en continuant simplement là où il s'arrête.\n\n"
            "Texte à compléter :\n"
            "-------------------\n"
            f"{text}\n"
        )

        try:
            ext = llm(CHRONO_SHORT_REPORT_SYSTEM, ext_user, **ext_kw)
            ext = (ext or "").strip()
            if ext:
                text = (text + "\n\n" + ext).strip()
        except Exception:
            # si la complétion échoue, on garde juste le texte principal
            pass

    return text


def summarize_chrono_short(
    case_id: str,
    *,
    case_title: str = "",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    max_docs: Optional[int] = None,
    excerpt_chars: int = CHRONO_EXCERPT_CHARS,
    llm: Optional[Callable[[str, str], str]] = None,
    llm_kwargs: Optional[Dict] = None,
    progress_cb: Optional[Callable[[str, Dict], None]] = None,
) -> str:
    """
    Variante courte de la synthèse chronologique, en plusieurs passes :

    1) Construit une chronologie détaillée (sans synthèse mensuelle) via summarize_chrono_long().
    2) Découpe cette chronologie en segments (max_chars).
    3) Pour chaque segment, produit un mini-rapport structuré (5 parties).
    4) Fusionne ces mini-rapports en un rapport final structuré(~5 pages) avec le plan :
       - 1. Les faits
       - 2. L’enquête
       - 3. Les suspects
       - 4. Les arrestations et garde à vue
       - 5. Conclusion
    """
    if llm is None:
        raise RuntimeError(
            "summarize_chrono_short nécessite un callable "
            "`llm(system, user, **kwargs)`"
        )

    base_kwargs = dict(llm_kwargs or {})

    def emit(stage: str, payload: Dict | None = None):
        """
        Petit wrapper local pour la progression :
        - si progress_cb est fourni (routes_chrono ou summarize_chrono_short),
          on l'appelle avec (stage, payload).
        - sinon, on ne fait rien.
        """
        if progress_cb:
            try:
                progress_cb(stage, payload or {})
            except Exception:
                # On ne casse pas la synthèse si le callback plante
                pass

    # 1) Chronologie détaillée (sans synthèse mensuelle)
    doc_kwargs = {k: v for k, v in base_kwargs.items() if k in ("model", "case_id")}

    # Plafond éventuel spécifique synthèse courte
    effective_max_docs = max_docs
    if effective_max_docs is None and CHRONO_SHORT_DEFAULT_MAX_DOCS > 0:
        effective_max_docs = CHRONO_SHORT_DEFAULT_MAX_DOCS

    emit("start", {"message": "Construction de la chronologie détaillée..."})
    timeline_md = summarize_chrono_long(
        case_id=case_id,
        case_title=case_title,
        date_from=date_from,
        date_to=date_to,
        include_monthly=False,  # pas besoin de synthèses mensuelles pour le rapport court
        max_docs=effective_max_docs,
        excerpt_chars=excerpt_chars,
        llm=llm,
        llm_kwargs=doc_kwargs,
        progress_cb=emit,
    )

    # 2) Découpage en segments raisonnables
    segments = _split_timeline_md(timeline_md, max_chars=CHRONO_TIMELINE_SEGMENT_CHARS)

    # Cas simple : un seul segment → un seul appel
    if len(segments) == 1:
        emit("short_report", {"message": "Synthèse courte (1 segment) en cours..."})
        report = build_short_report_from_timeline(
            case_title=case_title or f"Affaire {case_id}",
            timeline_md=segments[0],
            llm=None,  # ⚠️ on force le wrapper local (Ollama direct)
            llm_kwargs=base_kwargs,
            case_id=case_id,
        )
        emit("done", {"message": "Synthèse courte terminée"})
        return report

    # 3) Multi-pass : mini-rapports par segment
    mini_reports: List[str] = []
    total = len(segments)
    for i, seg in enumerate(segments, start=1):
        emit(
            "short_report_chunk",
            {
                "message": f"Synthèse partielle {i}/{total}",
                "index": i,
                "total": total,
            },
        )
        # Pour les mini-rapports, budget plus modeste
        kw_chunk = dict(base_kwargs)
        kw_chunk.setdefault("max_tokens", CHRONO_SHORT_MINI_MAX_TOKENS)
        kw_chunk.setdefault("temperature", 0.2)

        user = build_short_report_user_prompt(
            case_title=case_title or f"Affaire {case_id}",
            timeline_md=seg,
        )
        part = llm(CHRONO_SHORT_REPORT_SYSTEM, user, **kw_chunk)
        mini_reports.append(part.strip())

    # 4) Fusion des mini-rapports en rapport final
    emit("short_report", {"message": "Fusion des synthèses partielles..."})
    merged_timeline = "\n\n---\n\n".join(mini_reports)

    final_kwargs = dict(base_kwargs)
    # Rapport final plus riche (~5 pages)
    final_kwargs.setdefault("max_tokens", CHRONO_SHORT_FINAL_MAX_TOKENS)
    final_kwargs.setdefault("temperature", 0.2)

    final_report = build_short_report_from_timeline(
        case_title=case_title or f"Affaire {case_id}",
        timeline_md=merged_timeline,
        llm=None,  # ⚠️ idem : wrapper local
        llm_kwargs=final_kwargs,
        case_id=case_id,
    )
    emit("done", {"message": "Synthèse courte terminée"})

    return final_report
