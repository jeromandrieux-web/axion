# backend/context_utils.py
# Gestion du contexte RAG, estimation de tokens, et SSE.
# Auteur: Jax

import json
import re
import os
import logging
from typing import List, Tuple, Dict, Any, Optional

from config import OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT

logger = logging.getLogger(__name__)

# précompiler regex réutilisées
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Estimation & tronquage de tokens
# ---------------------------------------------------------------------------

def approx_tokens(s: str) -> int:
    """
    Estimation grossière du nombre de tokens.
    On essaye tiktoken si dispo, sinon fallback chars/4.
    """
    if not s:
        return 0
    if not isinstance(s, str):
        s = str(s)

    # essai tiktoken
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    except Exception:
        # fallback simple
        return max(1, len(s) // 4)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Tronque un texte pour respecter un budget max_tokens (approx)."""
    if max_tokens <= 0:
        return ""
    if not text:
        return text

    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        toks = enc.encode(text)
        if len(toks) <= max_tokens:
            return text
        toks = toks[:max_tokens]
        return enc.decode(toks)
    except Exception:
        approx_chars = max_tokens * 4
        if len(text) <= approx_chars:
            return text
        return text[:approx_chars]


# ---------------------------------------------------------------------------
# SSE helper (utilisé par chat_service & routes_chat_upload)
# ---------------------------------------------------------------------------

def sse(data: Dict[str, Any]) -> str:
    """
    Convertit un dict en ligne Server-Sent Events.
    """
    return "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"


# ---------------------------------------------------------------------------
# Construction du contexte RAG
# ---------------------------------------------------------------------------

# On importe retrieval pour profiter de group_results_by_unit si disponible.
try:
    import retrieval as RET  # exécution depuis backend/
except Exception:  # pragma: no cover
    from backend import retrieval as RET  # type: ignore


def _format_unit_header(unit: Dict[str, Any]) -> str:
    """
    Formate un en-tête lisible pour un PV / rapport / unité logique.
    """
    title = unit.get("unit_title") or "Document"
    source = unit.get("source") or "?"
    date = unit.get("date")
    doc_type = unit.get("doc_type")

    meta_bits = []
    if doc_type:
        meta_bits.append(doc_type)
    if date:
        meta_bits.append(date)
    meta_bits.append(source)

    return f"### {title}\n({', '.join(meta_bits)})\n"


def _build_context_from_grouped_units(
    grouped_units: List[Dict[str, Any]],
    max_tokens: int,
) -> str:
    """
    Construit un contexte linéaire à partir d'unités groupées (PV / rapports).
    """
    if not grouped_units:
        return ""

    parts: List[str] = []
    used = 0

    for unit in grouped_units:
        body_chunks: List[str] = []
        for ch in unit.get("chunks", []):
            txt = ch.get("text") or ""
            if not txt.strip():
                continue
            body_chunks.append(txt.strip())

        if not body_chunks:
            continue

        body = "\n".join(body_chunks).strip()
        header = _format_unit_header(unit)
        unit_text = f"{header}\n{body}\n"

        need = approx_tokens(unit_text)
        if used + need <= max_tokens:
            parts.append(unit_text)
            used += need
            continue

        # plus assez de budget → on essaie de prendre une partie
        remaining = max_tokens - used
        if remaining <= 0:
            break
        truncated = _truncate_to_tokens(unit_text, remaining)
        if truncated.strip():
            parts.append(truncated)
            used = max_tokens
        break

    return "\n".join(parts).strip()


def _build_context_flat(
    results: List[Tuple[str, Dict[str, Any]]],
    max_tokens: int,
) -> str:
    """
    Mode "legacy" sans regroupement par PV : concatène simplement les chunks.
    """
    parts: List[str] = []
    used = 0

    for text, meta in results:
        if not text or not text.strip():
            continue

        src = meta.get("source")
        page = meta.get("page")
        header_bits = []
        if src:
            header_bits.append(f"source: {src}")
        if page:
            header_bits.append(f"page: {page}")

        if header_bits:
            chunk = f"[{', '.join(header_bits)}]\n{text.strip()}\n"
        else:
            chunk = text.strip() + "\n"

        need = approx_tokens(chunk)
        if used + need <= max_tokens:
            parts.append(chunk)
            used += need
        else:
            remaining = max_tokens - used
            if remaining <= 0:
                break
            truncated = _truncate_to_tokens(chunk, remaining)
            if truncated.strip():
                parts.append(truncated)
                used = max_tokens
            break

    return "\n".join(parts).strip()


def _make_sources_from_grouped(grouped_units: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Construit la structure "sources" envoyée au front à partir des unités groupées.
    On agrège pages et métadonnées par (source, unit_index).
    """
    out: List[Dict[str, Any]] = []
    for unit in grouped_units:
        pages = []
        for ch in unit.get("chunks", []):
            meta = ch.get("meta") or {}
            p = meta.get("page")
            if isinstance(p, int):
                pages.append(p)
        uniq_pages = sorted(sorted(set(pages))) if pages else []

        out.append({
            "source": unit.get("source"),
            "unit_index": unit.get("unit_index"),
            "unit_title": unit.get("unit_title"),
            "doc_type": unit.get("doc_type"),
            "date": unit.get("date"),
            "pages": uniq_pages,
        })
    return out


def _make_sources_from_flat(results: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Fallback si on ne peut pas grouper par unités : agrège par (source, page).
    """
    by_key: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for _, meta in results:
        src = meta.get("source")
        page = meta.get("page")
        key = (src, page)
        if key not in by_key:
            by_key[key] = {
                "source": src,
                "pages": set(),
                "unit_index": meta.get("unit_index"),
                "unit_title": meta.get("unit_title"),
                "doc_type": meta.get("doc_type"),
                "date": meta.get("date"),
            }
        if isinstance(page, int):
            by_key[key]["pages"].add(page)

    out: List[Dict[str, Any]] = []
    for v in by_key.values():
        pages = sorted(v.pop("pages"))
        v["pages"] = pages
        out.append(v)
    return out


def build_bounded_context(
    results: List[Tuple[str, Dict[str, Any]]],
    user_query: str,
    model_name: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Fonction API utilisée par:
      - ChatService._prepare_context
      - routes_chat_upload

    Signature historique:
        ctx, sources = build_bounded_context(results, message, model)

    - `results` : liste de (texte, meta) retournée par retrieve_hybrid
    - `user_query` : texte de la question de l'utilisateur (non utilisé pour
      le contexte lui-même, mais gardé pour évolutions futures).
    - `model_name` : nom du modèle (pourrait servir à adapter le budget).

    Retourne:
        - ctx : gros texte pour le prompt RAG
        - sources : liste de dicts pour affichage des références côté front
    """
    if not results:
        return "", []

    # Budget de contexte :
    # on laisse de la place pour le prompt + la réponse.
    try:
        max_ctx_total = int(os.getenv("CTX_MAX_TOKENS", str(OLLAMA_NUM_CTX)))
    except Exception:
        max_ctx_total = OLLAMA_NUM_CTX

    # place réservée pour la génération
    reserve_for_answer = OLLAMA_NUM_PREDICT
    # petite marge pour instructions / système
    reserve_for_prompt = 512

    ctx_budget = max(512, max_ctx_total - reserve_for_answer - reserve_for_prompt)

    # 1) Regroupement par unité logique si possible
    grouped_units: List[Dict[str, Any]] = []
    can_group = hasattr(RET, "group_results_by_unit")

    if can_group:
        try:
            grouped_units = RET.group_results_by_unit(results)  # type: ignore
        except Exception:
            logger.exception("group_results_by_unit a échoué, fallback en mode plat.")
            grouped_units = []

    if grouped_units:
        ctx = _build_context_from_grouped_units(grouped_units, max_tokens=ctx_budget)
        sources = _make_sources_from_grouped(grouped_units)
        return ctx, sources

    # 2) Fallback : pas de regroupement par unités → mode plat
    ctx = _build_context_flat(results, max_tokens=ctx_budget)
    sources = _make_sources_from_flat(results)
    return ctx, sources


# ---------------------------------------------------------------------------
# Normalisation pour BM25 (conservée pour compatibilité)
# ---------------------------------------------------------------------------

def norm_bm25(text: str) -> List[str]:
    """Normalisation du texte pour BM25 (retourne tokens)."""
    import unicodedata
    s = unicodedata.normalize("NFKC", text or "")
    s = s.replace("\u00A0", " ").lower()
    s = ''.join(
        c
        for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
    s = _WS_RE.sub(' ', s).strip()
    return s.split()
