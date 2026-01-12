# backend/routes_phones.py
# Recherche et "fiche téléphone" pour une affaire
# Auteur: Jax

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Form
from fastapi.responses import JSONResponse

import ingest as ING
from utils_normalize import PHONE_RE, normalize_phone, extract_with_regex

# LLM wrapper (même pattern que routes_chrono)
try:
    from backend.llm_client import llm_call  # type: ignore
except Exception:
    from llm_client import llm_call  # type: ignore

# Dé-pseudonymisation de sûreté (au cas où des tags PERS_XXX / TEL_XXX auraient fuité)
try:
    from identity_mapper import deanonymize_text_for_case
except Exception:
    def deanonymize_text_for_case(text: str, case_id: Optional[str] = None) -> str:  # type: ignore
        return text

router = APIRouter()

STORAGE: Path = ING.STORAGE  # même storage que le reste de l'app
logger = logging.getLogger("axion.routes_phones")


# -------------------------------------------------------------
# Core : scan des numéros dans bm25.json
# -------------------------------------------------------------

def _scan_phone_hits(case_id: str, normalized_number: str) -> List[Dict[str, Any]]:
    """
    Parcourt bm25.json de l'affaire et renvoie toutes les occurrences
    du numéro normalisé (E.164, ex: +33612345678).

    Retourne une liste de dicts {source, page, chunk_id, raw, snippet}.
    """
    casedir = STORAGE / "cases" / case_id
    bm25_path = casedir / "bm25.json"
    if not bm25_path.exists():
        raise HTTPException(status_code=404, detail="Affaire introuvable ou index bm25 absent")

    data = json.loads(bm25_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks") or []
    sources = data.get("sources") or []

    hits: List[Dict[str, Any]] = []

    for i, text in enumerate(chunks):
        meta = sources[i] if i < len(sources) else {}
        matches = extract_with_regex(
            text=text,
            regex=PHONE_RE,
            person=None,
            normalizer=normalize_phone,
            default_cc="+33",
        )
        for m in matches:
            if m.get("value") == normalized_number:
                span = m.get("span") or (0, 0)
                i0, i1 = span
                i0 = max(0, i0 - 120)
                i1 = min(len(text), i1 + 120)
                snippet = text[i0:i1].strip()

                hits.append(
                    {
                        "source": meta.get("source"),
                        "page": meta.get("page"),
                        "chunk_id": meta.get("chunk_id") or meta.get("local_chunk_id") or i,
                        "raw": m.get("raw"),
                        "snippet": snippet,
                    }
                )

    return hits


# -------------------------------------------------------------
# 1) API brute : lookup JSON
# -------------------------------------------------------------

@router.get("/api/case/phone_lookup")
def api_case_phone_lookup(
    case_id: str = Query(...),
    number: str = Query(..., description="Numéro brut: 06..., +33..., 0033..., etc."),
):
    """
    Recherche d'un numéro de téléphone dans toute l'affaire, retour JSON brut.
    """
    case_id = (case_id or "").strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id manquant")

    if not number or not number.strip():
        raise HTTPException(status_code=400, detail="Numéro manquant")

    normalized = normalize_phone(number)
    if not normalized:
        raise HTTPException(status_code=400, detail="Numéro invalide ou non reconnu")

    hits = _scan_phone_hits(case_id, normalized)

    return JSONResponse(
        {
            "case_id": case_id,
            "input_number": number,
            "normalized_number": normalized,
            "hits": hits,
        }
    )


# -------------------------------------------------------------
# 2) Fiche téléphone Markdown + fichier dans /summaries
# -------------------------------------------------------------

def _build_phone_report_markdown(
    case_id: str,
    normalized_number: str,
    hits: List[Dict[str, Any]],
) -> str:
    """
    Construit un Markdown structuré pour la fiche téléphone.
    (sans LLM pour l'instant, mais on lui prépare un beau contexte)
    """
    title = f"Fiche téléphone — {normalized_number}"
    lines: List[str] = []
    lines.append(f"# {title}\n")
    lines.append(f"- Affaire : `{case_id}`")
    lines.append(f"- Numéro normalisé : `{normalized_number}`")
    lines.append(f"- Nombre d'occurrences détectées : **{len(hits)}**\n")

    if not hits:
        lines.append("> Aucun contexte textuel n'a été trouvé pour ce numéro dans l'affaire.")
        return "\n".join(lines) + "\n"

    # Regroupement par document
    by_source: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for h in hits:
        src = h.get("source") or "(source inconnue)"
        by_source.setdefault(src, []).append(h)

    lines.append("## 1. Vue d’ensemble\n")
    for src, items in sorted(by_source.items(), key=lambda kv: kv[0]):
        pages = sorted({it.get("page") for it in items if it.get("page") is not None})
        pages_str = ", ".join(str(p) for p in pages) if pages else "nc."
        lines.append(f"- **{src}** — pages {pages_str} — {len(items)} occurrence(s)")

    lines.append("\n## 2. Occurrences détaillées par document\n")

    for src, items in sorted(by_source.items(), key=lambda kv: kv[0]):
        lines.append(f"### {src}\n")
        for idx, it in enumerate(items, start=1):
            page = it.get("page")
            raw = it.get("raw")
            snippet = (it.get("snippet") or "").replace("\n", " ").strip()
            page_lbl = f"page {page}" if page is not None else "page inconnue"
            lines.append(f"**Occurrence {idx} ({page_lbl})**")
            if raw:
                lines.append(f"- Forme dans le texte : `{raw}`")
            lines.append(f"- Contexte : {snippet}\n")

    return "\n".join(lines) + "\n"


def _save_phone_report(
    case_id: str,
    normalized_number: str,
    md: str,
) -> str:
    """
    Sauvegarde la fiche dans backend/storage/cases/<case_id>/summaries/fiche_tel_XXXX.md
    et renvoie le chemin web (/storage/...).
    """
    casedir = STORAGE / "cases" / case_id / "summaries"
    casedir.mkdir(parents=True, exist_ok=True)

    safe_num = normalized_number.replace("+", "plus").replace(" ", "").replace("/", "_")
    filename = f"fiche_telephone_{safe_num}.md"
    out_path = casedir / filename
    out_path.write_text(md, encoding="utf-8")

    # mappage comme routes_chrono : /storage/... vers STORAGE root
    rel = out_path.relative_to(STORAGE)
    return f"/storage/{rel}".replace("\\", "/")


@router.post("/api/phones/report")
def api_phone_report(
    case_id: str = Form(...),
    number: str = Form(...),
    model: Optional[str] = Form(None),
):
    """
    Génère une fiche téléphone :
    - normalise le numéro,
    - cherche les occurrences dans l'affaire,
    - produit un Markdown structuré,
    - sauvegarde le fichier dans /summaries,
    - renvoie le chemin de téléchargement.
    """
    case_id = (case_id or "").strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id manquant")

    if not number or not number.strip():
        raise HTTPException(status_code=400, detail="Numéro manquant")

    normalized = normalize_phone(number)
    if not normalized:
        raise HTTPException(status_code=400, detail="Numéro invalide ou non reconnu")

    hits = _scan_phone_hits(case_id, normalized)

    # Markdown "de base"
    md_base = _build_phone_report_markdown(case_id, normalized, hits)

    # (Optionnel) passage par LLM pour un petit résumé introductif / analyse
    intro = ""
    try:
        if hits:
            # contexte succinct pour le LLM
            occurrences_txt = "\n".join(
                f"- {h.get('source')} p.{h.get('page')}: {(h.get('snippet') or '').replace(chr(10), ' ')[:260]}"
                for h in hits[:40]
            )
            system = (
                "Tu es un analyste judiciaire français. On t'indique un numéro de téléphone "
                "et des extraits de PV/rapports où il apparaît. "
                "Tu rédiges une courte synthèse factuelle (10-15 lignes) sur l'utilisation "
                "de ce numéro dans la procédure, sans inventer."
            )
            user = (
                f"Affaire : {case_id}\n"
                f"Numéro : {normalized}\n\n"
                "Extraits où le numéro apparaît :\n"
                f"{occurrences_txt}\n\n"
                "Rédige une synthèse factuelle, en français, adaptée à une fiche téléphone."
            )
            llm_kwargs = {"case_id": case_id}
            if model:
                llm_kwargs["model"] = model

            intro = llm_call(system, user, **llm_kwargs).strip()
    except Exception:
        # en cas de problème LLM, on ne plante pas la fiche
        logger.exception("routes_phones: erreur lors de l'appel LLM pour la synthèse de la fiche téléphone")

    if intro:
        md = f"# Fiche téléphone — {normalized}\n\n## Synthèse\n\n{intro}\n\n---\n\n" + md_base
    else:
        md = md_base

    # 🔐 Sûreté supplémentaire : dé-pseudonymisation de la fiche complète avant sauvegarde / retour
    try:
        de_md = deanonymize_text_for_case(md or "", case_id)
        if de_md == (md or "") and case_id:
            de_md = deanonymize_text_for_case(md or "", None)
        md = de_md
    except Exception:
        logger.exception("routes_phones: deanonymize_text_for_case failed for final phone report")

    path = _save_phone_report(case_id, normalized, md)

    return JSONResponse(
        {
            "case_id": case_id,
            "input_number": number,
            "normalized_number": normalized,
            "hits_count": len(hits),
            "path": path,
            "download_url": path,  # pour faire comme la fiche personne
            "answer": md,          # ⬅️ texte complet de la fiche pour le chat
        }
    )
