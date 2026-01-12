# backend/routes_cases.py
# Routes d'affaire : création, liste des documents, recherche
# Auteur: Jax

import logging
import re
import os
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Form, Query
from fastapi.responses import PlainTextResponse

import ingest as ING
import retrieval as RET  # nécessaire pour /api/case/search

router = APIRouter()
logger = logging.getLogger("axion.routes_cases")

# ---------------------------------------------------------------------------
# 🔐 VALIDATION case_id UNIFIÉE
# ---------------------------------------------------------------------------

_CASE_ID_RE = re.compile(r'^[A-Za-z0-9_\-\. ]{1,128}$')

def _sanitize_case_id(case_id: str) -> str:
    if case_id is None:
        raise HTTPException(status_code=400, detail="case_id manquant")

    cid = (case_id or "").strip()
    if not cid or not _CASE_ID_RE.fullmatch(cid):
        raise HTTPException(
            status_code=400,
            detail=(
                "case_id invalide — caractères permis : lettres, chiffres, espace, "
                "underscore (_), tiret (-), point (.). "
                "Interdits : '/', retours à la ligne. Longueur 1..128."
            ),
        )
    return cid


# ---------------------------------------------------------------------------
# 📁 LISTE DES AFFAIRES
# ---------------------------------------------------------------------------

@router.get("/api/cases")
def api_cases():
    try:
        return {"cases": ING.list_cases()}
    except Exception as e:
        logger.exception("list_cases failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 🆕 CRÉATION D’UNE AFFAIRE
# ---------------------------------------------------------------------------

@router.post("/api/case/create")
def api_case_create(case_id: str = Form(...)):
    case_id = _sanitize_case_id(case_id)
    try:
        ING.ensure_case(case_id)
        return {"ok": True, "case_id": case_id}
    except Exception as e:
        logger.exception("ensure_case failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 📄 LISTE DES DOCUMENTS INGÉRÉS (bruts)
# ---------------------------------------------------------------------------

@router.get("/api/case/docs")
def api_case_docs(case_id: str):
    case_id = _sanitize_case_id(case_id)
    try:
        return {"case_id": case_id, "docs": ING.list_docs(case_id)}
    except Exception as e:
        logger.exception("list_docs failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 📑 LISTE DES UNITÉS LOGIQUES (PV / RAPPORTS / DOCUMENTS)
# ---------------------------------------------------------------------------

@router.get("/api/case/docs/logical")
def api_case_logical_docs(case_id: str):
    """
    Retourne toutes les unités logiques (PV, rapports, actes…) détectées
    pour chaque document ingéré dans l'affaire.
    """
    case_id = _sanitize_case_id(case_id)

    try:
        meta = ING._load_case_meta(case_id)
        docs = meta.get("docs", [])

        logical = []

        for doc in docs:
            fname = doc.get("name")
            for u in doc.get("units", []):
                logical.append({
                    "file": fname,
                    "unit_index": u.get("index"),
                    "unit_title": u.get("title"),
                    "page_start": u.get("page_start"),
                    "page_end": u.get("page_end"),
                })

        # tri naturel : fichier → page_start
        logical.sort(key=lambda x: (x["file"], x["page_start"] or 0))

        return {
            "case_id": case_id,
            "logical_docs": logical
        }

    except Exception as e:
        logger.exception("logical_docs failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# 🔍 RECHERCHE HYBRIDE + GROUPÉE PAR UNITÉ LOGIQUE
# ---------------------------------------------------------------------------

@router.get("/api/case/search")
def api_case_search(
    case_id: str,
    q: str = Query(..., alias="query"),
    top_k: int = Query(12),
    group: bool = Query(False)
):
    """
    Lance une recherche hybride (BM25 + embeddings).
    Si group=true → regroupe les résultats par PV/rapport.
    """
    case_id = _sanitize_case_id(case_id)

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="requête vide")

    try:
        raw = RET.retrieve_hybrid(case_id, q, top_k=top_k)

        # mode brut classique
        if not group:
            return {
                "case_id": case_id,
                "query": q,
                "results": [
                    {"text": t, "meta": m}
                    for (t, m) in raw
                ]
            }

        # mode groupé : PV / unité logique
        grouped = RET.group_results_by_unit(raw)

        return {
            "case_id": case_id,
            "query": q,
            "grouped_results": grouped
        }

    except Exception as e:
        logger.exception("case search failed")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Helper : Fallback sur bm25.json quand l'unité logique ne matche pas
# ---------------------------------------------------------------------------

def _fallback_unit_from_bm25(
    case_id: str,
    source: str,
    unit_index: int,
) -> str:
    """
    Fallback utilisé quand l'index d'unité logique (split_pages_into_units)
    ne correspond à rien.

    On va alors chercher dans bm25.json :
      - tous les chunks dont meta['source'] == source,
      - d'abord un chunk dont meta['local_chunk_id'] == unit_index,
      - sinon, on interprète unit_index comme position dans la liste
        de ces chunks.

    ⚠️ Ne lève PAS d'HTTPException : renvoie toujours un texte (même si c'est
    juste un message explicatif). Ainsi, l'UI n'a jamais de 404 pour ce lien.
    """
    try:
        casedir = ING.STORAGE / "cases" / case_id
        bm25_path = casedir / "bm25.json"

        if not bm25_path.exists():
            return (
                f"[Extrait introuvable]\n"
                f"Aucun index bm25.json trouvé pour l'affaire {case_id}.\n"
            )

        try:
            data = json.loads(bm25_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.exception("fallback_unit_from_bm25: lecture bm25.json échouée")
            return (
                f"[Extrait introuvable]\n"
                f"bm25.json illisible pour l'affaire {case_id} : {e}\n"
            )

        chunks = data.get("chunks") or []
        sources = data.get("sources") or []

        if not chunks or not sources:
            return (
                f"[Extrait introuvable]\n"
                f"Aucun contenu indexé dans bm25.json pour l'affaire {case_id}.\n"
            )

        # indices des chunks correspondant à ce 'source'
        indices_for_source = [
            i for i, meta in enumerate(sources)
            if (meta or {}).get("source") == source
        ]

        if not indices_for_source:
            return (
                f"[Extrait introuvable]\n"
                f"Aucun extrait trouvé dans bm25.json pour le fichier : {source}.\n"
            )

        # tentons d'abord via local_chunk_id
        by_local_id = {}
        for i in indices_for_source:
            meta = sources[i] or {}
            lcid = meta.get("local_chunk_id")
            if isinstance(lcid, int):
                by_local_id[lcid] = i

        chosen_idx = None

        if by_local_id and unit_index in by_local_id:
            chosen_idx = by_local_id[unit_index]
        else:
            # fallback pour unit_index comme position dans indices_for_source
            if 0 <= unit_index < len(indices_for_source):
                chosen_idx = indices_for_source[unit_index]

        if chosen_idx is None or not (0 <= chosen_idx < len(chunks)):
            return (
                f"[Extrait introuvable]\n"
                f"Aucun chunk correspondant à unit_index={unit_index} pour {source}.\n"
            )

        txt = chunks[chosen_idx]
        meta = sources[chosen_idx] or {}
        page = meta.get("page")

        header = f"{source} — extrait {unit_index}\n"
        if page is not None:
            header += f"(page {page})\n"
        header += "\n"

        body = txt.strip() or "(extrait vide)"

        return header + body + "\n"

    except Exception as e:
        logger.exception("fallback_unit_from_bm25: erreur inattendue")
        return (
            f"[Extrait introuvable]\n"
            f"Erreur technique lors du fallback bm25 : {e}\n"
        )


# ---------------------------------------------------------------------------
# 📜 RÉCUPÉRATION D’UNE UNITÉ LOGIQUE (PV / RAPPORT) PAR LIEN SOURCE
# ---------------------------------------------------------------------------

@router.get("/api/case/unit", response_class=PlainTextResponse)
def api_case_unit(
    case_id: str,
    source: str = Query(..., description="Nom du fichier brut (ex: dossier.pdf)"),
    unit_index: int = Query(..., description="Index de l'unité logique (PV/rapport)"),
):
    """
    Retourne le texte reconstitué d'une unité logique (PV / rapport) à partir
    du fichier brut (souvent un PDF) stocké dans l'affaire, en réutilisant
    la même logique de découpe que lors de l'ingestion (split_pages_into_units).

    Utilisé par les liens des 'pills' de sources dans l'UI.

    Si l'index d'unité logique ne correspond à rien (par ex. lien basé sur local_chunk_id
    ou découpage OCR différent), on tente un fallback en lisant directement bm25.json.

    ⚠️ Cette route NE renvoie plus de 404 pour les problèmes d'index : au pire,
    elle renvoie un texte explicatif. Ainsi les liens ne sont jamais "morts".
    """
    case_id = _sanitize_case_id(case_id)

    # 1) On tente d'abord la reconstruction propre via le PDF
    try:
        casedir = ING.STORAGE / "cases" / case_id
        raw_path = casedir / "raw" / source

        if not raw_path.exists():
            logger.warning(
                "api_case_unit: fichier brut introuvable (%s) pour l'affaire %s, fallback bm25",
                source, case_id
            )
            return _fallback_unit_from_bm25(case_id, source, unit_index)

        suffix = raw_path.suffix.lower()
        pages = []

        if suffix == ".pdf":
            try:
                import fitz  # PyMuPDF
            except ImportError:
                logger.error("api_case_unit: PyMuPDF non installé, fallback bm25")
                return _fallback_unit_from_bm25(case_id, source, unit_index)

            doc = fitz.open(str(raw_path))
            try:
                for i, page in enumerate(doc):
                    text = page.get_text("text") or ""
                    pages.append((i + 1, text))
            finally:
                doc.close()
        else:
            # fallback simple pour des .txt ou autres
            text = raw_path.read_text(encoding="utf-8", errors="ignore")
            pages = [(1, text)]

        if not pages:
            logger.warning(
                "api_case_unit: aucune page lisible dans %s pour l'affaire %s, fallback bm25",
                source, case_id
            )
            return _fallback_unit_from_bm25(case_id, source, unit_index)

        # re-segmentation en unités logiques (PV / rapports / etc.)
        try:
            try:
                from utils_ingest import (
                    split_pages_into_units,
                    detect_doc_date,
                    detect_doc_type,
                )
            except Exception:
                from backend.utils_ingest import (  # type: ignore
                    split_pages_into_units,
                    detect_doc_date,
                    detect_doc_type,
                )
        except Exception as e:
            logger.exception("api_case_unit: import utils_ingest failed, fallback bm25")
            return _fallback_unit_from_bm25(case_id, source, unit_index)

        units = split_pages_into_units(pages, raw_path.name)

        if not units:
            logger.warning(
                "api_case_unit: aucune unité logique détectée dans %s (affaire %s), fallback bm25",
                source, case_id
            )
            return _fallback_unit_from_bm25(case_id, source, unit_index)

        # index hors bornes → fallback
        if unit_index < 0 or unit_index >= len(units):
            logger.warning(
                "api_case_unit: unit_index %s hors bornes (max=%s) pour %s — fallback bm25",
                unit_index,
                len(units) - 1,
                source,
            )
            return _fallback_unit_from_bm25(case_id, source, unit_index)

        # OK : on renvoie l'unité logique complète
        unit = units[unit_index]
        title = unit.get("title") or f"{source} (unité {unit_index})"
        unit_pages = unit.get("pages", [])

        # métadonnées (date / type) à partir des premières pages de l'unité
        first_text = " \n".join(p[1] for p in unit_pages[:2]) if unit_pages else ""
        doc_date = detect_doc_date(first_text, raw_path.name)
        doc_type = detect_doc_type(first_text, raw_path.name)

        meta_bits = []
        if doc_type:
            meta_bits.append(doc_type)
        if doc_date:
            meta_bits.append(doc_date)
        meta_bits.append(source)

        header = title
        if meta_bits:
            header += "\n(" + ", ".join(meta_bits) + ")\n"
        header += "\n"

        body_parts = []
        for page_num, txt in unit_pages:
            if txt and txt.strip():
                body_parts.append(f"[Page {page_num}]\n{txt.strip()}")

        body = "\n\n".join(body_parts).strip()

        if not body:
            body = "(Unité logique vide ou texte illisible)"

        return header + body + "\n"

    except Exception as e:
        # En cas de pépin inattendu, on log et on tente encore le fallback bm25
        logger.exception("api_case_unit: erreur inattendue, fallback bm25")
        fb = _fallback_unit_from_bm25(case_id, source, unit_index)
        # On ajoute un petit commentaire technique à la fin
        return fb + f"\n\n[Note technique] Erreur lors de la reconstitution PDF : {e}\n"
