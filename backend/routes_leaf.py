# backend/routes_leaf.py
# Routes de synthèse LEAF (streaming SSE)
# Auteur: Jax

from __future__ import annotations

import json
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse

from config import STORAGE as STORAGE_DIR                 # même logique que routes_summarize_long
from context_utils import sse                             # helper SSE
from summarizer import summarize_leaf_long                # moteur LEAF
from power_utils import prevent_sleep                     # empêche la mise en veille
from identity_mapper import deanonymize_text_for_case

logger = logging.getLogger("axion.routes_leaf")
router = APIRouter()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


def _leaf_outfile(case_id: str) -> Path:
    """
    Renvoie le chemin du fichier de sortie pour la synthèse LEAF.
    """
    casedir = STORAGE_DIR / "cases" / case_id
    outdir = casedir / "summaries"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / "synthese_leaf.md"


@router.get("/api/leaf/{case_id}/download")
def api_leaf_download(case_id: str):
    """
    Téléchargement direct de la synthèse LEAF.
    Compatible avec l’ancien front et le nouveau.
    """
    case_id = (case_id or "").strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id manquant")

    outfile = _leaf_outfile(case_id)
    if not outfile.exists():
        raise HTTPException(status_code=404, detail="Aucune synthèse LEAF trouvée")

    return FileResponse(
        path=str(outfile),
        filename=f"synthese_leaf_{case_id}.md",
        media_type="text/markdown",
    )


@router.post("/api/leaf/stream")
async def api_leaf_stream(
    request: Request,
    case_id: str = Form(...),
    model: Optional[str] = Form(None),
):
    """
    Synthèse LEAF (dossier très détaillé) en streaming SSE.

    - Lit bm25.json dans STORAGE/cases/<case_id>/
    - Construit les chunks pour le moteur LEAF
    - Appelle summarize_leaf_long(case_id, chunks, model, progress_cb)
    - Écrit le résultat dans STORAGE/cases/<case_id>/summaries/synthese_leaf.md
    - Renvoie via SSE des événements de progression, puis un 'end' avec le path
      du fichier sous /storage/... pour téléchargement côté front.
    """
    case_id = (case_id or "").strip()
    if not case_id:
        raise HTTPException(status_code=400, detail="case_id manquant")

    casedir = STORAGE_DIR / "cases" / case_id
    bm25 = casedir / "bm25.json"
    if not bm25.exists():
        raise HTTPException(
            status_code=404,
            detail="Affaire introuvable ou index (bm25.json) manquant",
        )

    # --- Lecture des chunks (même logique que routes_summarize_long.py) ---
    raw = json.loads(bm25.read_text(encoding="utf-8"))
    chunks: List[Dict[str, Any]]

    if isinstance(raw, dict) and "chunks" in raw:
        # format { "chunks": [...], "sources": [...] }
        chunks = []
        texts = raw.get("chunks", [])
        srcs = raw.get("sources", [])
        for i, txt in enumerate(texts):
            meta = srcs[i] if i < len(srcs) else {}
            chunks.append({"text": txt, "meta": meta})
    elif isinstance(raw, list):
        # format déjà [{text, meta}] ou similaire
        chunks = raw
    else:
        raise HTTPException(status_code=500, detail="Format inattendu de bm25.json")

    q: asyncio.Queue = asyncio.Queue()

    def progress_cb(evt: Dict[str, Any]):
        """
        Callback appelé par summarize_leaf_long pour signaler la progression.
        On le renvoie vers la queue async consommée par le SSE.

        On ignore volontairement les evt.type == "end" produits par le moteur,
        pour ne renvoyer qu'un seul 'end' final enrichi avec le path.
        """
        try:
            if not isinstance(evt, dict):
                return
            if evt.get("type") == "end":
                # on laisse le 'end' final géré par le worker (avec path)
                return

            # --- Dé-pseudonymisation si l'événement contient du texte (message/content) ---
            try:
                if isinstance(evt.get("message"), str):
                    de_msg = deanonymize_text_for_case(evt["message"], case_id)
                    if de_msg == evt["message"] and case_id:
                        de_msg = deanonymize_text_for_case(evt["message"], None)
                    evt["message"] = de_msg

                if isinstance(evt.get("content"), str):
                    de_content = deanonymize_text_for_case(evt["content"], case_id)
                    if de_content == evt["content"] and case_id:
                        de_content = deanonymize_text_for_case(evt["content"], None)
                    evt["content"] = de_content
            except Exception:
                logger.exception("routes_leaf: deanonymize_text_for_case failed for progress event")

            # On passe par l'event loop pour pousser dans la queue depuis le thread du moteur
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(q.put_nowait, evt)
            except RuntimeError:
                # en dernier recours, on tente un put direct (rare)
                try:
                    q.put_nowait(evt)
                except Exception:
                    logger.exception("routes_leaf: progress_cb failed (put_nowait)")
        except Exception:  # on log mais on ne casse pas la synthèse
            logger.exception("routes_leaf: erreur dans progress_cb")

    async def worker():
        try:
            # Empêche la mise en veille pendant la synthèse LEAF
            with prevent_sleep(reason=f"Synthèse LEAF {case_id}", logger=logger):
                # Appel du moteur LEAF dans un thread pour ne pas bloquer l'event loop
                result = await asyncio.to_thread(
                    summarize_leaf_long,
                    case_id,
                    chunks,
                    model,
                    progress_cb,
                )

                # Enregistrement du fichier de synthèse
                outfile = _leaf_outfile(case_id)

                # --- Dé-pseudonymisation du résultat final avant écriture ---
                try:
                    de_result = deanonymize_text_for_case(result or "", case_id)
                    if de_result == (result or "") and case_id:
                        de_result = deanonymize_text_for_case(result or "", None)
                    result = de_result
                except Exception:
                    logger.exception("routes_leaf: deanonymize_text_for_case failed for final result")

                outfile.write_text(result or "", encoding="utf-8")

                rel = f"/storage/cases/{case_id}/summaries/{outfile.name}"
                await q.put({"type": "progress", "value": 1.0, "stage": "done"})
                await q.put({"type": "end", "case_id": case_id, "path": rel})

        except Exception as e:
            logger.exception("routes_leaf: worker failed")
            await q.put({"type": "error", "message": str(e)})

    # Lancer le worker en tâche de fond
    asyncio.create_task(worker())

    async def evt_gen():
        while True:
            evt = await q.get()
            # On renvoie les événements au format SSE
            yield sse(evt)
            if evt.get("type") in ("end", "error"):
                break

    return StreamingResponse(
        evt_gen(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
