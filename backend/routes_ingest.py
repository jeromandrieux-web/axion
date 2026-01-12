# backend/routes_ingest.py
# Endpoint d’ingestion des fichiers (streaming SSE)
# Auteur: Jax

import logging
import asyncio
import tempfile
import os
import re
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse

import ingest as ING

# limites configurables
MAX_INGEST_FILES = int(os.getenv("INGEST_MAX_FILES", "50"))
MAX_INGEST_FILE_BYTES = int(
    os.getenv("INGEST_MAX_FILE_BYTES", str(200 * 1024 * 1024))
)  # 200MB

# 🔐 même règle que dans routes_cases.py :
# - Autorise : lettres, chiffres, espace, underscore, tiret, point
# - Refuse : '/', retours à la ligne
# - Longueur 1..128
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


router = APIRouter()
logger = logging.getLogger("axion.routes_ingest")


def _sse(event: dict) -> str:
    import json
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/api/ingest/stream")
async def api_ingest_stream(
    case_id: str = Form(...),
    files: Optional[List[UploadFile]] = None,
):
    """
    Endpoint d’ingestion des fichiers d’une affaire :
    - contrôle du nombre et de la taille des fichiers
    - enregistre les fichiers dans un répertoire temporaire
    - lance ING.ingest_files(...) dans un thread
    - streame la progression via SSE
    """
    files = files or []
    case_id = _sanitize_case_id(case_id)

    if len(files) > MAX_INGEST_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"trop de fichiers (max {MAX_INGEST_FILES})",
        )

    # Lire les UploadFile pendant le scope du handler
    tmp = tempfile.TemporaryDirectory(prefix="axion_ing_")
    tmpdir = Path(tmp.name)
    saved: List[Path] = []

    try:
        ING.ensure_case(case_id)

        for idx, uf in enumerate(files):
            fn = (uf.filename or f"upload_{idx}").replace(os.path.sep, "_")
            p = tmpdir / fn
            written = 0

            try:
                with open(p, "wb") as f:
                    while True:
                        chunk = await uf.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        written += len(chunk)
                        if written > MAX_INGEST_FILE_BYTES:
                            raise HTTPException(
                                status_code=413,
                                detail=f"Fichier {fn} trop volumineux",
                            )
            finally:
                try:
                    await uf.close()
                except Exception:
                    pass

            saved.append(p)

    except HTTPException:
        # erreur contrôlée → cleanup puis propagation
        try:
            tmp.cleanup()
        except Exception:
            logger.exception("failed to cleanup tmpdir on error")
        raise
    except Exception:
        logger.exception("unexpected ingest error while reading uploads")
        try:
            tmp.cleanup()
        except Exception:
            logger.exception("failed to cleanup tmpdir on error")
        raise HTTPException(
            status_code=500,
            detail="Erreur ingestion: lecture des fichiers",
        )

    async def event_gen():
        loop = asyncio.get_running_loop()
        q: "asyncio.Queue[dict]" = asyncio.Queue()

        def progress_cb(ev: dict):
            try:
                loop.call_soon_threadsafe(q.put_nowait, ev)
            except Exception:
                logger.exception(
                    "progress_cb failed to enqueue event: %s", ev
                )

        # lancer ingestion dans un thread
        ingest_task = asyncio.create_task(
            asyncio.to_thread(ING.ingest_files, case_id, saved, progress_cb)
        )

        try:
            # informer le client que le stream démarre
            yield _sse({"type": "start"})

            # boucle d'écoute d'événements
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    # si la tâche est terminée et la queue vide, on sort
                    if ingest_task.done() and q.empty():
                        break
                    continue

                yield _sse(ev)
                if ev.get("type") in ("done", "end", "error"):
                    break

            # on attend la fin de la tâche pour capturer une éventuelle exception
            try:
                await ingest_task
            except Exception as e:
                logger.exception("ingest failed after thread completion")
                yield _sse({"type": "error", "message": str(e)})
                yield _sse({"type": "end"})
            else:
                yield _sse({"type": "end"})
        finally:
            # ⚠️ nettoyage seulement après la fin de la tâche
            try:
                if not ingest_task.done():
                    await ingest_task
            except Exception:
                # on log mais on continue le cleanup
                logger.exception("ingest task raised during final await")
            try:
                tmp.cleanup()
            except Exception:
                logger.exception("failed to cleanup tmpdir")

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
