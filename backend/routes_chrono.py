# backend/routes_chrono.py
# Route SSE /api/chrono/stream — synthèse chronologique (PV & rapports)
# Auteur: Jax

"""
Route SSE /api/chrono/stream — synthèse chronologique (PV & rapports)

Dépendances :
- summarizer_chrono.summarize_chrono_long
- summarizer_chrono.summarize_chrono_short
- une fonction utilitaire llm_call(system, user, **kwargs) -> str
- write_summary_file(case_id, text, name="synthese_chrono.md")

Stack : FastAPI.
"""
from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

# Imports tolérant le chemin relatif/absolu
try:
    from backend.summarizer_chrono import summarize_chrono_long, summarize_chrono_short
except Exception:
    from summarizer_chrono import summarize_chrono_long, summarize_chrono_short

# Wrapper LLM — import tolérant le chemin
try:
    from backend.llm_client import llm_call  # signature: (system, user, **kwargs) -> str
except Exception:  # pragma: no cover
    from llm_client import llm_call  # type: ignore

# Gestion de la mise en veille (comme pour la synthèse longue)
try:
    from backend.power_utils import prevent_sleep, restore_sleep
except Exception:  # pragma: no cover
    from power_utils import prevent_sleep, restore_sleep  # type: ignore

# Écriture du fichier de sortie
STORAGE_ROOT = Path("backend/storage").resolve()

logger = logging.getLogger("axion.routes_chrono")


def write_summary_file(case_id: str, text: str, name: str = "synthese_chrono.md") -> str:
    case_dir = STORAGE_ROOT / "cases" / case_id / "summaries"
    case_dir.mkdir(parents=True, exist_ok=True)
    out_path = case_dir / name
    out_path.write_text(text, encoding="utf-8")
    # Retourne un chemin relatif que votre UI sait ouvrir
    rel = str(out_path.relative_to(STORAGE_ROOT))
    return f"/storage/{rel}"  # ex : route statique /storage/ qui mappe STORAGE_ROOT


router = APIRouter()


def sse_event(event: str, data: str) -> bytes:
    # format SSE : event: <name>\n data: <json>\n\n
    return f"event: {event}\ndata: {data}\n\n".encode("utf-8")


@router.post("/api/chrono/stream")
async def chrono_stream(request: Request):
    body = await request.json()
    case_id = str(body.get("case_id"))
    date_from = body.get("from_date")
    date_to = body.get("to_date")
    # Pour l’instant on ne remet pas la synthèse mensuelle : chronologie pure doc par doc.
    include_monthly = False
    max_docs = body.get("max_docs")  # peut rester None (tous les docs ou plafond .env)
    case_title = body.get("case_title") or ""
    model = body.get("model")  # optionnel : si tu veux forcer un modèle depuis le front

    # Nouveau : mode = "long" (détaillé) ou "short" (rapport costaud 5 parties)
    mode = (body.get("mode") or "long").lower()
    if mode not in ("long", "short"):
        mode = "long"

    # File d'émission depuis le thread de travail
    queue: asyncio.Queue[bytes] = asyncio.Queue()

    def progress_cb(stage: str, payload: Dict[str, Any]):
        """
        Callback de progression depuis summarize_chrono_long / summarize_chrono_short.
        On empaquette en event SSE 'progress'.
        """
        base = {"stage": stage, "mode": mode}
        if payload:
            base.update(payload)
        data = json.dumps(base, ensure_ascii=False)
        queue.put_nowait(sse_event("progress", data))

    def job():
        sleep_token = None
        try:
            # Empêcher la mise en veille pendant la synthèse chrono
            try:
                sleep_token = prevent_sleep(
                    reason=f"synthèse chrono ({mode}) pour l’affaire {case_id}",
                    logger=logger,
                )
            except Exception:
                logger.exception("routes_chrono: prevent_sleep a échoué")

            # kwargs pour le wrapper LLM
            llm_kwargs = {
                "model": model,
                "case_id": case_id,
            }

            if mode == "short":
                # Rapport court structuré (Les faits, L’enquête, Les suspects, Arrestations/GAV, Conclusion)
                md = summarize_chrono_short(
                    case_id,
                    case_title=case_title,
                    date_from=date_from,
                    date_to=date_to,
                    max_docs=max_docs,
                    llm=llm_call,
                    llm_kwargs=llm_kwargs,
                    progress_cb=progress_cb,
                )
                filename = "synthese_chrono_courte.md"
            else:
                # Mode long : synthèse chronologique détaillée doc par doc
                md = summarize_chrono_long(
                    case_id,
                    case_title=case_title,
                    date_from=date_from,
                    date_to=date_to,
                    include_monthly=include_monthly,
                    max_docs=max_docs,
                    llm=llm_call,
                    llm_kwargs=llm_kwargs,
                    progress_cb=progress_cb,
                )
                filename = "synthese_chrono.md"

            path = write_summary_file(case_id, md, name=filename)
            data = json.dumps({"path": path, "mode": mode}, ensure_ascii=False)
            queue.put_nowait(sse_event("done", data))
        except Exception as e:
            logger.exception("routes_chrono: job failed")
            # En cas d'erreur : event 'error' avec message (le front affiche un alert)
            data = json.dumps({"message": str(e), "mode": mode}, ensure_ascii=False)
            queue.put_nowait(sse_event("error", data))
            # On termine le flux proprement
            queue.put_nowait(
                sse_event(
                    "done",
                    json.dumps({"path": None, "mode": mode}, ensure_ascii=False),
                )
            )
        finally:
            # Restaure le comportement normal de mise en veille
            if sleep_token is not None:
                try:
                    restore_sleep(sleep_token, logger=logger)
                except Exception:
                    logger.exception("routes_chrono: restore_sleep a échoué")

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.run_in_executor(executor, job)

    async def event_gen():
        while True:
            ev = await queue.get()
            yield ev
            if ev.startswith(b"event: done"):
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream")
