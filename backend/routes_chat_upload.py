# backend/routes_chat_upload.py
# Chargement d’un document pour le chat (sans résumé auto)
# Auteur: Jax

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from ingest import chunks_from_bytes

logger = logging.getLogger("axion.routes_chat_upload")
router = APIRouter()

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
UPLOAD_DOC_MAX_CHARS = int(os.getenv("UPLOAD_DOC_MAX_CHARS", "60000"))


@router.post("/api/chat/upload")
async def chat_upload_document(
    file: UploadFile = File(...),
):
    """
    Charge un document, en extrait le texte, et le renvoie au front.
    AUCUN appel LLM ici.

    Le texte pourra ensuite être utilisé comme contexte (doc_text) dans /api/chat/stream.
    """
    try:
        logger.info("chat_upload: receiving file name=%s content_type=%s", file.filename, file.content_type)
        content = await file.read()
        logger.debug("chat_upload: received %d bytes", len(content))

        if not content:
            return JSONResponse(
                {"ok": False, "error": "Fichier vide ou illisible."},
                status_code=400,
            )

        if len(content) > MAX_UPLOAD_BYTES:
            return JSONResponse(
                {
                    "ok": False,
                    "error": f"Fichier trop volumineux (max {MAX_UPLOAD_BYTES // (1024*1024)} Mo).",
                },
                status_code=400,
            )

        # Extraction simple via chunks_from_bytes (PDF, DOCX, TXT…)
        results = chunks_from_bytes(content, file.filename)
        extracted_parts = []

        if results:
            for r in results:
                if isinstance(r, tuple):
                    txt = r[0] or ""
                else:
                    txt = (
                        r.get("text")
                        or r.get("content")
                        or r.get("page_content")
                        or r.get("chunk")
                        or ""
                    )
                if txt:
                    extracted_parts.append(txt)

        full_ctx = "\n\n".join(extracted_parts).strip()

        if not full_ctx:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "Je n’ai pas réussi à extraire du texte exploitable de ce fichier.",
                },
                status_code=400,
            )

        # On limite un peu pour ne pas exploser le contexte, mais on reste généreux
        doc_text = full_ctx[:UPLOAD_DOC_MAX_CHARS]

        logger.debug("chat_upload: doc_text_len=%d snippet=%r", len(doc_text or ""), (doc_text or "")[:200].replace("\n"," "))
        return {"ok": True, "filename": file.filename, "doc_text": doc_text}
    except Exception:
        logger.exception("chat_upload: failed to process upload")
        raise HTTPException(status_code=500, detail="upload failed")
