# backend/routes_chat.py
# Routes de chat (simple et streaming SSE)
# Auteur: Jax

import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models import ChatRequest, ChatResponse
from chat_service import ChatService

router = APIRouter()
logger = logging.getLogger("axion.routes_chat")

# 🔥 Modèle unique pour toute l'application
FORCED_MODEL = "deepseek-v3.1:671b-cloud"

# Headers SSE pour éviter le buffering proxy
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

chat_service = ChatService()


def _force_model(req: ChatRequest) -> ChatRequest:
    """
    On force systématiquement le modèle DeepSeek et on ignore
    tout modèle éventuellement envoyé par le front.
    """
    original = getattr(req, "model", None)
    req.model = FORCED_MODEL
    logger.debug("routes_chat._force_model: original_model=%r forced_model=%r", original, FORCED_MODEL)
    return req


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Endpoint chat simple (non-stream).
    """
    try:
        # Diagnostic : log court des champs importants sans afficher tout le document
        try:
            doc_len = len(getattr(req, "doc_text", "") or "")
        except Exception:
            doc_len = None
        logger.debug(
            "routes_chat.chat: incoming model=%s case_id=%s use_memory=%s doc_text_len=%s",
            getattr(req, "model", None),
            getattr(req, "case_id", None),
            getattr(req, "use_memory", None),
            doc_len,
        )

        req = _force_model(req)
        return await chat_service.chat(req)
    except Exception as e:
        logger.exception("chat error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Endpoint chat streaming SSE.
    """
    try:
        # Diagnostic : log court des champs importants sans afficher tout le document
        try:
            doc_len = len(getattr(req, "doc_text", "") or "")
        except Exception:
            doc_len = None
        logger.debug(
            "routes_chat.chat_stream: incoming model=%s case_id=%s use_memory=%s doc_text_len=%s",
            getattr(req, "model", None),
            getattr(req, "case_id", None),
            getattr(req, "use_memory", None),
            doc_len,
        )

        req = _force_model(req)
        resp = await chat_service.chat_stream(req)

        if isinstance(resp, StreamingResponse):
            resp.headers.update(_SSE_HEADERS)

        return resp

    except HTTPException:
        # on laisse remonter telles quelles les HTTPException explicites
        raise
    except Exception as e:
        logger.exception("chat_stream error")
        msg = str(e)

        async def _err():
            yield f"event: error\ndata: {msg}\n\n"

        return StreamingResponse(
            _err(),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
