# backend/routes_admin.py
# Routes d'administration pour la gestion des modèles
# Auteur: Jax

import logging
import os
from typing import List
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from model_utils import get_models_from_env, get_default_model

router = APIRouter()
logger = logging.getLogger("axion.routes_admin")

# whitelist configurable via AXION_ALLOWED_MODELS (comma separated). Falls back to builtin list.
ALLOWED = set(
    m.strip()
    for m in os.getenv("AXION_ALLOWED_MODELS", "mistral:latest,deepseek-v3.1:671b-cloud").split(",")
    if m.strip()
)

def _models() -> List[str]:
    try:
        models = [m for m in (get_models_from_env() or []) if m in ALLOWED]
        if not models:
            logger.info("Aucun modèle disponible après filtrage ALLOWED=%s", sorted(ALLOWED))
        return models
    except Exception:
        logger.exception("get_models_from_env failed")
        return []

@router.get("/", response_class=HTMLResponse)
def ui(request: Request):
    models = _models()
    default_model = None
    try:
        default_model = get_default_model()
    except Exception:
        logger.exception("get_default_model failed")
    if default_model not in models:
        default_model = models[0] if models else None

    templates = getattr(request.app.state, "templates", None)
    ctx = {"request": request, "models": models, "default_model": default_model}
    if templates:
        return templates.TemplateResponse("index.html", ctx)
    body = f"""<html><body>
        <h1>Axion</h1>
        <p>Modèles disponibles: {", ".join(models) if models else "(aucun)"}<br/>
        Modèle par défaut: {default_model or "(non défini)"}.</p>
    </body></html>"""
    return HTMLResponse(body)

@router.get("/api/models")
def list_models():
    return JSONResponse({"models": _models()})
