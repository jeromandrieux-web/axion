# backend/routes_summarize_long.py
# Synthèse longue d'enquête — DÉSACTIVÉE dans cette version
# Auteur: Jax (adapté)

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("axion.routes_summarize_long")
router = APIRouter()


@router.post("/api/summary/long/stream")
async def summary_long_stream(request: Request):
    """
    Cette route existait pour lancer la 'synthèse longue d'enquête' hiérarchique.

    Dans cette version, la fonctionnalité est désactivée :
    - La synthèse thématique (LEAF) remplace ce besoin.
    - La synthèse chronologique + rapport de synthèse couvrent les autres cas.

    On renvoie simplement une erreur claire au cas où quelque chose l'appellerait encore.
    """
    logger.info("Appel à /api/summary/long/stream alors que la synthèse longue est désactivée.")
    raise HTTPException(
        status_code=410,
        detail=(
            "La synthèse longue d'enquête est désactivée dans cette version. "
            "Utilisez la synthèse thématique (LEAF) ou la synthèse chronologique."
        ),
    )
