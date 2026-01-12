# backend/model_utils.py
# Gestion centralisée des modèles et paramètres selon le backend.
# Auteur: Jax

import os
import re
import logging
from functools import lru_cache
from typing import List
from typing import Tuple

logger = logging.getLogger(__name__)

# modèles autorisés : lettres, chiffres, :, -, _, ., longueur limite
_MODEL_RE = re.compile(r'^[A-Za-z0-9:_\-\.\+]{1,128}$')

def order_and_filter_models(models: List[str]) -> List[str]:
    """Ordonne les modèles selon vos préférences : deepseek-v3.1:671b-cloud -> mistral:latest"""
    cleaned = []
    for m in models:
        if not m:
            continue
        mm = m.strip()
        if not mm:
            continue
        # valider forme du nom de modèle pour éviter valeurs inattendues
        if not _MODEL_RE.fullmatch(mm):
            logger.debug("skipping invalid model name from env: %r", mm)
            continue
        cleaned.append(mm)
    
    # Priorité : deepseek-v3.1:671b-cloud d'abord, puis mistral:latest
    def score(m: str):
        ml = m.lower()
        if "deepseek-v3.1:671b-cloud" in ml:
            return 0  # Priorité la plus haute
        elif "mistral:latest" in ml:
            return 1  # Deuxième priorité
        else:
            return 2  # Autres modèles (au cas où)
    
    # Déduplication en préservant l'ordre initial
    seen = set()
    unique = []
    for m in cleaned:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    
    # Tri par priorité
    return sorted(unique, key=score)

@lru_cache(maxsize=1)
def get_models_from_env() -> List[str]:
    raw = (os.getenv("OLLAMA_MODELS", "") or "").strip()
    if not raw:
        # Fallback avec vos modèles préférés
        raw = "deepseek-v3.1:671b-cloud,mistral:latest"
    models = [x.strip() for x in raw.split(",") if x.strip()]
    return order_and_filter_models(models)

def effective_retrieval_params(model: str) -> tuple:
    """Retourne les paramètres RAG selon le modèle"""
    from config import DEFAULT_TOP_K, DEFAULT_NEIGHBOR_EXPAND, DEFAULT_MMR_LAMBDA, DEFAULT_MAX_PER_DOC
    
    m = (model or "").lower()

    def pick(prefix: str):
        try:
            tk = int(os.getenv(f"TOP_K_{prefix}", str(DEFAULT_TOP_K)))
            nb = int(os.getenv(f"NEIGHBOR_EXPAND_{prefix}", str(DEFAULT_NEIGHBOR_EXPAND)))
            ml = float(os.getenv(f"MMR_LAMBDA_{prefix}", str(DEFAULT_MMR_LAMBDA)))
            mpd = int(os.getenv(f"MAX_PER_DOC_{prefix}", str(DEFAULT_MAX_PER_DOC)))
        except Exception:
            logger.exception("invalid numeric env for retrieval params, falling back to defaults for %s", prefix)
            return DEFAULT_TOP_K, DEFAULT_NEIGHBOR_EXPAND, DEFAULT_MMR_LAMBDA, DEFAULT_MAX_PER_DOC
        return tk, nb, ml, mpd

    # Mapping spécifique à vos modèles
    if "deepseek-v3.1:671b-cloud" in m:
        return pick("DEEPSEEK_V3_1_671B_CLOUD")
    elif "mistral:latest" in m:
        return pick("QWEN3_8B")
    
    # Par défaut
    return DEFAULT_TOP_K, DEFAULT_NEIGHBOR_EXPAND, DEFAULT_MMR_LAMBDA, DEFAULT_MAX_PER_DOC

@lru_cache(maxsize=1)
def get_default_model() -> str:
    """Retourne le modèle par défaut (deepseek-v3.1:671b-cloud si disponible)"""
    models = get_models_from_env()
    if models:
        return models[0]  # Le premier de la liste triée (deepseek en priorité)
    return "mistral:latest"  # Fallback

