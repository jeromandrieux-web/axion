# backend/llm_profiles.py
# Profils de génération LLM selon le type de tâche
# Auteur : Jax

from __future__ import annotations
import os
from typing import Literal, Dict, Any

TaskType = Literal[
    "chat",
    "chat_deep",
    "doc_quick",
    "doc_deep",
    "long_summary",
]

SizeHint = Literal["short", "long", "inventory"]

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# Valeurs "par défaut" globales
DEFAULT_MAX_TOKENS = _env_int("DEFAULT_MAX_TOKENS", 3072)

# ⚖️ Pour du judiciaire/RAG : une température modérée est plus stable
DEFAULT_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.2)
DEFAULT_TOP_P = _env_float("OLLAMA_TOP_P", 0.9)

# Hard max lié au backend : on préfère OLLAMA_NUM_PREDICT si défini, sinon DeepSeek override, sinon default
OLLAMA_NUM_PREDICT = _env_int("OLLAMA_NUM_PREDICT", 0)
DEEPSEEK_NUM_PREDICT = _env_int("DEEPSEEK_V3_1_671B_CLOUD_NUM_PREDICT", 0)

MODEL_MAX_TOKENS = (
    OLLAMA_NUM_PREDICT if OLLAMA_NUM_PREDICT > 0
    else (DEEPSEEK_NUM_PREDICT if DEEPSEEK_NUM_PREDICT > 0 else DEFAULT_MAX_TOKENS)
)

# ✅ Plafond de sécurité (évite qu’un .env bizarre demande 200k tokens de sortie)
HARD_CAP = _env_int("HARD_MAX_TOKENS_CAP", 16384)
MODEL_MAX_TOKENS = min(MODEL_MAX_TOKENS, HARD_CAP)

# ✅ Boost inventaire (si tu veux sortir des listes longues / recensements)
INVENTORY_MAX_TOKENS = _env_int("INVENTORY_MAX_TOKENS", min(12288, MODEL_MAX_TOKENS))

def get_profile(task: TaskType, size_hint: str | None = None) -> Dict[str, Any]:
    """
    Renvoie un dict avec :
      - num_predict
      - temperature
      - top_p
    """

    num_predict = min(DEFAULT_MAX_TOKENS, MODEL_MAX_TOKENS)
    temperature = DEFAULT_TEMPERATURE
    top_p = DEFAULT_TOP_P

    if task == "chat":
        # discussion normale
        num_predict = min(DEFAULT_MAX_TOKENS, MODEL_MAX_TOKENS)

    elif task == "chat_deep":
        # RAG + analyse : on pousse un peu
        num_predict = min(int(DEFAULT_MAX_TOKENS * 1.3), MODEL_MAX_TOKENS)

        # En contexte enquête/RAG, la stabilité est prioritaire : on évite les températures trop hautes
        # (si l'utilisateur les a réglées haut dans .env, on n'écrase pas, on borne juste)
        temperature = min(temperature, 0.7)

    elif task == "doc_quick":
        base = int(DEFAULT_MAX_TOKENS * 0.8)
        num_predict = min(base, MODEL_MAX_TOKENS)

    elif task == "doc_deep":
        base = int(DEFAULT_MAX_TOKENS * 1.2)
        num_predict = min(base, MODEL_MAX_TOKENS)
        temperature = min(temperature, 0.7)

    elif task == "long_summary":
        num_predict = _env_int("LONG_SUM_FINAL_MAX_TOKENS", 2048)
        num_predict = min(num_predict, MODEL_MAX_TOKENS)

    # ✅ size hints
    if size_hint == "short":
        num_predict = min(num_predict, 1024)

    elif size_hint == "long":
        num_predict = min(int(num_predict * 1.1), MODEL_MAX_TOKENS)

    elif size_hint == "inventory":
        # Forcer un budget de sortie plus long (utile pour recenser des actes/PV)
        num_predict = min(max(num_predict, INVENTORY_MAX_TOKENS), MODEL_MAX_TOKENS)
        temperature = min(temperature, 0.7)

    return {
        "num_predict": int(num_predict),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
