# backend/config.py
# Configuration centrale de l'application Axion.
# Auteur: Jax

import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
load_dotenv(dotenv_path=str(ROOT / ".env"))

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except Exception:
        logger.warning("Invalid int for %s=%r, using %s", name, v, default)
        return default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        logger.warning("Invalid float for %s=%r, using %s", name, v, default)
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v == "":
        return default
    return str(v).strip() not in ("0", "false", "False", "no", "n", "")

# Configuration principale
STORAGE = Path(os.getenv("STORAGE_DIR", str(ROOT / "backend" / "storage"))).resolve()
# Ensure storage path exists to avoid later IO errors
try:
    STORAGE.mkdir(parents=True, exist_ok=True)
except Exception:
    logger.exception("unable to create STORAGE dir %s", STORAGE)

# Paramètres RAG par défaut
DEFAULT_TOP_K = _env_int("TOP_K", 20)
DEFAULT_NEIGHBOR_EXPAND = _env_int("NEIGHBOR_EXPAND", 1)
DEFAULT_MMR_LAMBDA = _env_float("MMR_LAMBDA", 0.6)
DEFAULT_MAX_PER_DOC = _env_int("MAX_PER_DOC", 4)

# Paramètres spécifiques à deepseek-v3.1:671b-cloud (modèle cloud puissant)
TOP_K_DEEPSEEK_V3_1_671B_CLOUD = _env_int("TOP_K_DEEPSEEK_V3_1_671B_CLOUD", 25)
NEIGHBOR_EXPAND_DEEPSEEK_V3_1_671B_CLOUD = _env_int("NEIGHBOR_EXPAND_DEEPSEEK_V3_1_671B_CLOUD", 2)
MMR_LAMBDA_DEEPSEEK_V3_1_671B_CLOUD = _env_float("MMR_LAMBDA_DEEPSEEK_V3_1_671B_CLOUD", 0.5)
MAX_PER_DOC_DEEPSEEK_V3_1_671B_CLOUD = _env_int("MAX_PER_DOC_DEEPSEEK_V3_1_671B_CLOUD", 5)

# Paramètres spécifiques à mistral:latest
TOP_K_QWEN3_8B = _env_int("TOP_K_QWEN3_8B", 20)
NEIGHBOR_EXPAND_QWEN3_8B = _env_int("NEIGHBOR_EXPAND_QWEN3_8B", 1)
MMR_LAMBDA_QWEN3_8B = _env_float("MMR_LAMBDA_QWEN3_8B", 0.6)
MAX_PER_DOC_QWEN3_8B = _env_int("MAX_PER_DOC_QWEN3_8B", 4)

# Paramètres OCR
OCR_ENABLED = _env_bool("OCR_ENABLE", True)
OCR_LANG = os.getenv("OCR_LANG", "fra")
OCR_DPI = _env_int("OCR_DPI", 300)
OCR_MIN_CHARS = _env_int("OCR_MIN_CHARS", 80)

# Paramètres chunks
CHUNK_SIZE = _env_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 120)

# Paramètres LEAF
LEAF_TARGET_PAGES = _env_int("LEAF_TARGET_PAGES", 25)
LEAF_WORDS_PER_PAGE = _env_int("LEAF_WORDS_PER_PAGE", 580)
LEAF_GROUP_SIZE = _env_int("LEAF_GROUP_SIZE", 6)
LEAF_TOP_K = _env_int("LEAF_TOP_K", 100)
LEAF_NUM_PREDICT = _env_int("LEAF_NUM_PREDICT", 1536)
LEAF_SECTION_TOPK_EACH = _env_int("LEAF_SECTION_TOPK_EACH", 12)
LEAF_SECTION_MAX_CHARS = _env_int("LEAF_SECTION_MAX_CHARS", 5500)

# Paramètres analyse
ANALYZE_TOP_K = _env_int("ANALYZE_TOP_K", 60)
ANALYZE_MAX_ROWS = _env_int("ANALYZE_MAX_ROWS", 40)
ANALYZE_MAX_TIMELINE = _env_int("ANALYZE_MAX_TIMELINE", 120)

# Context tokens (adaptés aux modèles cloud)
OLLAMA_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 8192)
OLLAMA_NUM_PREDICT = _env_int("OLLAMA_NUM_PREDICT", 900)

# Paramètres cloud spécifiques
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

