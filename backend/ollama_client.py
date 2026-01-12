# backend/ollama_client.py
# Client simplifié pour Ollama LLM via son API HTTP locale.
# Auteur: Jax
#
# ✅ Corrections majeures :
# - Supporte max_tokens (alias) ET num_predict (Ollama) sans TypeError
# - DEFAULT_NUM_PREDICT vient de .env (OLLAMA_NUM_PREDICT) au lieu de None
# - Envoie aussi top_k / repeat_penalty / mirostat_* si présents
# - Stream : /api/chat OK, mais system prompt configurable
# - Logs plus propres

import os
import json
import logging
import requests
from typing import Iterator, Optional, Dict, Any, List, Callable

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

# Modèle forcé (Axion)
FORCED_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "deepseek-v3.1:671b-cloud")

# Contexte / génération
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "65536"))

# ✅ Important : ne pas laisser None par défaut, sinon tu perds le contrôle.
# On aligne sur .env, avec fallback raisonnable.
DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", os.getenv("DEFAULT_MAX_TOKENS", "3072")))

DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))
DEFAULT_TOP_K = int(os.getenv("OLLAMA_TOP_K", "40"))
DEFAULT_REPEAT_PENALTY = float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.05"))

# mirostat (0 désactive)
DEFAULT_MIROSTAT_ETA = float(os.getenv("OLLAMA_MIROSTAT_ETA", "0"))
DEFAULT_MIROSTAT_TAU = float(os.getenv("OLLAMA_MIROSTAT_TAU", "0"))

# timeouts et TLS
TIMEOUT = (
    int(os.getenv("OLLAMA_CONNECT_TIMEOUT", "5")),
    int(os.getenv("OLLAMA_READ_TIMEOUT", "600")),
)

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
VERIFY_TLS = os.getenv("OLLAMA_VERIFY", "1") != "0"

SESSION = requests.Session()
SESSION.verify = VERIFY_TLS
if OLLAMA_API_KEY:
    SESSION.headers.update({"Authorization": f"Bearer {OLLAMA_API_KEY}"})


def _resolve_num_predict(
    *,
    num_predict: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> int:
    """
    Unifie les noms.
    Priorité :
      - num_predict (Ollama)
      - max_tokens (alias)
      - DEFAULT_NUM_PREDICT (.env)
    """
    val = num_predict if num_predict is not None else max_tokens
    if val is None:
        val = DEFAULT_NUM_PREDICT
    try:
        val = int(val)
    except Exception:
        val = DEFAULT_NUM_PREDICT
    # garde-fou
    if val < 1:
        val = DEFAULT_NUM_PREDICT
    return val


def _options(
    *,
    num_predict: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Options Ollama cohérentes avec .env.
    """
    n_pred = _resolve_num_predict(num_predict=num_predict, max_tokens=max_tokens)

    opts: Dict[str, Any] = {
        "num_ctx": DEFAULT_NUM_CTX,
        "num_predict": n_pred,
        "temperature": float(DEFAULT_TEMPERATURE if temperature is None else temperature),
        "top_p": float(DEFAULT_TOP_P if top_p is None else top_p),
        "top_k": int(DEFAULT_TOP_K if top_k is None else top_k),
        "repeat_penalty": float(DEFAULT_REPEAT_PENALTY if repeat_penalty is None else repeat_penalty),
    }

    # mirostat : si tau/eta > 0, on le met (sinon on n’envoie rien)
    try:
        eta = float(DEFAULT_MIROSTAT_ETA)
        tau = float(DEFAULT_MIROSTAT_TAU)
        if eta > 0 and tau > 0:
            # Ollama : mirostat=1 active généralement (selon versions)
            opts["mirostat"] = 1
            opts["mirostat_eta"] = eta
            opts["mirostat_tau"] = tau
    except Exception:
        pass

    return opts


# backward-compatible alias attendu ailleurs dans le code
def _build_options(
    model: str,
    *,
    num_predict: Optional[int] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
) -> Dict[str, Any]:
    return _options(
        num_predict=num_predict,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
    )


def generate(
    model: str,
    prompt: str,
    *,
    num_predict: Optional[int] = None,
    max_tokens: Optional[int] = None,  # ✅ alias accepté
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
) -> str:
    """
    Appel simple → /api/generate, non-stream.
    """
    payload = {
        "model": FORCED_MODEL or model,
        "prompt": prompt,
        "stream": False,
        "options": _options(
            num_predict=num_predict,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
        ),
    }

    url = f"{BASE_URL}/api/generate"
    logger.debug(
        "ollama.generate -> url=%s model=%s prompt_len=%d snippet=%r options=%r",
        url,
        payload["model"],
        len(prompt or ""),
        (prompt or "")[:200].replace("\n", " "),
        payload["options"],
    )

    r = SESSION.post(url, json=payload, timeout=TIMEOUT)
    r.raise_for_status()

    data = r.json()
    if not isinstance(data, dict):
        return ""
    return data.get("response", "") or ""


def stream_generate(
    model: str,
    prompt: str,
    *,
    num_predict: Optional[int] = None,
    max_tokens: Optional[int] = None,  # ✅ alias accepté
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    deanonymize: Optional[Callable[[str], str]] = None,
    system_prompt: Optional[str] = None,
) -> Iterator[str]:
    """
    Stream via /api/chat.
    """
    url = f"{BASE_URL}/api/chat"
    payload = {
        "model": FORCED_MODEL or model,
        "stream": True,
        "messages": [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "options": _build_options(
            model,
            num_predict=num_predict,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
        ),
    }

    logger.debug(
        "ollama.stream_generate -> url=%s model=%s prompt_len=%d snippet=%r options=%r",
        url,
        payload["model"],
        len(prompt or ""),
        (prompt or "")[:300].replace("\n", " "),
        payload["options"],
    )

    try:
        with SESSION.post(url, json=payload, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw:
                    continue

                if isinstance(raw, bytes):
                    line = raw.decode("utf-8", errors="ignore")
                else:
                    line = raw

                try:
                    obj = json.loads(line)
                except Exception:
                    logger.debug("failed to parse chat stream line: %r", line)
                    continue

                msg = obj.get("message") or {}
                chunk = msg.get("content") or obj.get("response")
                if chunk:
                    if deanonymize:
                        try:
                            chunk = deanonymize(chunk)
                        except Exception:
                            logger.exception("stream_generate: deanonymize hook failed")
                    yield chunk

                if obj.get("done"):
                    break
    except Exception:
        logger.exception("stream_generate: chat stream failed for model=%s", model)
        raise


def generate_long(
    model: str,
    prompt: str,
    *,
    max_tokens: int = 4096,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repeat_penalty: Optional[float] = None,
    deanonymize: Optional[Callable[[str], str]] = None,
) -> str:
    """
    Produit une longue réponse en concaténant un stream.
    En cas d'erreur, fallback sur generate().
    """
    parts: List[str] = []
    try:
        for chunk in stream_generate(
            model,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            deanonymize=deanonymize,
        ):
            parts.append(chunk)
    except Exception:
        logger.exception("generate_long failed for model=%s, falling back to generate()", model)
        try:
            txt = generate(
                model,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repeat_penalty=repeat_penalty,
            )
            if deanonymize:
                try:
                    txt = deanonymize(txt)
                except Exception:
                    logger.exception("generate_long: deanonymize hook failed on fallback")
            return txt
        except Exception:
            logger.exception("generate() fallback also failed for model=%s", model)
            return ""

    return "".join(parts)
