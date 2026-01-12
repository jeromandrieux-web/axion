# backend/llm_client.py
# fonctions utilitaires pour appels LLM hors route /api/chat
# Auteur: Jax

"""
llm_client.py — petit wrapper LLM générique pour les appels hors route /api/chat

Stratégie par défaut :
- choisit un modèle via env / model_utils
- appelle l’endpoint HTTP local /api/chat en mode non-stream pour obtenir une réponse

Variables d'environnement reconnues :
- AXION_BASE_URL : base URL du serveur (défaut: http://127.0.0.1:8000)
- AXION_DEFAULT_MODEL : modèle à utiliser si non fourni
- CHAT_MESSAGE_MAX_LEN : limite dure côté API (défaut: 10000)

⚠️ L'anonymisation / dé-pseudonymisation (noms, téléphones) est gérée côté backend
dans chat_service.py via identity_mapper.py. Ce client se contente de forwarder
le system, le message, le case_id et le modèle.
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Optional

try:
    from backend.model_utils import get_models_from_env as _get_models
except Exception:  # pragma: no cover
    try:
        from model_utils import get_models_from_env as _get_models
    except Exception:  # pragma: no cover
        _get_models = lambda: []  # type: ignore


def _pick_model(preferred: Optional[str] = None) -> Optional[str]:
    if preferred:
        return preferred
    env_default = os.getenv("AXION_DEFAULT_MODEL")
    if env_default:
        return env_default
    models = _get_models() or []
    for m in ("deepseek-v3.1:671b-cloud", "mistral:latest"):
        if m in models:
            return m
    return models[0] if models else None


def llm_call(system: str, user: str, **kwargs) -> str:
    """Appel LLM simple via l'endpoint local /api/chat (non-stream).

    kwargs reconnus:
      - model: str (optionnel)
      - case_id: str (optionnel)
      - mode: str (optionnel, défaut 'plain')
      - max_tokens: int (optionnel, passé à l'API /api/chat)
    """
    base = os.getenv("AXION_BASE_URL", "http://127.0.0.1:8000")
    model = _pick_model(kwargs.get("model"))
    if not model:
        raise RuntimeError("Aucun modèle disponible (configurer AXION_DEFAULT_MODEL ou model_utils)")

    # === Respect de la contrainte ChatRequest.message (max_length=30000) ===
    max_len = int(os.getenv("CHAT_MESSAGE_MAX_LEN", "30000"))
    msg = user or ""
    if len(msg) > max_len:
        note = "\n\n[Note: le contexte a été tronqué pour respecter la limite de longueur du message.]"
        keep = max_len - len(note)
        if keep < 0:
            keep = max_len
            note = ""
        msg = msg[:keep] + note

    max_tokens = kwargs.get("max_tokens")  # peut être None

    # case_id propre (ou None)
    raw_case_id = kwargs.get("case_id")
    if isinstance(raw_case_id, str):
        case_id = raw_case_id.strip() or None
    else:
        case_id = raw_case_id or None

    payload = {
        "message": msg,
        "model": model,
        "case_id": case_id,
        "mode": kwargs.get("mode") or "plain",
        # Champs supplémentaires, ignorés par ChatRequest mais utiles côté service LLM
        "system": system,
        "stream": "plain",
    }
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    url = base.rstrip("/") + "/api/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise RuntimeError(
            f"HTTP {e.code} depuis /api/chat: {e.reason} ; response_body={body!r}"
        )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Appel /api/chat échoué: {e}")

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("answer") is not None:
            return str(obj["answer"]).strip()
        return raw.strip()
    except Exception:
        return raw.strip()
