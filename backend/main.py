# backend/main.py
# Point d'entrée principal de l'application Axion.
# Auteur: Jax

import uvicorn
import importlib
import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles

# 🔇 Désactiver explicitement le parallélisme des tokenizers HF
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 👇 on force le chargement du .env à la racine du projet
ROOT_DIR = Path(__file__).resolve().parent.parent  # /Users/jax/axion
env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=os.getenv("AXION_LOG_LEVEL", "INFO"))
logger = logging.getLogger("axion.main")

BACKEND_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.dirname(BACKEND_DIR))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def _load_create_app():
    last_err = None
    for name in ("backend.app_factory", "app_factory"):
        try:
            mod = importlib.import_module(name)
            create = getattr(mod, "create_app", None)
            if callable(create):
                return create
            last_err = RuntimeError(f"{name}.create_app non callable")
        except Exception as e:
            last_err = e
            logger.debug("import %s failed: %s", name, e)
    raise ModuleNotFoundError("backend/app_factory.py introuvable") from last_err

create_app = _load_create_app()
try:
    app = create_app()
except Exception:
    logger.exception("échec lors de l'initialisation de l'application")
    raise

app.mount("/static", StaticFiles(directory="backend/static"), name="static")

if __name__ == "__main__":
    host = os.getenv("AXION_HOST", "0.0.0.0")
    port = int(os.getenv("AXION_PORT", "8000"))
    reload_flag = os.getenv("UVICORN_RELOAD", "0") == "1"
    logger.info("Démarrage uvicorn host=%s port=%s reload=%s", host, port, reload_flag)
    uvicorn.run("backend.main:app", host=host, port=port, reload=reload_flag)
