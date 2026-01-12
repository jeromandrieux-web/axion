# backend/app_factory.py
# Création de l'application FastAPI Axion avec montage des routes et des fichiers statiques

import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request, Response

logger = logging.getLogger("axion.app")

# ROOT robuste
try:
    from .config import ROOT
except Exception:
    try:
        from config import ROOT
    except Exception:
        ROOT = str(Path(__file__).resolve().parents[1])


def create_app() -> FastAPI:
    app = FastAPI(title="Axion")

    # BASE project folder
    root = Path(ROOT)

    # Monter les assets statiques et templates (si présents)
    static_dir = root / "backend" / "static" if (root / "backend" / "static").exists() else root / "static"
    templates_dir = root / "backend" / "templates" if (root / "backend" / "templates").exists() else root / "templates"

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # 👉 exposer le stockage des affaires (pour prévisualiser les synthèses générées)
    # on tente d'abord backend/storage, sinon storage/ à la racine ; on monte même si le dir n'existe pas encore
    storage_dir = root / "backend" / "storage"
    if not storage_dir.exists():
        storage_dir = root / "storage"
    try:
        app.mount("/storage", StaticFiles(directory=str(storage_dir), check_dir=False), name="storage")
    except TypeError:
        # FastAPI < 0.100 n'avait pas check_dir : fallback en vérifiant l'existence
        if storage_dir.exists():
            app.mount("/storage", StaticFiles(directory=str(storage_dir)), name="storage")
        else:
            logger.warning("/storage non monté (dossier inexistant et StaticFiles sans check_dir)")

    if templates_dir.exists() and templates_dir.is_dir():
        templates = Jinja2Templates(directory=str(templates_dir))
        # rendre accessible via app.state si d'autres modules veulent l'utiliser
        app.state.templates = templates

    # Routers
    try:
        from .routes_admin import router as admin_router
    except Exception:
        from routes_admin import router as admin_router
    app.include_router(admin_router)

    try:
        from .routes_chat import router as chat_router
    except Exception:
        from routes_chat import router as chat_router
    app.include_router(chat_router)

    # chat sur fichier uploadé (hors affaire)
    try:
        from .routes_chat_upload import router as chat_upload_router
    except Exception:
        from routes_chat_upload import router as chat_upload_router
    app.include_router(chat_upload_router)

    try:
        from .routes_cases import router as cases_router
    except Exception:
        from routes_cases import router as cases_router
    app.include_router(cases_router)

    try:
        from .routes_extract import router as extract_router
    except Exception:
        from routes_extract import router as extract_router
    app.include_router(extract_router)

    try:
        from .routes_leaf import router as leaf_router
    except Exception:
        from routes_leaf import router as leaf_router
    app.include_router(leaf_router)

    try:
        from .routes_ingest import router as ingest_router
    except Exception:
        from routes_ingest import router as ingest_router
    app.include_router(ingest_router)

    # route person_evidence (insérée ici de façon sûre)
    try:
        from .routes_person_evidence import router as person_evidence_router
    except Exception:
        try:
            from routes_person_evidence import router as person_evidence_router
        except Exception:
            person_evidence_router = None
    if person_evidence_router:
        app.include_router(person_evidence_router)

    # 👉 NEW: routes téléphone (lookup + fiche téléphone)
    phones_router = None
    try:
        from .routes_phones import router as phones_router
    except Exception:
        try:
            from routes_phones import router as phones_router
        except Exception:
            phones_router = None
    if phones_router:
        app.include_router(phones_router)

    # 👉 route de synthèse chronologique (SSE)
    try:
        from .routes_chrono import router as chrono_router
    except Exception:
        from routes_chrono import router as chrono_router
    app.include_router(chrono_router)
    
    # Synthèse longue (SSE) — version propre basée sur summarizer_long.py
    long_summary_router = None
    try:
        from .routes_summarize_long import router as long_summary_router
    except Exception:
        try:
            from routes_summarize_long import router as long_summary_router
        except Exception:
            long_summary_router = None

    if long_summary_router:
        app.include_router(long_summary_router)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/apple-touch-icon.png", include_in_schema=False)
    @app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
    def apple_icon():
        return Response(status_code=204)

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/readyz")
    def readyz():
        try:
            from model_utils import get_models_from_env
            allowed = {"mistral:latest", "deepseek-v3.1:671b-cloud"}
            models = [m for m in (get_models_from_env() or []) if m in allowed]
            return {"ok": bool(models)}
        except Exception:
            logger.exception("readyz check failed")
            return {"ok": False}

    # route index (si templates montés)
    if hasattr(app.state, "templates"):
        templates = app.state.templates

        @app.get("/", include_in_schema=False)
        def index(request: Request):
            return templates.TemplateResponse("index.html", {"request": request, "title": "Axion"})

    return app
