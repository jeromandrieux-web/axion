# backend/ingest.py
# Ingestion de documents dans une affaire : OCR, découpage, embeddings, indexation.
# Auteur: Jax
#
# ✅ Version améliorée (Axion) :
# - Ajout de métadonnées d’UNITÉ / ACTE (unit_id, unit_type, page_start/end)
# - Ajout d’un "UNIT HEADER CHUNK" (ancre acte/PV) pour améliorer la couverture RAG
# - Chunking toujours par page (citation page fiable), mais enrichi par contexte d’acte
# - Préfixe e5 "passage:" conservé
# - Compatibilité rétro : bm25.json garde chunks/sources/tokens, mais sources enrichies

import os
import re
import json
import time
import hashlib
import unicodedata
import logging
import shutil
import io
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple, Any

import numpy as np
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from time import perf_counter

# 👉 métadonnées (date / type) + découpage logique (unités PV/rapports)
# Supporte exécution avec ou sans package "backend"
try:
    from utils_ingest import (  # type: ignore
        detect_doc_date,
        detect_doc_type,
        split_pages_into_units,
    )
except Exception:  # pragma: no cover
    from backend.utils_ingest import (  # type: ignore  # noqa: E402
        detect_doc_date,
        detect_doc_type,
        split_pages_into_units,
    )

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ProgressCB = Optional[Callable[[dict], None]]

# ---------- Config ----------
STORAGE = Path(os.getenv("STORAGE_DIR", "backend/storage")).resolve()

# OCR
OCR_ENABLE = os.getenv("OCR_ENABLE", "1") == "1"
OCR_FORCE = os.getenv("OCR_FORCE", "0") == "1"
OCR_LANG = os.getenv("OCR_LANG", "fra")
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
OCR_MIN_CHARS = int(os.getenv("OCR_MIN_CHARS", "40"))

# Chunking (⚠️ CHUNK_SIZE en caractères, pas tokens)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ✅ Ajouts : ancre unité/acte
UNIT_ANCHOR_ENABLE = os.getenv("UNIT_ANCHOR_ENABLE", "1") == "1"
UNIT_ANCHOR_MAX_CHARS = int(os.getenv("UNIT_ANCHOR_MAX_CHARS", "2400"))
UNIT_ANCHOR_HEAD_PAGES = int(os.getenv("UNIT_ANCHOR_HEAD_PAGES", "2"))  # pages prises pour l’ancre (début unité)

# Embeddings
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", os.getenv("EMB_MODEL", DEFAULT_EMBED_MODEL))
EMBEDDING_DIM_ENV = os.getenv("EMBEDDING_DIM")

# cache global du modèle d'embedding
_sentence_model: Optional[SentenceTransformer] = None

_WS_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# ✅ Extraction "facts identité" (robuste : multi-lignes + narratif + labels variants)
# ---------------------------------------------------------------------------

_LABELS: Dict[str, List[str]] = {
    "nom": ["nom", "nom de famille"],
    "prenom": ["prénom", "prenom", "prenoms", "prénom(s)", "prenom(s)"],
    "naissance": [
        "date et lieu de naissance",
        "date de naissance",
        "lieu de naissance",
        "naissance",
        "né le",
        "nee le",
        "née le",
    ],
    "filiation": ["filiation", "parents", "père", "pere", "mère", "mere"],
    "adresse": [
        "adresse",
        "domicile",
        "résidence",
        "residence",
        "demeurant",
        "résidant",
        "residant",
        "domicilié",
        "domicilie",
        "domiciliée",
        "habite",
        "demeure",
    ],
    "situation_familiale": [
        "situation familiale",
        "etat matrimonial",
        "état matrimonial",
        "marié",
        "mariée",
        "celibataire",
        "célibataire",
        "divorcé",
        "divorcée",
        "pacsé",
        "pacsée",
        "veuf",
        "veuve",
        "enfants",
        "conjoint",
    ],
    "profession": ["profession", "emploi", "activité", "activite", "occupation"],
    "nationalite": ["nationalité", "nationalite", "nation"],
}


def _fix_ocr_common(s: str) -> str:
    if not s:
        return ""
    return (
        s.replace("N0M", "NOM")
        .replace("PRÉN0M", "PRÉNOM")
        .replace("PREN0M", "PRENOM")
        .replace("D0MICILE", "DOMICILE")
    )


def _norm_label(s: str) -> str:
    """Normalisation légère pour comparer les libellés."""
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = "".join(
        c
        for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


# map label normalisé -> clé canonique
_LABEL_TO_KEY: Dict[str, str] = {}
for _k, _names in _LABELS.items():
    for _n in _names:
        _LABEL_TO_KEY[_norm_label(_n)] = _k


def _compile_label_regex() -> re.Pattern:
    """
    Regex unique qui capte n'importe quel libellé connu dans un groupe "label",
    puis attend ":" ou "-" après le libellé.
    """
    all_labels = sorted(_LABEL_TO_KEY.keys(), key=len, reverse=True)
    pat = (
        r"\b(?P<label>"
        + "|".join(re.escape(l) for l in all_labels)
        + r")\b\s*[:\-]\s*"
    )
    return re.compile(pat, re.IGNORECASE)


_FIELD_START_RE = _compile_label_regex()


def _clean_value(v: str) -> str:
    v = (v or "").strip()
    v = re.sub(r"\s+", " ", v).strip()
    if len(v) > 450:
        v = v[:450].rstrip()
    return v


def _extract_fields_block_multiline(text: str) -> Dict[str, str]:
    """
    Extraction "formulaire":
    - détecte LABEL:
    - prend la valeur sur la ligne
    - absorbe les lignes suivantes jusqu'à un nouveau LABEL:
    """
    if not text:
        return {}

    out: Dict[str, str] = {}
    lines = [l.rstrip() for l in text.splitlines()]

    current_key: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal current_key, buf
        if current_key and buf:
            val = _clean_value(" ".join(x.strip() for x in buf if x.strip()))
            if val and current_key not in out:
                out[current_key] = val
        current_key = None
        buf = []

    for raw in lines[:260]:
        line = _fix_ocr_common(raw).strip()
        if not line:
            continue

        m = _FIELD_START_RE.search(line)
        if m:
            flush()
            lab = _norm_label(m.group("label") or "")
            key = _LABEL_TO_KEY.get(lab)
            if not key:
                continue
            current_key = key
            rest = line[m.end() :].strip()
            if rest:
                buf.append(rest)
            continue

        if current_key:
            buf.append(line)

    flush()
    return out


def _extract_fields_narrative(text: str) -> Dict[str, str]:
    """
    Extraction "narratif" : capte des phrases du type
    - "demeurant ...", "domicilié à ..."
    - "né(e) le ... à ..."
    - "nationalité ...", "profession ...", "situation familiale ..."
    """
    if not text:
        return {}

    t = _fix_ocr_common(text)
    t = re.sub(r"\s+", " ", t).strip()
    out: Dict[str, str] = {}

    # Adresse narratif
    m = re.search(
        r"\b(?:demeurant|résidant|residant|domicili[ée]\s+(?:à|a)|domicilie\s+(?:a|à)|habite|demeure)\b\s*(.+?)"
        r"(?=(?:\b(?:né|nee|née)\b|\bfiliation\b|\bnationalit|\bprofession\b|\bsituation familiale\b|$))",
        t,
        re.IGNORECASE,
    )
    if m:
        out["adresse"] = _clean_value(m.group(1))

    # Naissance narratif
    m = re.search(
        r"\b(?:né|nee|née)\s+le\s+(.+?)"
        r"(?=(?:\bfiliation\b|\badresse\b|\bdemeurant\b|\bnationalit|\bprofession\b|\bsituation familiale\b|$))",
        t,
        re.IGNORECASE,
    )
    if m:
        out["naissance"] = _clean_value(m.group(1))

    # Nationalité
    m = re.search(
        r"\bnationalit[ée]\s*[:\-]?\s*(.+?)(?=(?:\bprofession\b|\bsituation familiale\b|\bfiliation\b|\badresse\b|$))",
        t,
        re.IGNORECASE,
    )
    if m:
        out["nationalite"] = _clean_value(m.group(1))

    # Profession
    m = re.search(
        r"\b(?:profession|emploi|occupation|activit[ée])\b\s*[:\-]?\s*(.+?)(?=(?:\bsituation familiale\b|\bfiliation\b|\badresse\b|$))",
        t,
        re.IGNORECASE,
    )
    if m and "profession" not in out:
        out["profession"] = _clean_value(m.group(1))

    # Situation familiale
    m = re.search(
        r"\b(?:situation familiale|etat matrimonial|état matrimonial)\b\s*[:\-]?\s*(.+?)(?=(?:\bfiliation\b|\badresse\b|\bprofession\b|$))",
        t,
        re.IGNORECASE,
    )
    if m:
        out["situation_familiale"] = _clean_value(m.group(1))

    return out


def _merge_fields(primary: Dict[str, str], secondary: Dict[str, str]) -> Dict[str, str]:
    out = dict(primary or {})
    for k, v in (secondary or {}).items():
        if k not in out and v:
            out[k] = v
    return out


def _extract_identity_fields_robust(text: str) -> Dict[str, str]:
    # Combine formulaire + narratif
    block = _extract_fields_block_multiline(text)
    narr = _extract_fields_narrative(text)
    return _merge_fields(block, narr)


def _fields_are_good_enough(fields: Dict[str, str], doc_type: Optional[str] = None) -> bool:
    """
    Stop condition du scan adaptatif :
    - NOM + PRÉNOM
    - + (NAISSANCE + ADRESSE) idéalement, sinon >=4 champs
    """
    if not fields:
        return False
    if not fields.get("nom") or not fields.get("prenom"):
        return False
    if fields.get("naissance") and fields.get("adresse"):
        return True
    return len(fields) >= 4


def _make_identity_fact_chunk(fields: Dict[str, str]) -> Optional[str]:
    if not fields:
        return None
    if not (fields.get("nom") and fields.get("prenom")):
        return None

    parts: List[str] = []
    for k, label in [
        ("nom", "NOM"),
        ("prenom", "PRÉNOM"),
        ("naissance", "NAISSANCE"),
        ("nationalite", "NATIONALITÉ"),
        ("profession", "PROFESSION"),
        ("situation_familiale", "SITUATION FAMILIALE"),
        ("filiation", "FILIATION"),
        ("adresse", "ADRESSE"),
    ]:
        v = fields.get(k)
        if v:
            parts.append(f"{label}: {v}")

    if len(parts) < 2:
        return None

    return "IDENTITÉ (extrait structuré) — " + " ; ".join(parts)


def _unit_looks_like_identity(
    unit_title: str, doc_type: str, unit_pages: List[Tuple[int, str]]
) -> bool:
    """
    Heuristique :
    - doc_type pv_identite
    - ou titre contient identité/état civil/signalement/filiation
    - ou première page contient >=3 champs reconnus
    """
    t = (unit_title or "").lower()
    dt = (doc_type or "").lower()

    if dt == "pv_identite":
        return True

    if any(
        k in t
        for k in ["identité", "identite", "etat civil", "état civil", "signalement", "filiation"]
    ):
        return True

    blob = "\n".join((txt or "") for _, txt in (unit_pages[:1] if unit_pages else []))
    fields = _extract_identity_fields_robust(blob)
    return len(fields) >= 3


# ---------------------------------------------------------------------------
# ✅ Unités (actes/PV) : id + type + ancre (header chunk)
# ---------------------------------------------------------------------------

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()

def _infer_unit_type(unit: Dict[str, Any], doc_type_fallback: str) -> str:
    """
    On essaie de récupérer un type d’acte depuis split_pages_into_units.
    Si non présent, fallback au doc_type global.
    """
    for key in ("unit_type", "type", "doc_type", "kind"):
        v = unit.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # heuristique sur le titre si besoin
    title = (unit.get("title") or "").lower()
    if any(k in title for k in ["audition", "entretien"]):
        return "pv_audition"
    if any(k in title for k in ["perquisition", "visite domiciliaire"]):
        return "pv_perquisition"
    if any(k in title for k in ["garde à vue", "garde a vue", "gav"]):
        return "pv_gav"
    if any(k in title for k in ["réquisition", "requisition"]):
        return "requisition"
    if any(k in title for k in ["rapport", "synthèse", "synthese"]):
        return "rapport"
    return (doc_type_fallback or "document").strip() or "document"

def _unit_id(source_name: str, unit_index: int, page_start: Optional[int], page_end: Optional[int], title: str) -> str:
    base = f"{source_name}|u{unit_index}|p{page_start or 0}-{page_end or 0}|{title or ''}"
    return _sha1(base)

def _make_unit_anchor_text(
    *,
    source_name: str,
    doc_type: str,
    doc_date: Optional[str],
    unit_title: str,
    unit_type: str,
    page_start: Optional[int],
    page_end: Optional[int],
    unit_pages: List[Tuple[int, str]],
) -> Optional[str]:
    """
    Chunk "ancre" : donne au retrieval un point d’entrée au niveau acte/PV.
    On n’essaie pas de résumer (risque hallucination). On concatène des extraits du début d’unité.
    """
    if not UNIT_ANCHOR_ENABLE:
        return None

    head = []
    for _, txt in (unit_pages[: max(0, UNIT_ANCHOR_HEAD_PAGES)] if unit_pages else []):
        if txt:
            head.append(txt)
    head_txt = _clean_ws("\n".join(head))
    if not head_txt:
        return None

    if len(head_txt) > UNIT_ANCHOR_MAX_CHARS:
        head_txt = head_txt[:UNIT_ANCHOR_MAX_CHARS].rstrip()

    header = [
        "ACTE / UNITÉ (ancre de retrieval — extrait source, non résumé)",
        f"SOURCE: {source_name}",
        f"DOC_TYPE: {doc_type or ''}",
        f"DOC_DATE: {doc_date or ''}",
        f"UNIT_TITLE: {unit_title or ''}",
        f"UNIT_TYPE: {unit_type or ''}",
        f"PAGES_UNIT: {page_start or ''}-{page_end or ''}",
        "",
        head_txt,
    ]
    return "\n".join(header).strip()


# ---------------------------------------------------------------------------
# Helpers progression
# ---------------------------------------------------------------------------

def _safe_cb(cb: ProgressCB, payload: Dict[str, Any]) -> None:
    if cb is None:
        return
    try:
        cb(payload)
    except Exception:
        logger.exception("progress callback failed: %r", payload)

def _push_progress(
    cb: ProgressCB,
    *,
    percent: Optional[float] = None,
    label: Optional[str] = None,
    ocr_pages: Optional[int] = None,
    stage: Optional[str] = None,
) -> None:
    """Envoie un évènement de progression SSE (progress + phase) de manière tolérante."""
    if cb is None:
        return

    if percent is not None:
        payload: Dict[str, Any] = {"type": "progress", "percent": float(percent)}
        if ocr_pages is not None:
            payload["ocr_pages"] = int(ocr_pages)
        if stage:
            payload["stage"] = stage
        _safe_cb(cb, payload)

    if label is not None:
        payload2: Dict[str, Any] = {"type": "phase", "label": label}
        if stage:
            payload2["stage"] = stage
        _safe_cb(cb, payload2)


# ---------------------------------------------------------------------------
# Helpers embeddings
# ---------------------------------------------------------------------------

def _model_name_in_use(model: SentenceTransformer) -> str:
    return str(getattr(model, "model_name_or_path", "") or "")

def _format_passages_for_model(texts: List[str], model_name: str) -> List[str]:
    """
    Certains modèles (ex: e5) préfèrent que chaque texte soit préfixé par "passage:".
    Applique ce préfixe quand le nom du modèle contient "e5".
    """
    if "e5" in (model_name or "").lower():
        return [f"passage: {t}" for t in texts]
    return texts

def _encode_with_progress(
    model: SentenceTransformer,
    texts: List[str],
    *,
    batch_size: int = 64,
    progress_cb: ProgressCB = None,
    ocr_pages: Optional[int] = None,
    file_idx: int = 0,
    total_files: int = 1,
    file_name: Optional[str] = None,
) -> np.ndarray:
    total = len(texts)
    if total == 0:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype="float32")

    total_files = max(total_files, 1)
    base = (file_idx / total_files) * 100.0
    end = ((file_idx + 1) / total_files) * 100.0
    span = max(5.0, end - base)

    all_vecs: List[np.ndarray] = []
    real_model_name = _model_name_in_use(model) or EMBEDDING_MODEL

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_for_model = _format_passages_for_model(batch, model_name=real_model_name)

        vecs = model.encode(
            batch_for_model,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_vecs.append(vecs)

        done = i + len(batch)
        frac = done / total
        pct = base + span * frac

        _push_progress(
            progress_cb,
            percent=pct,
            label=f"Embeddings {file_name or ''} : {done}/{total} blocs…",
            ocr_pages=ocr_pages,
            stage="embeddings",
        )

    return np.vstack(all_vecs)


# ---------------------------------------------------------------------------
# Helpers texte
# ---------------------------------------------------------------------------

def _clean_ws(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s

def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = _clean_ws(text)
    parts = re.split(r"([.!?])", text)
    sents: List[str] = []
    buf: List[str] = []
    for part in parts:
        if not part:
            continue
        buf.append(part)
        if part in (".", "!", "?"):
            sent = "".join(buf).strip()
            if sent:
                sents.append(sent)
            buf = []
    if buf:
        sent = "".join(buf).strip()
        if sent:
            sents.append(sent)
    return sents

def _chunk_from_sentences(sents: List[str], size: int, overlap: int) -> List[str]:
    if not sents:
        return []
    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for sent in sents:
        l = len(sent)
        if cur_len + l > size and cur:
            chunks.append(" ".join(cur).strip())
            if overlap > 0:
                overlap_text = " ".join(cur)[-overlap:].split()
                cur = [" ".join(overlap_text)] if overlap_text else []
                cur_len = len(" ".join(cur))
            else:
                cur = []
                cur_len = 0
        cur.append(sent)
        cur_len += l + 1
    if cur:
        chunks.append(" ".join(cur).strip())
    return chunks

def _ocr_page_text(page, dpi: int = OCR_DPI, lang: str = OCR_LANG) -> str:
    try:
        import importlib
        pytesseract = importlib.import_module("pytesseract")
        Image = importlib.import_module("PIL.Image")
    except Exception:
        logger.warning("OCR demandé mais pytesseract ou Pillow (PIL) n’est pas installé")
        return ""
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    text = pytesseract.image_to_string(img, lang=lang or "eng")
    return text or ""

def _load_sentence_model() -> SentenceTransformer:
    global _sentence_model
    if _sentence_model is not None:
        return _sentence_model

    try:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _sentence_model = SentenceTransformer(EMBEDDING_MODEL)
        return _sentence_model
    except Exception:
        logger.exception("Failed to load embedding model '%s'.", EMBEDDING_MODEL)

    if EMBEDDING_MODEL != DEFAULT_EMBED_MODEL:
        try:
            logger.info("Falling back to default embedding model: %s", DEFAULT_EMBED_MODEL)
            _sentence_model = SentenceTransformer(DEFAULT_EMBED_MODEL)
            return _sentence_model
        except Exception:
            logger.exception("Failed to load default embedding model '%s'.", DEFAULT_EMBED_MODEL)

    raise RuntimeError(
        f"Impossible de charger un modèle d'embedding. "
        f"EMBEDDING_MODEL='{EMBEDDING_MODEL}', DEFAULT='{DEFAULT_EMBED_MODEL}'."
    )

def _ensure_case_dirs(case_id: str) -> Dict[str, Path]:
    casedir = STORAGE / "cases" / case_id
    rawdir = casedir / "raw"
    idxdir = casedir
    casedir.mkdir(parents=True, exist_ok=True)
    rawdir.mkdir(parents=True, exist_ok=True)
    return {"path": casedir, "raw": rawdir, "idx": idxdir}

def _load_case_meta(case_id: str) -> Dict:
    casedir = STORAGE / "cases" / case_id
    meta_path = casedir / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("failed to load meta.json for case %s", case_id)
            return {"created": time.time(), "n_chunks": 0, "docs": []}
    return {"created": time.time(), "n_chunks": 0, "docs": []}

def _save_case_meta(case_id: str, meta: Dict) -> None:
    casedir = STORAGE / "cases" / case_id
    meta_path = casedir / "meta.json"
    tmp = casedir / f".meta.{time.time()}.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(meta_path))


def ingest_files(
    case_id: str,
    file_paths: List[Path],
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> Dict:
    cb = progress_cb or (lambda x: None)
    t_start_all = perf_counter()

    info = _ensure_case_dirs(case_id)
    casedir = Path(info["path"])
    rawdir = casedir / "raw"
    meta_path = casedir / "meta.json"
    bm25_path = casedir / "bm25.json"
    faiss_path = casedir / "faiss.index"
    vecs_path = casedir / "vectors.npy"

    meta = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.exists()
        else {"created": time.time(), "n_chunks": 0, "docs": []}
    )
    existing_hashes = {d["hash"] for d in meta.get("docs", []) if "hash" in d}

    # corpus existant
    if bm25_path.exists():
        bm = json.loads(bm25_path.read_text(encoding="utf-8"))
        all_chunks = bm.get("chunks", [])
        all_sources = bm.get("sources", [])
    else:
        all_chunks = []
        all_sources = []

    # modèle d'embedding
    embed_model = _load_sentence_model()
    embedding_dim = embed_model.get_sentence_embedding_dimension()

    if EMBEDDING_DIM_ENV:
        try:
            if int(EMBEDDING_DIM_ENV) != embedding_dim:
                logger.warning(
                    "EMBEDDING_DIM dans .env (%s) ≠ dimension réelle du modèle (%s). On utilise %s.",
                    EMBEDDING_DIM_ENV,
                    embedding_dim,
                    embedding_dim,
                )
        except Exception:
            logger.warning("EMBEDDING_DIM invalide (%r), ignoré.", EMBEDDING_DIM_ENV)

    # faiss / vecs existants
    if faiss_path.exists() and vecs_path.exists():
        try:
            index = faiss.read_index(str(faiss_path))
            vecs = np.load(str(vecs_path))
            if index.d != embedding_dim or (
                vecs is not None and vecs.ndim == 2 and vecs.shape[1] != embedding_dim
            ):
                logger.warning(
                    "Dimension FAISS (%s) ou vectors.npy (%s) ≠ embedding_dim (%s). On recrée les index.",
                    index.d,
                    vecs.shape[1] if (vecs is not None and vecs.ndim == 2) else "?",
                    embedding_dim,
                )
                index = faiss.IndexFlatIP(embedding_dim)
                vecs = None
        except Exception:
            logger.exception("impossible de relire l'index FAISS existant, on repart à zéro")
            index = faiss.IndexFlatIP(embedding_dim)
            vecs = None
    else:
        index = faiss.IndexFlatIP(embedding_dim)
        vecs = None

    ocr_pages = 0
    total_files = len(file_paths)
    cb({"type": "start", "total_files": total_files})
    _push_progress(progress_cb, percent=0.0, label="Analyse des fichiers…", stage="start")

    for file_idx, fp in enumerate(file_paths):
        fp = Path(fp)
        if not fp.exists():
            logger.warning("ingest: fichier %s introuvable, on saute", fp)
            continue

        _push_progress(progress_cb, label=f"Lecture du fichier {fp.name}…", stage="read")

        h = _hash_file(fp)
        if h in existing_hashes:
            logger.info("ingest: file %s already in case %s, skipping", fp.name, case_id)
            percent = int(((file_idx + 1) / max(total_files, 1)) * 100)
            cb({"type": "progress", "current": file_idx + 1, "total": total_files, "percent": percent})
            continue

        # copier le fichier brut
        raw_target = rawdir / fp.name
        shutil.copy2(str(fp), str(raw_target))

        suffix = fp.suffix.lower()
        pages: List[Tuple[int, str]] = []

        if suffix == ".pdf":
            doc = fitz.open(str(fp))
            for i, page in enumerate(doc):
                t = _clean_ws(page.get_text("text") or "")
                need_ocr = OCR_ENABLE and (len(t) < OCR_MIN_CHARS or OCR_FORCE)
                if need_ocr:
                    try:
                        t_ocr = _ocr_page_text(page, dpi=OCR_DPI, lang=OCR_LANG)
                        if len(t_ocr) >= len(t):
                            t = _clean_ws(t_ocr)
                            ocr_pages += 1
                    except Exception:
                        logger.exception("OCR failed on page %s of %s", i + 1, fp)
                pages.append((i + 1, t))
            doc.close()

        elif suffix in (".txt", ".csv", ".log"):
            text = fp.read_text(encoding="utf-8", errors="ignore")
            pages = [(1, _clean_ws(text))]

        elif suffix == ".docx":
            try:
                import importlib
                Document = importlib.import_module("docx").Document
                docx_doc = Document(str(fp))
                text = "\n".join(p.text for p in docx_doc.paragraphs)
                pages = [(1, _clean_ws(text))]
            except Exception:
                text = fp.read_text(encoding="utf-8", errors="ignore")
                pages = [(1, _clean_ws(text))]
        else:
            text = fp.read_text(encoding="utf-8", errors="ignore")
            pages = [(1, _clean_ws(text))]

        # métadonnées doc globales
        text_first_pages = " \n".join(p[1] for p in pages[:2]) if pages else ""
        doc_date = detect_doc_date(text_first_pages, fp.name)
        doc_type = detect_doc_type(text_first_pages, fp.name)

        _push_progress(progress_cb, label=f"Découpage en unités (PV/actes) pour {fp.name}…", stage="chunk")

        # découpage en unités logiques (PV/rapports/actes)
        units = split_pages_into_units(pages, fp.name)

        doc_chunks: List[Tuple[str, Dict]] = []
        base_chunk_id = int(meta.get("n_chunks", 0))

        for unit_index, unit in enumerate(units):
            unit_title = unit.get("title") or fp.name
            unit_pages: List[Tuple[int, str]] = unit.get("pages", []) or []

            page_start = unit_pages[0][0] if unit_pages else None
            page_end = unit_pages[-1][0] if unit_pages else None

            unit_type = _infer_unit_type(unit, doc_type_fallback=doc_type)
            uid = _unit_id(fp.name, unit_index, page_start, page_end, unit_title)

            # ✅ 1) UNIT HEADER CHUNK (ancre acte) — améliore la couverture “PV oubliés”
            anchor = _make_unit_anchor_text(
                source_name=fp.name,
                doc_type=doc_type,
                doc_date=doc_date,
                unit_title=unit_title,
                unit_type=unit_type,
                page_start=page_start,
                page_end=page_end,
                unit_pages=unit_pages,
            )
            if anchor:
                src_anchor = {
                    "source": fp.name,
                    "page": page_start or 1,
                    "chunk_id": base_chunk_id + len(doc_chunks),
                    "local_chunk_id": -2,  # réservé
                    "date": doc_date,
                    "doc_type": doc_type,
                    "unit_index": unit_index,
                    "unit_id": uid,
                    "unit_title": unit_title,
                    "unit_type": unit_type,
                    "page_start": page_start,
                    "page_end": page_end,
                    "is_unit_anchor": True,
                }
                doc_chunks.append((anchor, src_anchor))

            # ✅ 2) FACT CHUNK identité : 1 par unité identité (si on arrive à extraire)
            if _unit_looks_like_identity(unit_title, doc_type, unit_pages):
                fields_acc: Dict[str, str] = {}
                blob_parts: List[str] = []

                for _, txt in unit_pages[:5]:
                    blob_parts.append(txt or "")
                    blob = "\n".join(blob_parts)

                    fields_now = _extract_identity_fields_robust(blob)
                    for k, v in fields_now.items():
                        if k not in fields_acc and v:
                            fields_acc[k] = v

                    if _fields_are_good_enough(fields_acc, doc_type):
                        break

                fact = _make_identity_fact_chunk(fields_acc)
                if fact:
                    src_fact = {
                        "source": fp.name,
                        "page": page_start or 1,
                        "chunk_id": base_chunk_id + len(doc_chunks),
                        "local_chunk_id": -1,
                        "date": doc_date,
                        "doc_type": "pv_identite" if (doc_type != "pv_identite") else doc_type,
                        "unit_index": unit_index,
                        "unit_id": uid,
                        "unit_title": f"{unit_title} — IDENTITÉ (facts)",
                        "unit_type": "pv_identite",
                        "page_start": page_start,
                        "page_end": page_end,
                        "is_fact": True,
                    }
                    doc_chunks.append((fact, src_fact))

            # ✅ 3) Chunking texte normal (par page → citation page fiable)
            for page_num, page_text in unit_pages:
                if not page_text:
                    continue
                sents = _split_sentences(page_text)
                chunks = _chunk_from_sentences(sents, CHUNK_SIZE, CHUNK_OVERLAP)
                for local_id, ch in enumerate(chunks):
                    src = {
                        "source": fp.name,
                        "page": page_num,
                        "chunk_id": base_chunk_id + len(doc_chunks),
                        "local_chunk_id": local_id,
                        "date": doc_date,
                        "doc_type": doc_type,
                        "unit_index": unit_index,
                        "unit_id": uid,
                        "unit_title": unit_title,
                        "unit_type": unit_type,
                        "page_start": page_start,
                        "page_end": page_end,
                    }
                    doc_chunks.append((ch, src))

        # embeddings + ajout corpus
        if doc_chunks:
            texts = [c[0] for c in doc_chunks]
            vec = _encode_with_progress(
                embed_model,
                texts,
                batch_size=64,
                progress_cb=progress_cb,
                ocr_pages=ocr_pages,
                file_idx=file_idx,
                total_files=total_files,
                file_name=fp.name,
            )

            if vec.ndim == 1:
                vec = vec.reshape(1, -1)

            if vec.shape[1] != embedding_dim:
                logger.error(
                    "Les embeddings produits font %s dims au lieu de %s",
                    vec.shape[1],
                    embedding_dim,
                )
            else:
                if vecs is None:
                    vecs = vec
                else:
                    vecs = np.vstack([vecs, vec])
                index.add(vec)

            for ch, src in doc_chunks:
                all_chunks.append(ch)
                all_sources.append(src)

            meta["docs"].append(
                {
                    "name": fp.name,
                    "hash": h,
                    "pages": len(pages),
                    "chunks": len(doc_chunks),
                    "added": time.time(),
                    "date": doc_date,
                    "doc_type": doc_type,
                    "units": [
                        {
                            "index": i,
                            "id": _unit_id(
                                fp.name,
                                i,
                                (u.get("pages") or [(None, "")])[0][0] if u.get("pages") else None,
                                (u.get("pages") or [(None, "")])[-1][0] if u.get("pages") else None,
                                u.get("title") or "",
                            ),
                            "title": u.get("title"),
                            "type": _infer_unit_type(u, doc_type_fallback=doc_type),
                            "page_start": u["pages"][0][0] if u.get("pages") else None,
                            "page_end": u["pages"][-1][0] if u.get("pages") else None,
                        }
                        for i, u in enumerate(units)
                    ],
                }
            )
            meta["n_chunks"] = len(all_chunks)

            cb({"type": "file", "file": fp.name, "chunks": len(doc_chunks)})

        percent = int(((file_idx + 1) / max(total_files, 1)) * 100)
        cb(
            {
                "type": "progress",
                "current": file_idx + 1,
                "total": total_files,
                "percent": percent,
                "file": fp.name,
            }
        )

    # ====== écriture des indexes ======
    _push_progress(progress_cb, label="Construction de l’index texte (BM25)…", stage="bm25")

    tokens = [c.split() for c in all_chunks]
    if tokens:
        _ = BM25Okapi(tokens)
        bm = {"chunks": all_chunks, "sources": all_sources, "tokens": tokens}
        tmp_bm25 = casedir / f".bm25.{time.time()}.tmp"
        tmp_bm25.write_text(json.dumps(bm, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp_bm25), str(bm25_path))
    else:
        logger.warning("ingest: aucun chunk extrait pour %s, index BM25 non généré", case_id)

    _push_progress(progress_cb, label="Indexation vectorielle (FAISS)…", stage="faiss")

    if index is not None and vecs is not None:
        tmp_vecs = casedir / f".vectors.{time.time()}.npy"
        np.save(str(tmp_vecs), vecs)
        os.replace(str(tmp_vecs), str(vecs_path))

        tmp_faiss = casedir / f".faiss.{time.time()}.tmp"
        faiss.write_index(index, str(tmp_faiss))
        os.replace(str(tmp_faiss), str(faiss_path))

    tmp_meta = casedir / f".meta.{time.time()}.tmp"
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_meta), str(meta_path))

    elapsed = perf_counter() - t_start_all
    logger.info("ingest: wrote indexes in %.2fs", elapsed)

    _push_progress(progress_cb, percent=100.0, label="Ingestion terminée ✅", ocr_pages=ocr_pages, stage="done")

    cb(
        {
            "type": "done",
            "case_id": case_id,
            "n_docs": len(meta["docs"]),
            "n_chunks": len(all_chunks),
            "ocr_pages": ocr_pages,
            "elapsed": elapsed,
        }
    )

    return {
        "case_id": case_id,
        "n_docs": len(meta["docs"]),
        "n_chunks": len(all_chunks),
        "ocr_pages": ocr_pages,
    }


# ---------------------------------------------------------------------------
# Chat sur fichier déposé (hors affaire)
# ---------------------------------------------------------------------------
def chunks_from_bytes(data: bytes, filename: str) -> List[Tuple[str, Dict]]:
    suffix = Path(filename).suffix.lower()
    pages: List[Tuple[int, str]] = []

    if suffix == ".pdf":
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            for i, page in enumerate(doc):
                t = _clean_ws(page.get_text("text") or "")
                need_ocr = OCR_ENABLE and (len(t) < OCR_MIN_CHARS or OCR_FORCE)
                if need_ocr:
                    try:
                        t_ocr = _ocr_page_text(page, dpi=OCR_DPI, lang=OCR_LANG)
                        if len(t_ocr) >= len(t):
                            t = _clean_ws(t_ocr)
                    except Exception:
                        logger.exception("OCR échoué sur upload (page %s)", i + 1)
                pages.append((i + 1, t))
        finally:
            doc.close()

    elif suffix in (".txt", ".csv", ".log"):
        text = data.decode("utf-8", errors="ignore")
        pages = [(1, _clean_ws(text))]

    elif suffix == ".docx":
        try:
            import importlib
            Document = importlib.import_module("docx").Document
            f = io.BytesIO(data)
            docx_doc = Document(f)
            text = "\n".join(p.text for p in docx_doc.paragraphs)
            pages = [(1, _clean_ws(text))]
        except Exception:
            text = data.decode("utf-8", errors="ignore")
            pages = [(1, _clean_ws(text))]
    else:
        text = data.decode("utf-8", errors="ignore")
        pages = [(1, _clean_ws(text))]

    results: List[Tuple[str, Dict]] = []
    global_chunk_id = 0

    text_first_pages = " \n".join(p[1] for p in pages[:2]) if pages else ""
    doc_date = detect_doc_date(text_first_pages, filename)
    doc_type = detect_doc_type(text_first_pages, filename)

    units = split_pages_into_units(pages, filename)

    for unit_index, unit in enumerate(units):
        unit_title = unit.get("title") or filename
        unit_pages: List[Tuple[int, str]] = unit.get("pages", []) or []

        page_start = unit_pages[0][0] if unit_pages else None
        page_end = unit_pages[-1][0] if unit_pages else None
        unit_type = _infer_unit_type(unit, doc_type_fallback=doc_type)
        uid = _unit_id(filename, unit_index, page_start, page_end, unit_title)

        # ✅ UNIT HEADER CHUNK upload
        anchor = _make_unit_anchor_text(
            source_name=filename,
            doc_type=doc_type,
            doc_date=doc_date,
            unit_title=unit_title,
            unit_type=unit_type,
            page_start=page_start,
            page_end=page_end,
            unit_pages=unit_pages,
        )
        if anchor:
            meta_anchor = {
                "source": filename,
                "page": page_start or 1,
                "chunk_id": global_chunk_id,
                "local_chunk_id": -2,
                "date": doc_date,
                "doc_type": doc_type,
                "unit_index": unit_index,
                "unit_id": uid,
                "unit_title": unit_title,
                "unit_type": unit_type,
                "page_start": page_start,
                "page_end": page_end,
                "is_unit_anchor": True,
            }
            results.append((anchor, meta_anchor))
            global_chunk_id += 1

        # ✅ FACT chunk identité hors affaire aussi
        if _unit_looks_like_identity(unit_title, doc_type, unit_pages):
            fields_acc: Dict[str, str] = {}
            blob_parts: List[str] = []

            for _, txt in unit_pages[:5]:
                blob_parts.append(txt or "")
                blob = "\n".join(blob_parts)

                fields_now = _extract_identity_fields_robust(blob)
                for k, v in fields_now.items():
                    if k not in fields_acc and v:
                        fields_acc[k] = v

                if _fields_are_good_enough(fields_acc, doc_type):
                    break

            fact = _make_identity_fact_chunk(fields_acc)
            if fact:
                meta_fact = {
                    "source": filename,
                    "page": page_start or 1,
                    "chunk_id": global_chunk_id,
                    "local_chunk_id": -1,
                    "date": doc_date,
                    "doc_type": "pv_identite" if (doc_type != "pv_identite") else doc_type,
                    "unit_index": unit_index,
                    "unit_id": uid,
                    "unit_title": f"{unit_title} — IDENTITÉ (facts)",
                    "unit_type": "pv_identite",
                    "page_start": page_start,
                    "page_end": page_end,
                    "is_fact": True,
                }
                results.append((fact, meta_fact))
                global_chunk_id += 1

        for page_num, page_text in unit_pages:
            if not page_text:
                continue
            sents = _split_sentences(page_text)
            chunks = _chunk_from_sentences(sents, CHUNK_SIZE, CHUNK_OVERLAP)
            for local_id, ck in enumerate(chunks):
                meta = {
                    "source": filename,
                    "page": page_num,
                    "chunk_id": global_chunk_id,
                    "local_chunk_id": local_id,
                    "date": doc_date,
                    "doc_type": doc_type,
                    "unit_index": unit_index,
                    "unit_id": uid,
                    "unit_title": unit_title,
                    "unit_type": unit_type,
                    "page_start": page_start,
                    "page_end": page_end,
                }
                results.append((ck, meta))
                global_chunk_id += 1

    return results


# ---------------------------------------------------------------------------
# API PUBLIQUES
# ---------------------------------------------------------------------------
def ensure_case(case_id: str) -> dict:
    info = _ensure_case_dirs(case_id)
    casedir: Path = info["path"]
    meta_path = casedir / "meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("ensure_case: meta.json corrompu pour %s, on le régénère", case_id)
    meta = {"created": time.time(), "n_chunks": 0, "docs": []}
    _save_case_meta(case_id, meta)
    return meta

def list_docs(case_id: str) -> List[Dict]:
    meta = _load_case_meta(case_id)
    return meta.get("docs", [])

def list_cases() -> List[str]:
    cases_dir = STORAGE / "cases"
    if not cases_dir.exists():
        return []
    out: List[str] = []
    for entry in cases_dir.iterdir():
        if entry.is_dir():
            out.append(entry.name)
    out.sort()
    return out
