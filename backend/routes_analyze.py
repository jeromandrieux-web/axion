# backend/routes_analyze.py
# Routes d'analyse procédurale par personne
# Auteur: Jax

from __future__ import annotations
import os, re, json, unicodedata
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

router = APIRouter()

STORAGE = Path(os.getenv("STORAGE_DIR", "backend/storage")).resolve()
ANALYZE_TOP_K = int(os.getenv("ANALYZE_TOP_K", "80"))
ANALYZE_MAX_ROWS = int(os.getenv("ANALYZE_MAX_ROWS", "40"))
ANALYZE_MAX_TIMELINE = int(os.getenv("ANALYZE_MAX_TIMELINE", "120"))

# -------------------------
# Utils
# -------------------------

def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _load_case_chunks(case_id: str) -> Tuple[List[str], List[Dict]]:
    casedir = STORAGE / "cases" / case_id
    data = json.loads((casedir / "bm25.json").read_text(encoding="utf-8"))
    return data["chunks"], data["sources"]

def _fallback_retrieve(case_id: str, top_k: int) -> List[Dict]:
    chunks, sources = _load_case_chunks(case_id)
    hits = []
    for i, t in enumerate(chunks[:top_k]):
        hits.append({
            "text": t,
            "meta": sources[i] if i < len(sources) else {}
        })
    return hits

def _split_sentences(text: str) -> List[str]:
    # Découpage simple par ponctuation forte
    parts = re.split(r'(?<=[\.\!\?])\s+', text)
    # Ajoute lignes sans ponctuation finale (e.g. puces)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # coupe les très longues lignes en segments ~300 chars
        while len(p) > 300:
            cut = p.rfind(' ', 0, 300)
            if cut < 120:
                break
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out

# Dates (FR & ISO)
MONTHS_FR = r"(janv(?:ier)?|févr(?:ier)?|mars|avril|mai|juin|juil(?:let)?|août|sept(?:embre)?|oct(?:obre)?|nov(?:embre)?|déc(?:embre)?)"
DATE_PATTERNS = [
    re.compile(rf"\b(\d{{1,2}}\s+{MONTHS_FR}\s+\d{{4}})\b", re.IGNORECASE),   # 12 mars 2024
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),                                   # 2024-03-12
    re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b"),                   # 12/03/2024 ou 12-03-24
]
def _extract_date(s: str) -> Optional[str]:
    for rx in DATE_PATTERNS:
        m = rx.search(s)
        if m:
            return m.group(1)
    return None

# Indicateurs charge / décharge
INDICATORS_CHARGE = {
    'reconnu','aveu','fraude','compromettant','trace','localisation',
    'appel','virement','preuve','impliqué','implication','en relation',
    'utilisé','titulaire','au nom de','contacté','correspondant'
}
INDICATORS_DECHARGE = {
    'alibi','démenti','erreur','absence','aucun lien','non impliqué','décorrélé',
    'confusion','homonyme','autorisé','légal'
}

def _classify_sentence(s: str) -> str:
    low = s.lower()
    if any(w in low for w in INDICATORS_CHARGE):
        return 'charge'
    if any(w in low for w in INDICATORS_DECHARGE):
        return 'decharge'
    return 'neutre'

# -------------------------
# RAG retrieve
# -------------------------
def _retrieve(case_id: str, query: str, top_k: int) -> List[Dict]:
    # Essaie d'utiliser le retriever intégré si présent
    try:
        from .rag import retrieve  # type: ignore
        return (retrieve(case_id=case_id, query=query, top_k=top_k))  # peut être sync ou async selon ton impl
    except Exception:
        return _fallback_retrieve(case_id, top_k)

def _ensure_hits(hits) -> List[Dict]:
    # Uniformise {text, meta:{source,page,chunk_id}}
    out = []
    for h in hits or []:
        if isinstance(h, dict) and "text" in h:
            meta = h.get("meta") or {}
            out.append({"text": h["text"], "meta": meta})
    return out

# -------------------------
# Endpoint
# -------------------------
@router.get("/api/analyze")
async def analyze(case_id: str = Query(...), person: str = Query(...), aliases: Optional[str] = Query(None)):
    """
    Retour:
      {
        person: str,
        summary_table: [ {label, status, refs:[{source,page,theme,item}]} ],
        timeline_faits: [ {date, event, kind, source} ]
      }
    """
    # Prépare la requête RAG
    query = f"chronologie faits matériels éléments à charge à décharge {person}"
    # aliases ? "Jean Dupont|J. Dupont"
    aliases_list = []
    if aliases:
        aliases_list = [a.strip() for a in aliases.split("|") if a.strip()]

    # Récupération
    hits = _retrieve(case_id, query, ANALYZE_TOP_K)
    hits = _ensure_hits(hits)

    # Agrégation
    timeline: List[Dict] = []
    table: List[Dict] = []

    # Normalise le pattern de personne (person + alias) pour filtrer les phrases
    person_terms = [person] + aliases_list
    person_terms_norm = [ _norm(p).lower() for p in person_terms if p ]
    def _mentions_person(sentence: str) -> bool:
        s = _norm(sentence).lower()
        return any(t and t in s for t in person_terms_norm) if person_terms_norm else True

    # Dédup sur libellé normalisé
    seen_labels = set()

    for h in hits:
        txt = h.get("text") or ""
        meta = h.get("meta") or {}
        src = meta.get("source") or "?"
        page = meta.get("page")
        src_label = f"{src}{(' p.'+str(page)) if page is not None else ''}"

        for sent in _split_sentences(txt):
            if not _mentions_person(sent):
                continue

            # Timeline
            d = _extract_date(sent)
            if d:
                kind = _classify_sentence(sent)
                timeline.append({
                    "date": d,
                    "event": sent.strip(),
                    "kind": kind,
                    "source": src_label
                })

            # Tableau (1 ligne par phrase pertinente, dédupliquée)
            label = sent.strip()
            norm_key = _norm(label).lower()
            if not norm_key or norm_key in seen_labels:
                continue
            seen_labels.add(norm_key)

            status = _classify_sentence(sent)
            status_lbl = 'à charge' if status == 'charge' else ('à décharge' if status == 'decharge' else 'neutre')

            row = {
                "label": label[:220] + ("…" if len(label) > 220 else ""),
                "status": status_lbl,
                "refs": [{
                    "source": src,
                    "page": page,
                    "theme": "faits",
                    "item": person
                }]
            }
            table.append(row)

    # Nettoyage / tri
    # - timeline: garde l'ordre d'apparition, dédup (date+event)
    seen_t = set()
    tl_uniq = []
    for ev in timeline:
        k = (ev["date"], _norm(ev["event"]).lower())
        if k in seen_t:
            continue
        seen_t.add(k)
        tl_uniq.append(ev)

    # Coupe pour l'UI
    table = table[:ANALYZE_MAX_ROWS]
    tl_uniq = tl_uniq[:ANALYZE_MAX_TIMELINE]

    return JSONResponse({
        "person": person,
        "summary_table": table,
        "timeline_faits": tl_uniq
    })
