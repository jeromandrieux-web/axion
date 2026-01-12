# backend/extractors.py
# Extraction thématique avancée dans les dossiers: téléphones, IBAN, emails, IPs, plaques, réseaux sociaux.
# Auteur: Jax

import os
import re
import json
import unicodedata
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# import utils_normalize — supporte import relatif (package) et non-relatif (exécution directe)
try:
    from .utils_normalize import (
        PHONE_RE, IBAN_RE, EMAIL_RE, IP_RE, PLATE_RE,
        normalize_phone, normalize_iban, normalize_email, normalize_plate,
        window_contains_person, extract_with_regex, stable_dedup
    )
except Exception:
    from utils_normalize import (
        PHONE_RE, IBAN_RE, EMAIL_RE, IP_RE, PLATE_RE,
        normalize_phone, normalize_iban, normalize_email, normalize_plate,
        window_contains_person, extract_with_regex, stable_dedup
    )

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

STORAGE = Path(os.getenv("STORAGE_DIR", "backend/storage")).resolve()

# Paramètres d'association personne <-> item
STRICT_THEME_LINK = os.getenv("STRICT_THEME_LINK", "0") == "1"
PERSON_WINDOW_CHARS = int(os.getenv("PERSON_WINDOW_CHARS", "80"))

# Renforcement plaques (conservé si tu veux garder un contrôle .env)
STRICT_PLATES = os.getenv("STRICT_PLATES", "1") == "1"

SUPPORTED_THEMES = {
    "phones", "iban", "emails", "plates", "plaques", "ips", "telegram", "whatsapp", "snapchat"
}

logger = logging.getLogger(__name__)

def _norm(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0"," ")
    s = s.lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = re.sub(r'\s+',' ', s).strip()
    return s

def _load_case_chunks(case_id: str) -> Tuple[List[str], List[Dict]]:
    casedir = STORAGE / "cases" / case_id
    bm25_path = casedir / "bm25.json"
    if not bm25_path.exists():
        logger.error("bm25.json introuvable pour le dossier %s (%s)", case_id, bm25_path)
        raise FileNotFoundError(f"bm25.json absent pour le dossier {case_id}")
    try:
        data = json.loads(bm25_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.exception("Erreur lecture/parsing bm25.json pour %s", case_id)
        raise
    chunks = data.get("chunks", []) or []
    sources = data.get("sources", []) or []
    return chunks, sources

def _mk_snippet(text: str, span: Tuple[int,int], max_len: int = 280, ctx: int = 120) -> str:
    """Construit un extrait autour du match (… gauche [match] droite …)."""
    if not text:
        return ""
    # import local pour éviter l'import circulaire au chargement du module
    from .ingest import _clean_ws
    a = max(0, span[0] - ctx)
    b = min(len(text), span[1] + ctx)
    snippet = text[a:b].strip()
    snippet = _clean_ws(snippet)
    return snippet[:max_len] + ("…" if len(snippet) > max_len else "")

def _occurrence(meta: Dict, snippet: str, linked: bool) -> Dict:
    return {
        "source": meta.get("source"),
        "page": meta.get("page"),
        "chunk_id": meta.get("chunk_id"),
        "snippet": snippet,
        "linked": bool(linked)
    }

# =======================
# 1) Téléphones — classification "utilisé" vs "en relation"
# =======================
# Seuils/poids (modifiable en .env si besoin)
USED_STRONG_W   = float(os.getenv("PHONE_USED_STRONG_W", 0.8))
USED_MED_W      = float(os.getenv("PHONE_USED_MED_W", 0.6))
CONTACT_W       = float(os.getenv("PHONE_CONTACT_W", 0.6))
WEAK_W          = float(os.getenv("PHONE_WEAK_W", 0.35))
UNLINKED_FACTOR = float(os.getenv("PHONE_UNLINKED_FACTOR", 0.5))
UNKNOWN_MIN     = float(os.getenv("PHONE_UNKNOWN_MIN", 0.30))
CONF_CAP        = float(os.getenv("PHONE_CONF_CAP", 0.95))

# Motifs (sur texte normalisé via _norm)
USED_PATTERNS_STRONG = [
    r'\bnumero(?: de(?: telephone| tel)?)?\s+de\s+{P}\b',
    r'\bligne\s+(?:de|au nom de)\s+{P}\b',
    r'\babonnement\s+(?:de|au nom de)\s+{P}\b',
    r'\battribuee?\s+a\s+{P}\b',
    r'\btitulaire\s*:?\s+{P}\b',
]
USED_PATTERNS_MED = [
    r'\b{P}\s+(?:utilise|utilisait|joignable au|contactable au)\b',
    r'\bau nom de\s+{P}\b',
    r'\bmsisdn\b.*\b{P}\b',
    r'\bimei\b.*\b{P}\b',
    r'\bimsi\b.*\b{P}\b',
]
CONTACT_PATTERNS = [
    r'\bappele(?:e)?\s+par\s+{P}\b',
    r'\bcontact\s+(?:de|avec)\s+{P}\b',
    r'\bcorrespondant\s+de\s+{P}\b',
    r'\bcommunication\s+avec\s+{P}\b',
    r'\bconversation\s+avec\s+{P}\b',
]
WEAK_CONTEXT = [
    r'\bappels?\b', r'\bappelant\b', r'\bappele\b', r'\bcontact\b', r'\bconversation\b'
]

def _score_occurrence_role(snippet: str, person_terms_norm: List[str], linked: bool) -> Tuple[float, float, List[str]]:
    """Retourne (used_score, contact_score, rules_hit) pour UNE occurrence."""
    s = _norm(snippet)
    rules: List[str] = []
    used, contact = 0.0, 0.0

    if person_terms_norm:
        P = '(?:' + '|'.join(re.escape(p) for p in person_terms_norm if p) + ')'
    else:
        P = r'.+?'

    def add_if(patterns: List[str], weight: float, bucket: str):
        nonlocal used, contact, rules
        for pat in patterns:
            rx = re.compile(pat.format(P=P))
            if rx.search(s):
                w = weight * (UNLINKED_FACTOR if not linked else 1.0)
                if bucket == 'used':
                    used += w; rules.append(f'used:{pat}')
                else:
                    contact += w; rules.append(f'contact:{pat}')

    add_if(USED_PATTERNS_STRONG, USED_STRONG_W, 'used')
    add_if(USED_PATTERNS_MED,    USED_MED_W,    'used')
    add_if(CONTACT_PATTERNS,     CONTACT_W,     'contact')

    if used == 0 and contact == 0:
        if any(re.search(p, s) for p in WEAK_CONTEXT):
            w = WEAK_W * (UNLINKED_FACTOR if not linked else 1.0)
            used += w * 0.5
            contact += w * 0.5
            rules.append('weak:context')

    return used, contact, rules

def _aggregate_phone_role(occurrences: List[Dict], person_terms_norm: List[str]) -> Tuple[str, float, List[Dict]]:
    """Agrège les scores "used" vs "contact" sur toutes les occurrences d'un numéro."""
    total_used = 0.0
    total_contact = 0.0
    evidences: List[Dict] = []

    for occ in occurrences:
        u, c, rules = _score_occurrence_role(occ.get("snippet", "") or "", person_terms_norm, bool(occ.get("linked")))
        total_used += u
        total_contact += c
        if rules:
            evidences.append({
                "source": occ.get("source"),
                "chunk_id": occ.get("chunk_id"),
                "page": occ.get("page"),
                "rule_hits": rules[:6],
                "snippet": occ.get("snippet", "")
            })

    total = total_used + total_contact
    if total <= 0:
        return ("unknown", UNKNOWN_MIN, evidences[:3])

    if total_used >= total_contact:
        role = "used_by_person"
        conf = min(CONF_CAP, max(UNKNOWN_MIN, total_used / (total + 0.5)))
    else:
        role = "contact_of_person"
        conf = min(CONF_CAP, max(UNKNOWN_MIN, total_contact / (total + 0.5)))

    evidences.sort(key=lambda e: -len(e.get("rule_hits", [])))
    return (role, conf, evidences[:3])

# =======================
# 1) Téléphones — extraction + classification
# =======================
def scan_case_phones(case_id: str, person: str, aliases: List[str] = None) -> Dict:
    """
    Extraction + classification par numéro:
      - items: rétro-compat (variants, occurrences)
      - used_by_person: liste prioritaire (number, confidence, evidence)
      - contacts: secondaires (number, confidence, evidence)
      - Association à la personne faite PAR OCCURRENCE (fenêtre ±PERSON_WINDOW_CHARS)
    """
    chunks, sources = _load_case_chunks(case_id)
    person_terms = [person] + (aliases or [])
    terms_norm = [_norm(t) for t in person_terms if t and t.strip()]
    results: Dict[str, Dict] = {}

    for i, text in enumerate(chunks):
        meta = sources[i]
        # Extractions déterministes depuis utils_normalize
        matches = extract_with_regex(
            text=text,
            regex=PHONE_RE,
            person=person,
            normalizer=normalize_phone,   # E.164 (+33…)
            default_cc="+33"
        )

        if not matches:
            continue

        for m in matches:
            e164 = m["value"]  # déjà normalisé
            linked = (m.get("assoc") == "strong")
            if STRICT_THEME_LINK and not linked:
                # En mode strict, on n'accepte que les occurrences proches du nom
                continue

            # Fabrique un snippet localisé autour du match
            span = m.get("span", (0, 0))
            snippet = _mk_snippet(text, span)

            entry = results.setdefault(e164, {"key": e164, "variants": set(), "occurrences": []})
            entry["variants"].add(m["raw"])
            entry["occurrences"].append(_occurrence(meta, snippet, linked))

    # Dédup occurrences par (source, chunk_id)
    items: List[Dict] = []
    for e in results.values():
        e["variants"] = sorted(e["variants"])
        seen = set(); uniq = []
        for o in e["occurrences"]:
            key = (o["source"], o["chunk_id"])
            if key in seen: 
                continue
            seen.add(key); uniq.append(o)
        e["occurrences"] = uniq
        items.append(e)

    # Classification par rôle + score
    used_group: List[Dict] = []
    contact_group: List[Dict] = []
    unknown_group: List[Dict] = []

    for e in items:
        role, conf, evidence = _aggregate_phone_role(e["occurrences"], terms_norm)
        summary = {
            "number": e["key"],
            "confidence": round(float(conf), 3),
            "occurrences": len(e["occurrences"]),
            "variants": e["variants"],
            "evidence": evidence,
        }
        if role == "used_by_person":
            used_group.append(summary)
        elif role == "contact_of_person":
            contact_group.append(summary)
        else:
            unknown_group.append(summary)

    used_group.sort(key=lambda x: (-(x["confidence"]), -x["occurrences"]))
    contact_group.sort(key=lambda x: (-(x["confidence"]), -x["occurrences"]))
    unknown_group.sort(key=lambda x: (-(x["confidence"]), -x["occurrences"]))

    # Rétro-compat: conserve items triés comme avant
    items.sort(key=lambda x: -len(x["occurrences"]))

    return {
        "person": person,
        "theme": "phones",
        "items": items,  # rétro-compat UI
        "used_by_person": used_group,
        "contacts": contact_group,
        "unknown": unknown_group[:10],  # optionnel: debug/inspection
    }

# =======================
# 2) IBAN / 3) Emails / 5) IPs / 4) Plaques — génériques
# =======================

def _aggregate_generic_theme(
    case_id: str,
    person: str,
    theme: str,
    regex: re.Pattern,
    normalizer=None,
    strict: bool = False,   # 👈 nouveau
) -> Dict:
    chunks, sources = _load_case_chunks(case_id)
    results: Dict[str, Dict] = {}

    for i, text in enumerate(chunks):
        meta = sources[i]

        matches = extract_with_regex(
            text=text,
            regex=regex,
            person=person,
            normalizer=normalizer
        )
        if not matches:
            continue

        for m in matches:
            val = m["value"]
            linked = (m.get("assoc") == "strong")

            # 👇 c'est ici qu'on applique le mode strict par REQUÊTE
            if strict and not linked:
                continue

            span = m.get("span", (0, 0))
            snippet = _mk_snippet(text, span)

            entry = results.setdefault(val, {"key": val, "variants": set(), "occurrences": []})
            entry["variants"].add(m["raw"])
            entry["occurrences"].append(_occurrence(meta, snippet, linked))

    items: List[Dict] = []
    for e in results.values():
        e["variants"] = sorted(e["variants"])
        seen=set(); uniq=[]
        for o in e["occurrences"]:
            k=(o["source"], o["chunk_id"])
            if k in seen: 
                continue
            seen.add(k); uniq.append(o)
        e["occurrences"]=uniq
        items.append(e)

    items.sort(key=lambda x: -len(x["occurrences"]))
    return {"person": person, "theme": theme, "items": items}

# =======================
# 6) Réseaux sociaux / handles (conserve tes règles existantes)
# =======================

RE_TG_HANDLE = re.compile(r'@([A-Za-z0-9_]{3,32})')
RE_TG_LINK   = re.compile(r'(?:t\.me|telegram\.me)/([A-Za-z0-9_]{3,32})', re.IGNORECASE)
RE_WA_LINK   = re.compile(r'(?:wa\.me|api\.whatsapp\.com)/(?:message/)?([A-Za-z0-9]+)', re.IGNORECASE)
RE_SNAP_LINK = re.compile(r'(?:snapchat\.com/add/|snapchat\.com/t/)([A-Za-z0-9\.\-_]{3,32})', re.IGNORECASE)
RE_SNAP_USER = re.compile(r'\b([A-Za-z0-9\.\-_]{3,15})\b')

def _extract_telegram(text: str, person: str) -> List[Tuple[str, Tuple[int,int], bool]]:
    hits=[]
    for m in RE_TG_HANDLE.finditer(text): 
        span=m.span(); hits.append(('@'+m.group(1), span, window_contains_person(text, person, span, PERSON_WINDOW_CHARS)))
    for m in RE_TG_LINK.finditer(text):   
        span=m.span(); hits.append(('@'+m.group(1), span, window_contains_person(text, person, span, PERSON_WINDOW_CHARS)))
    return hits

def _extract_whatsapp(text: str, person: str) -> List[Tuple[str, Tuple[int,int], bool]]:
    hits=[]
    for m in RE_WA_LINK.finditer(text): 
        span=m.span(); hits.append((m.group(0), span, window_contains_person(text, person, span, PERSON_WINDOW_CHARS)))
    return hits

def _extract_snapchat(text: str, person: str) -> List[Tuple[str, Tuple[int,int], bool]]:
    hits=[]
    if re.search(r'snap|snapchat', text, re.IGNORECASE):
        for m in RE_SNAP_LINK.finditer(text):
            span=m.span(); hits.append((m.group(1), span, window_contains_person(text, person, span, PERSON_WINDOW_CHARS)))
        for m in RE_SNAP_USER.finditer(text):
            u=m.group(1)
            if 3<=len(u)<=15:
                span=m.span(); hits.append((u, span, window_contains_person(text, person, span, PERSON_WINDOW_CHARS)))
    return hits

# =======================
# Dispatcher générique (public)
# =======================

def scan_case_theme(
    case_id: str,
    person: str,
    theme: str,
    aliases: List[str] = None,
    strict: Optional[bool] = None,   # 👈 nouveau
) -> Dict:
    """
    Scan thématique d'un dossier, éventuellement restreint aux occurrences proches de la personne.
    - if strict is None → on utilise la valeur globale STRICT_THEME_LINK
    - if strict is True → on ne garde que les occurrences liées à la personne
    - if strict is False → on garde tout
    """
    # on calcule le "strict" effectif
    effective_strict = STRICT_THEME_LINK if strict is None else bool(strict)

    theme = (theme or "").lower().strip()
    if theme == "phones":
        # on passe le strict jusqu'à la fonction specialized si besoin
        return scan_case_phones(case_id, person, aliases=aliases)

    # thèmes génériques
    if theme == "iban":
        return _aggregate_generic_theme(case_id, person, "iban", IBAN_RE, normalize_iban, strict=effective_strict)
    if theme == "emails":
        return _aggregate_generic_theme(case_id, person, "emails", EMAIL_RE, normalize_email, strict=effective_strict)
    if theme == "ips":
        return _aggregate_generic_theme(case_id, person, "ips", IP_RE, None, strict=effective_strict)
    if theme == "plates":
        return _aggregate_generic_theme(case_id, person, "plates", PLATE_RE, normalize_plate, strict=effective_strict)


    # Réseaux sociaux
    if theme in {"telegram", "whatsapp", "snapchat"}:
        chunks, sources = _load_case_chunks(case_id)
        results: Dict[str, Dict] = {}
        for i, text in enumerate(chunks):
            meta = sources[i]
            matches: List[Tuple[str, Tuple[int,int], bool]] = []
            if theme == "telegram":
                matches = _extract_telegram(text, person)
            elif theme == "whatsapp":
                matches = _extract_whatsapp(text, person)
            elif theme == "snapchat":
                matches = _extract_snapchat(text, person)

            if not matches:
                continue

            for val, span, linked in matches:
                if STRICT_THEME_LINK and not linked:
                    continue
                snippet = _mk_snippet(text, span)
                entry = results.setdefault(val, {"key": val, "variants": set(), "occurrences": []})
                entry["variants"].add(val)
                entry["occurrences"].append(_occurrence(meta, snippet, linked))

        items: List[Dict] = []
        for e in results.values():
            e["variants"] = sorted(e["variants"])
            seen=set(); uniq=[]
            for o in e["occurrences"]:
                k=(o["source"], o["chunk_id"])
                if k in seen: 
                    continue
                seen.add(k); uniq.append(o)
            e["occurrences"]=uniq
            items.append(e)

        items.sort(key=lambda x: -len(x["occurrences"]))
        return {"person": person, "theme": theme, "items": items}

    # Thème non supporté
    return {"person": person, "theme": theme, "items": []}
