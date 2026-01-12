# backend/analyzer.py
# Analyse thématique d'une affaire: détection de types de preuves à charge/décharge,
# extraction d'une timeline des faits matériels avec dates/horaires.
# Auteur: Jax

import os, re
from datetime import datetime
from typing import Dict, List, Optional
from .extractors import scan_case_theme, _norm
import functools
from typing import Pattern

MIN_LINKED_OCC_FOR_CHARGE_PHONES = int(os.getenv("ANALYZE_MIN_LINKED_OCC_FOR_CHARGE_PHONES", "2"))
STRICT_PROCEDURE_FILTER = os.getenv("ANALYZE_STRICT_PROCEDURE_FILTER", "1") == "1"
TIMELINE_INCLUDE_PROCEDURE = os.getenv("ANALYZE_TIMELINE_INCLUDE_PROCEDURE", "0") == "1"

PROCEDURE_HINTS = [
    "audition","perquisition","commission rogatoire","garde a vue","garde à vue","ordonnance",
    "réquisitoire","jugement","expertise","pv d'audition","réquisitions","decision","décision",
    "commission","procès-verbal","proces-verbal","opj","brigade","parquet","instruction"
]
def _looks_procedural(snippet:str)->bool:
    return any(w in _norm(snippet) for w in PROCEDURE_HINTS)

# -------------------- Détection "types de preuve" --------------------
# On bosse sur texte normalisé (minuscule, sans accents).
EVIDENCE_BUCKETS = {
    "empreintes_papillaires": {
        "charge": [r"\bempreinte(?:s)? (?:relevee|relevees|trouvee|trouvees)\b", r"\bcorrespondance d'?empreinte\b"],
        "decharge": [r"\baucune empreinte\b", r"\bempreinte (?:non|pas) exploitable\b", r"\bempreinte d'?un tiers\b"],
    },
    "adn": {
        "charge": [r"\badn (?:releve|relevee|trouve|trouvee)\b", r"\bprofil genetique (?:concordant|associe)\b"],
        "decharge": [r"\baucun adn\b", r"\badn d'?un tiers\b", r"\badn non (?:concordant|correspondant)\b"],
    },
    "temoignages": {
        "charge": [r"\btemoin (?:indique|decrit|reconnait)\b", r"\bidentifie (?:formellement|clairement)\b"],
        "decharge": [r"\btemoignage (?:contredit|imprecis|incertain)\b", r"\balibi (?:rapporte|corrobore)\b"],
    },
    "videos": {
        "charge": [r"\b(cctv|videosurveillance|camera)\b.*\bmontre\b", r"\bvu sur (?:cctv|video)\b"],
        "decharge": [r"\baucune image exploitable\b", r"\bimage floue|non identifiable\b"],
    },
    "alibi": {
        "charge": [r"\balibi (?:non (?:verifie|corrobore|credible)|contredit)\b"],
        "decharge": [r"\balibi (?:verifie|corrobore|coherent)\b", r"\bvu ailleurs\b"],
    },
    "geoloc_tel": {
        "charge": [r"\bbornage\b.*\bconcordant\b", r"\bantenne\b.*\bpresence\b", r"\bgeolocalis(?:e|ation)\b.*\bsite\b"],
        "decharge": [r"\bbornage\b.*\bnon (?:concordant|probant)\b", r"\bmultiples cellules\b"],
    },
    "transactions_bancaires": {
        "charge": [r"\bvirement\b|\btransfert\b|\bretrait\b|\bpaiement\b"],
        "decharge": [r"\baucun mouvement\b", r"\bcompte d'?un tiers\b"],
    },
    "plaques": {
        "charge": [r"\bplaque\b.*\bassociee\b|\bvehicule\b.*\butilise par\b|\bvu conduisant\b"],
        "decharge": [r"\bvehicule d'?un tiers\b|\bnon (?:titulaire|proprietaire)\b"],
    },
    "ips": {
        "charge": [r"\badresse ip\b.*\bconnexion\b|\blog\b.*\bdepuis\b"],
        "decharge": [r"\bip partagee\b|\breseau public\b|\bfree wifi\b"],
    },
}

# Téléphones: usage direct / fallback occurrences liées
THEME_USE_PATTERNS = {
    "phones": [r"\bappel (?:emis|recu)\b", r"\bconversation|ecoute\b", r"\butilise par|attribue a|titulaire\b"],
}

MONTHS = {
    "janvier":1,"fevrier":2,"février":2,"mars":3,"avril":4,"mai":5,"juin":6,
    "juillet":7,"aout":8,"août":8,"septembre":9,"octobre":10,"novembre":11,"decembre":12,"décembre":12
}
DATE_PATTERNS = [
    r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b', r'\b(\d{4}-\d{2}-\d{2})\b',
    r'\b(\d{1,2}\s+[A-Za-zéûîôàäëïöüç]+)\s+(\d{4})\b'
]
TIME_PATTERNS = [r'\b(\d{1,2}h\d{0,2})\b', r'\b(\d{1,2}:\d{2})\b']

def _extract_dates(snippet: str)->List[str]:
    out=[]
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, snippet, flags=re.IGNORECASE):
            out.append(m.group(1) if len(m.groups())==1 else f"{m.group(1)} {m.group(2)}")
    return out

def _extract_times(snippet:str)->List[str]:
    out=[]
    for pat in TIME_PATTERNS:
        for m in re.finditer(pat, snippet): out.append(m.group(1))
    return out

def _normalize_date(d:str)->Optional[str]:
    d=d.strip()
    m=re.fullmatch(r'(\d{1,2})/(\d{1,2})/(\d{2,4})', d)
    if m:
        dd,mm,yy=int(m.group(1)),int(m.group(2)),int(m.group(3)); yy+=2000 if yy<100 else 0
        try: return datetime(yy,mm,dd).date().isoformat()
        except: return None
    m=re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})', d)
    if m:
        try: return datetime(int(m.group(1)),int(m.group(2)),int(m.group(3))).date().isoformat()
        except: return None
    m=re.fullmatch(r'(\d{1,2})\s+([A-Za-zéûîôàäëïöüç]+)\s+(\d{4})', d.strip(), flags=re.IGNORECASE)
    if m:
        dd=int(m.group(1)); month=MONTHS.get(_norm(m.group(2)), None); yy=int(m.group(3))
        if month:
            try: return datetime(yy,month,dd).date().isoformat()
            except: return None
    return None

def _normalize_time(t:str)->Optional[str]:
    t=t.strip().lower().replace('h',':')
    if re.fullmatch(r'\d{1,2}$', t): t=f"{t}:00"
    if re.fullmatch(r'\d{1,2}:\d{2}$', t):
        try:
            hh,mm=map(int,t.split(':'))
            if 0<=hh<24 and 0<=mm<60: return f"{hh:02d}:{mm:02d}"
        except: return None
    return None

def _merge_dt(d:Optional[str], t:Optional[str])->Optional[str]:
    if not d and not t: return None
    return f"{d}T{t}" if (d and t) else d

def _match_any(rx_list:List[str], text_norm:str)->bool:
    # compile regexes with cache to avoid recompilation cost
    def _compile(rx: str) -> Pattern:
        return re.compile(rx, flags=re.IGNORECASE)
    return any(_compile(rx).search(text_norm) for rx in rx_list)

def analyze_case(case_id: str, person: str, aliases: Optional[List[str]] = None) -> Dict:
    themes = ["phones","iban","emails","plates","ips","telegram","whatsapp","snapchat"]
    analysis = {"charge": [], "decharge": [], "neutre": []}
    timeline_faits: List[Dict] = []
    # Buckets table: chaque type -> {'charge': [refs], 'decharge':[refs], 'neutre':[refs]}
    buckets: Dict[str, Dict[str, List[Dict]]] = {
        k: {"charge":[], "decharge":[], "neutre":[]} for k in EVIDENCE_BUCKETS.keys()
    }

    for theme in themes:
        scan = scan_case_theme(case_id, person, theme, aliases=aliases or [])
        items = scan.get("items", [])

        # Téléphones : compter occurrences liées par numéro
        phone_linked_counter: Dict[str,int] = {}
        if theme == "phones":
            for item in items:
                phone_linked_counter[item.get("key","?")] = sum(1 for o in item.get("occurrences", []) if o.get("linked"))

        for item in items:
            key = item.get("key","?")
            for occ in item.get("occurrences", []):
                snippet = occ.get("snippet","") or ""
                linked = bool(occ.get("linked"))
                tnorm = _norm(snippet)

                # 1) Décider charge/décharge/neutre (très simple)
                cls = "neutre"
                # Téléphones : usage direct ou fallback
                if theme == "phones":
                    if _match_any(THEME_USE_PATTERNS["phones"], tnorm) and linked:
                        cls = "charge"
                    elif phone_linked_counter.get(key,0) >= MIN_LINKED_OCC_FOR_CHARGE_PHONES:
                        cls = "charge"

                # 2) Mapper vers “type de preuve” + classer charge / decharge / neutre
                def push(bucket_name:str, force_cls:Optional[str]=None):
                    bcls = force_cls or cls
                    ref = {
                        "theme": theme,
                        "item": key,
                        "snippet": snippet,
                        "source": occ.get("source"),
                        "page": occ.get("page"),
                        # 👉 NOUVEAU : rattachement à l'unité logique (PV / rapport)
                        "unit_title": occ.get("unit_title"),
                        "unit_index": occ.get("unit_index"),
                        # 👉 NOUVEAU : type & date de doc si dispo via meta
                        "doc_type": occ.get("doc_type"),
                        "date_doc": occ.get("date"),
                    }
                    buckets[bucket_name][bcls].append(ref)
                    analysis[bcls].append(ref)

                # Empreintes / ADN / vidéos / etc.
                matched_bucket = False
                for bucket, pats in EVIDENCE_BUCKETS.items():
                    if _match_any(pats["decharge"], tnorm):
                        push(bucket, "decharge")
                        matched_bucket = True
                        break
                    if _match_any(pats["charge"], tnorm):
                        push(bucket, "charge")
                        matched_bucket = True
                        break

                if not matched_bucket:
                    # si non classé dans un bucket probant, ranger générique (selon cls actuel)
                    analysis[cls].append({
                        "theme": theme,
                        "item": key,
                        "snippet": snippet,
                        "source": occ.get("source"),
                        "page": occ.get("page"),
                        # même enrichissement pour la vue détaillée
                        "unit_title": occ.get("unit_title"),
                        "unit_index": occ.get("unit_index"),
                        "doc_type": occ.get("doc_type"),
                        "date_doc": occ.get("date"),
                    })

                # 3) Timeline des faits matériels
                looks_proc = _looks_procedural(snippet)
                if not TIMELINE_INCLUDE_PROCEDURE and looks_proc and STRICT_PROCEDURE_FILTER:
                    pass
                else:
                    d_iso, t_iso = None, None
                    for d in _extract_dates(snippet):
                        d_iso=_normalize_date(d)
                        if d_iso: break
                    for t in _extract_times(snippet):
                        t_iso=_normalize_time(t)
                        if t_iso: break
                    dt = _merge_dt(d_iso, t_iso)
                    if dt:
                        timeline_faits.append({
                            "date": dt,
                            "theme": theme,
                            "item": key,
                            "snippet": snippet,
                            "classification": cls,
                            # 👉 NOUVEAU : rattachement doc / unité
                            "source": occ.get("source"),
                            "page": occ.get("page"),
                            "unit_title": occ.get("unit_title"),
                            "unit_index": occ.get("unit_index"),
                            "doc_type": occ.get("doc_type"),
                            "date_doc": occ.get("date"),
                        })

    # Trier frise
    timeline_faits.sort(key=lambda x: x.get("date") or "")

    # Construire un tableau synthétique {label, statut, refs[]}
    def pick_refs(arr:List[Dict], n:int=2)->List[Dict]:
        # 1–2 références max pour la lisibilité
        return arr[:n] if len(arr)>n else arr

    summary_table = []
    LABELS = {
        "empreintes_papillaires": "Empreintes papillaires",
        "adn": "Empreintes génétiques (ADN)",
        "temoignages": "Témoignages",
        "videos": "Vidéosurveillance / Caméras",
        "alibi": "Alibi",
        "geoloc_tel": "Géolocalisation téléphonique",
        "transactions_bancaires": "Transactions bancaires",
        "plaques": "Plaques / Véh.",
        "ips": "Connexions IP",
    }
    for bucket, groups in buckets.items():
        if any(groups.values()):  # quelque chose trouvé
            if groups["charge"]:
                status="À charge"
                refs = pick_refs(groups["charge"])
            elif groups["decharge"]:
                status="À décharge"
                refs = pick_refs(groups["decharge"])
            else:
                status="Indéterminé"
                refs = pick_refs(groups["neutre"])
            summary_table.append({
                "label": LABELS.get(bucket, bucket),
                "status": status,
                "refs": refs
            })

    return {
        "person": person,
        "analysis": analysis,             # détail brut (toujours dispo)
        "summary_table": summary_table,   # ✅ tableau court et lisible
        "timeline_faits": timeline_faits  # ✅ frise (faits matériels, avec PV/rapport rattaché)
    }
