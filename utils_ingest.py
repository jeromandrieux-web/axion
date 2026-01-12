# -*- coding: utf-8 -*-
"""
utils_ingest.py — Détection de métadonnées documentaires pour l'ingestion

Fonctions exposées :
- detect_doc_date(text_first_pages: str, filename: str) -> str | None
- detect_doc_type(text_header: str, filename: str = "") -> str
- detect_unit_title_from_page(page_text: str) -> str | None
- split_pages_into_units(pages: list[tuple[int, str]], filename: str) -> list[dict]

Objectif :
Extraire une date ISO (YYYY-MM-DD), un type de document (pv, pv_audition, pv_gav,
pv_identite, rapport, expertise, ecoutes, constatations, document) et des unités logiques
(PV, rapports, etc.) à partir du contenu et du nom de fichier.

Aucune dépendance externe (standard library only).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable, Optional, List, Tuple, Dict, Any

# ------------------------------------------------------------
# DATES — Internationalisation (français)
# ------------------------------------------------------------

MONTHS_FR = {
    "janvier": 1,
    "février": 2, "fevrier": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "août": 8, "aout": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "décembre": 12, "decembre": 12,
}

# Ordre de priorité des motifs :
# 1) Formats jour-mois-année (usage FR) dans le CONTENU
# 2) Formats ISO (AAAA-MM-JJ)
# 3) Formats "12 mars 2021" (avec ou sans "le ")
# 4) Variantes filename (ISO, AAAA/MM/JJ, AAAAMMJJ, JJ-MM-AAAA)
DATE_PATTERNS = [
    # 12/03/2021 ou 12-03-2021 ou 12.03.2021
    re.compile(r"\b(\d{1,2})[\./\-](\d{1,2})[\./\-](\d{4})\b"),
    # 2021-03-12 ou 2021/03/12
    re.compile(r"\b(\d{4})[\-/](\d{1,2})[\-/](\d{1,2})\b"),
    # 12 mars 2021 / le 12 mars 2021 (insensible à la casse/accents simplifiés)
    re.compile(r"\b(?:le\s+)?(\d{1,2})\s+([a-zéèêëàâîïôöùûç]+)\s+(\d{4})\b", re.IGNORECASE),
    # 20210312 (collé)
    re.compile(r"\b(\d{4})(\d{2})(\d{2})\b"),
]

# Dans certains en-têtes on voit : "Fait à X, le 3/4/2020 à 14h30" → on isole la partie date.
TIME_TRAIL = re.compile(r"\b(?:à|a)\s*\d{1,2}[:h]\d{2}(?::\d{2})?\b", re.IGNORECASE)


def _to_iso_date(y: int, m: int, d: int) -> str:
    """Convertit en date ISO, en tentant de corriger des jours/mois hors bornes.
    Si datetime échoue (p. ex. 31/11), on borne le jour au max du mois.
    """
    try:
        return datetime(y, m, d).strftime("%Y-%m-%d")
    except ValueError:
        # Borne simple (sans calendrier précis pour bissextile) :
        # on retombe sur le dernier jour plausible du mois
        # pour éviter de perdre l'information (au pire, Y-M est juste).
        DAYS_IN_MONTH = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
        m = 1 if m < 1 or m > 12 else m
        d = 1 if d < 1 else min(d, DAYS_IN_MONTH[m])
        return f"{y:04d}-{m:02d}-{d:02d}"


def _normalize_text(s: str) -> str:
    return (s or "").strip()


def _scan_text_for_date(text: str) -> Optional[str]:
    """Scanne le texte pour trouver la première date plausible."""
    if not text:
        return None

    cleaned = TIME_TRAIL.sub("", text)

    for i, p in enumerate(DATE_PATTERNS):
        for m in p.finditer(cleaned):
            if i == 0:  # dd[./-]mm[./-]yyyy
                d, mth, y = map(int, m.groups())
                if 1 <= mth <= 12 and 1 <= d <= 31:
                    return _to_iso_date(y, mth, d)
            elif i == 1:  # yyyy[/-]mm[/-]dd
                y, mth, d = map(int, m.groups())
                if 1 <= mth <= 12 and 1 <= d <= 31:
                    return _to_iso_date(y, mth, d)
            elif i == 2:  # dd mois yyyy
                d = int(m.group(1))
                name = m.group(2).lower()
                y = int(m.group(3))
                mth = MONTHS_FR.get(name)
                if mth and 1 <= d <= 31:
                    return _to_iso_date(y, mth, d)
            else:        # yyyymmdd
                y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 1 <= mth <= 12 and 1 <= d <= 31:
                    return _to_iso_date(y, mth, d)
    return None


def _scan_filename_for_date(filename: str) -> Optional[str]:
    """Heuristiques de date dans les noms de fichier."""
    if not filename:
        return None
    name = filename

    # Essayer ISO d'abord (AAAA-MM-JJ / AAAA_MM_JJ / AAAA.MM.JJ)
    m = re.search(r"(\d{4})[\-_.](\d{2})[\-_.](\d{2})", name)
    if m:
        y, mth, d = map(int, m.groups())
        return _to_iso_date(y, mth, d)

    # 20210312 compact
    m = re.search(r"\b(\d{4})(\d{2})(\d{2})\b", name)
    if m:
        y, mth, d = map(int, m.groups())
        return _to_iso_date(y, mth, d)

    # JJ-MM-AAAA (fréquent dans certains exports)
    m = re.search(r"\b(\d{2})[\-_.](\d{2})[\-_.](\d{4})\b", name)
    if m:
        d, mth, y = map(int, m.groups())
        return _to_iso_date(y, mth, d)

    # JJ mois AAAA (rare dans filename, mais possible)
    m = re.search(r"\b(\d{1,2})\s+([a-zéèêëàâîïôöùûç]+)\s+(\d{4})\b", name, re.IGNORECASE)
    if m:
        d = int(m.group(1))
        mth = MONTHS_FR.get(m.group(2).lower())
        y = int(m.group(3))
        if mth:
            return _to_iso_date(y, mth, d)

    return None


def detect_doc_date(text_first_pages: str, filename: str) -> Optional[str]:
    """Retourne une date ISO YYYY-MM-DD si trouvée, sinon None."""
    text_first_pages = _normalize_text(text_first_pages)
    filename = _normalize_text(filename)

    found = _scan_text_for_date(text_first_pages)
    if found:
        return found
    return _scan_filename_for_date(filename)

# ------------------------------------------------------------
# TYPES DE DOCUMENTS — heuristiques
# ------------------------------------------------------------

# ✅ Ajout "pv_identite" + patterns identité (état civil / signalement / filiation)
DOC_TYPE_CANON = {
    "pv_audition": [r"\bpv d['’]?\s*audition\b", r"\bpv\s+audition\b", r"\baudition\s+de\b"],
    "pv_gav": [r"\bgarde\s*à\s*vue\b", r"\bgav\b"],
    "pv_identite": [
        r"\bv[ée]rification\s+d['’]?\s*identit[ée]\b",
        r"\bcontr[ôo]le\s+d['’]?\s*identit[ée]\b",
        r"\bcontrole\s+d['’]?\s*identit[ée]\b",
        r"\b[eé]tat\s+civil\b",
        r"\bsignalement\b",
        r"\bfiliation\b",
        r"\bpi[eè]ce\s+d['’]?\s*identit[ée]\b",
        r"\bcarte\s+nationale\b",
        r"\bpasseport\b",
        # on garde "identité" mais tard dans la liste pour éviter des faux positifs
        r"\bidentit[ée]\b",
    ],
    "pv": [r"\bproc[eè]s\-?verbal\b", r"\bproces\-?verbal\b", r"\bpv\b"],
    "rapport": [r"\brapport\b", r"\brap\.\b"],
    "expertise": [r"\bexpertise\b", r"\bexpert\b"],
    "ecoutes": [r"\b[eé]coutes?\b", r"\binterceptions?\b", r"\btranscriptions?\b"],
    "constatations": [r"\bconstat(?:ation|ations)?\b", r"\bproc[eè]s\-?verbal\s+de\s+constat\b"],
}

# Précompile les regex
DOC_TYPE_PATTERNS = {
    key: [re.compile(pat, re.IGNORECASE) for pat in patterns]
    for key, patterns in DOC_TYPE_CANON.items()
}


def detect_doc_type(text_header: str, filename: str = "") -> str:
    """Détecte un type de document selon des heuristiques robustes.

    On cherche d'abord un type spécialisé (pv_audition, pv_gav, pv_identite), puis pv, rapport,
    expertise, ecoutes, constatations. Si rien n'est trouvé, on renvoie 'document'.
    """
    blob = f"{text_header or ''}\n{filename or ''}".lower()
    for key in [
        "pv_audition",
        "pv_gav",
        "pv_identite",   # ✅ avant pv générique
        "pv",
        "rapport",
        "expertise",
        "ecoutes",
        "constatations",
    ]:
        for pat in DOC_TYPE_PATTERNS[key]:
            if pat.search(blob):
                return key
    return "document"


# ------------------------------------------------------------
# DÉTECTION D'UNITÉS LOGIQUES (PV / RAPPORT / …)
# ------------------------------------------------------------

# Motifs typiques de début de document (procès-verbal, rapport, expertise, etc.)
# ✅ Ajout motifs "identité" + "état civil" + "signalement"
DOC_START_PATTERNS = [
    r"\bproc[èe]s[- ]?verbal\b",
    r"\bpv d['’]?\s*audition\b",
    r"\bpv\s+audition\b",
    r"\baudition\s+de\b",
    r"\bgarde\s*à\s*vue\b",
    r"\bgav\b",
    # identité / état civil
    r"\bv[ée]rification\s+d['’]?\s*identit[ée]\b",
    r"\bcontr[ôo]le\s+d['’]?\s*identit[ée]\b",
    r"\b[eé]tat\s+civil\b",
    r"\bsignalement\b",
    r"\bfiliation\b",
    r"\brapport\b",
    r"\bexpertise\b",
    r"\bconstat(?:ation|ations)?\b",
]

DOC_START_REGEXES = [re.compile(pat, re.IGNORECASE) for pat in DOC_START_PATTERNS]

# ✅ Détection "formulaire identité" même sans titre clair
IDENTITY_FIELD_PATTERNS = [
    r"\bnom\s*[:\-]\s*",
    r"\bpr[eé]nom\s*[:\-]\s*",
    r"\bdate\s+et\s+lieu\s+de\s+naissance\s*[:\-]\s*",
    r"\bdate\s+de\s+naissance\s*[:\-]\s*",
    r"\blieu\s+de\s+naissance\s*[:\-]\s*",
    r"\bfiliation\s*[:\-]\s*",
    r"\badresse\s*[:\-]\s*",
    r"\bdomicile\s*[:\-]\s*",
]

IDENTITY_FIELD_REGEXES = [re.compile(p, re.IGNORECASE) for p in IDENTITY_FIELD_PATTERNS]

# Bonus OCR courant : "N0M", "PRÉN0M" (0 à la place de O)
IDENTITY_FIELD_OCR_PATTERNS = [
    r"\bn0m\s*[:\-]\s*",
    r"\bpr[eé]n0m\s*[:\-]\s*",
]
IDENTITY_FIELD_OCR_REGEXES = [re.compile(p, re.IGNORECASE) for p in IDENTITY_FIELD_OCR_PATTERNS]


def _looks_like_identity_form(page_text: str) -> bool:
    """Retourne True si la page ressemble à un bloc identité (champs Nom/Prénom/etc.)."""
    if not page_text:
        return False
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    head = "\n".join(lines[:30]).lower()

    score = 0
    for rx in IDENTITY_FIELD_REGEXES:
        if rx.search(head):
            score += 1
    for rx in IDENTITY_FIELD_OCR_REGEXES:
        if rx.search(head):
            score += 1

    # Seuil : 3 champs (ou 2 + filiation) = très probable
    if score >= 3:
        return True
    if score >= 2 and re.search(r"\bfiliation\s*[:\-]", head, re.IGNORECASE):
        return True
    return False


def _is_probable_title(line: str) -> bool:
    """Heuristique simple pour repérer une ligne qui ressemble à un titre."""
    if not line:
        return False
    line = line.strip()
    if len(line) < 8 or len(line) > 180:
        return False

    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio >= 0.6


def detect_unit_title_from_page(page_text: str) -> Optional[str]:
    """
    Essaie de trouver un titre de document (PV d'audition, PV identité, Rapport, etc.)
    dans le haut d'une page.

    On regarde :
    - d'abord les motifs explicites (PROCÈS-VERBAL, RAPPORT, EXPERTISE, ...)
    - sinon un formulaire identité (champs NOM/PRÉNOM/...) même sans titre clair
    - sinon, une ligne très en majuscules dans les premières lignes.
    """
    if not page_text:
        return None

    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    if not lines:
        return None

    # on se concentre sur le "header" de page (OCR peut décaler → un peu plus large)
    head_lines = lines[:12]

    # 1) motif explicite
    for line in head_lines:
        for rx in DOC_START_REGEXES:
            if rx.search(line):
                return line

    # 1bis) identité "formulaire" sans titre clair
    if _looks_like_identity_form(page_text):
        return "IDENTITÉ / ÉTAT CIVIL (détecté)"

    # 2) fallback : première ligne très majuscules
    for line in head_lines:
        if _is_probable_title(line):
            return line

    return None


def split_pages_into_units(
    pages: List[Tuple[int, str]],
    filename: str,
) -> List[Dict[str, Any]]:
    """
    Découpe une liste de (page_num, page_text) en unités logiques
    (PV, rapport, etc.), basées sur la détection de titres.

    Si aucune frontière claire n'est trouvée, tout le fichier est une seule unité.
    """
    units: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {"title": None, "pages": []}

    for page_num, page_text in pages:
        title = detect_unit_title_from_page(page_text)

        # Nouveau début de doc logique ?
        if title and current["pages"]:
            if not current["title"]:
                current["title"] = f"{filename} (partie {len(units) + 1})"
            units.append(current)
            current = {"title": title, "pages": []}

        # Initialisation éventuelle du titre si on n'en a pas encore
        if not current["title"] and title:
            current["title"] = title

        current["pages"].append((page_num, page_text))

    # Dernière unité
    if current["pages"]:
        if not current["title"]:
            current["title"] = filename
        units.append(current)

    return units


# ------------------------------------------------------------
# Helpers d'intégration (facultatifs)
# ------------------------------------------------------------

def enrich_document_meta(doc: dict, *, text_first_pages: str = "", filename: str = "") -> dict:
    """Complète un dict document avec `date` et `doc_type` si absents.

    doc: dict mutable (existant) avec au moins 'name' ou 'title'.
    Retourne le dict (muté) pour chaînage.
    """
    if not filename:
        filename = doc.get("name") or doc.get("title") or ""
    if not doc.get("date"):
        doc["date"] = detect_doc_date(text_first_pages, filename)
    if not doc.get("doc_type"):
        doc["doc_type"] = detect_doc_type(text_first_pages, filename)
    return doc


# ------------------------------------------------------------
# Mini tests manuels
# ------------------------------------------------------------
if __name__ == "__main__":
    samples: Iterable[tuple[str, str]] = [
        ("Procès-verbal d'audition du témoin X\nFait à Paris, le 12 mars 2021 à 14h30.", "PV_AUDITION_20210312_Dupont.pdf"),
        ("PROCÈS-VERBAL DE VÉRIFICATION D’IDENTITÉ\nNom: DUPONT\nPrénom: Jean\nDate et lieu de naissance: 12/03/1990 à Paris\nFiliation: ...\nAdresse: ...", "PV_IDENTITE_20210312_Dupont.pdf"),
        ("NOM : DUPONT\nPRÉNOM : Jean\nDATE ET LIEU DE NAISSANCE : 12/03/1990 à Paris\nFILIATION : ...\nADRESSE : ...", "INTERROGATOIRE_1.pdf"),
        ("RAPPORT d'enquête — Section financière\nDate: 2021-03-14", "Rapport_SF_2021-03-14.pdf"),
        ("Expertise balistique\nLe 03/04/2020", "expertise_2020_04_03.pdf"),
        ("Transcription des écoutes — Ligne 06…", "ECOUTES_2019-12-01.txt"),
        ("Procès-verbal de constatations", "PV_Constat_12-11-2018.pdf"),
        ("Sans date claire dans le contenu", "PV_GAV_20170505.pdf"),
        ("Note interne", "note_interne.pdf"),
    ]

    print("=== TEST DATES & TYPES ===")
    for text, fname in samples:
        d = detect_doc_date(text, fname)
        t = detect_doc_type(text, fname)
        print(f"{fname:35s} -> date={d!r}  type={t}")

    print("\n=== TEST SPLIT PAGES INTO UNITS ===")
    demo_pages = [
        (1, "PROCÈS-VERBAL D'AUDITION DE M. DUPONT\nFait à Paris, le 12 mars 2021...\n[...]"),
        (2, "Suite de l'audition de M. DUPONT...\n[...]"),
        (3, "NOM : DUPONT\nPRÉNOM : Jean\nDATE ET LIEU DE NAISSANCE : 12/03/1990 à Paris\nFILIATION : ...\nADRESSE : ...\n[...]"),
        (4, "Suite interrogation...\n[...]"),
        (5, "RAPPORT D'ENQUÊTE — Section financière\nDate: 2021-03-14\n[...]"),
        (6, "Suite du rapport d'enquête...\n[...]"),
    ]
    units = split_pages_into_units(demo_pages, "demo.pdf")
    for i, u in enumerate(units):
        print(f"Unité {i}: title={u['title']!r}, pages={[p[0] for p in u['pages']]}")
