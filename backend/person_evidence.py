# backend/person_evidence.py
# Module de recherche d'éléments procéduraux par personne
# Auteur: Jax

from __future__ import annotations
from typing import Any, Dict, List, Optional
import unicodedata
import re

from . import person_retrieval

Evidence = Dict[str, Any]


def _norm(text: str) -> str:
    """Normalisation simple : minuscules, suppression des accents, espaces compactés."""
    if not text:
        return ""
    text = text.lower()
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================
# Helpers PROCÉDURE / TECH
# =========================

def _looks_gav(t: str) -> bool:
    return any(k in t for k in [
        "garde a vue",
        "gav",
        "retenu",
        "retenue",
        "debut de garde a vue",
        "fin de garde a vue",
    ])


def _looks_crt(t: str) -> bool:
    return "commission rogatoire" in t or "crt" in t or "rogatoire technique" in t


def _looks_interception(t: str) -> bool:
    return any(k in t for k in [
        "ecoute",
        "interception",
        "captation",
        "enregistrement",
        "retranscription",
    ])


def _looks_geoloc(t: str) -> bool:
    return any(k in t for k in [
        "borne",
        "bornage",
        "geolocal",
        "cell-id",
        "cell id",
        "localisation antenne",
        "cdr",
        "fadet",
    ])


def _looks_balise(t: str) -> bool:
    return any(k in t for k in [
        "balise",
        "tracker",
        "geotrace",
        "dispositif de suivi",
        "pose de balise",
        "geotracking",
    ])


def _looks_sono(t: str) -> bool:
    return any(k in t for k in [
        "sonorisation",
        "captation sonore",
        "pose de micro",
        "vehicule sonorise",
        "piece sonorisee",
    ])


# =========================
# Helpers SCIENTIFIQUE
# =========================

def _looks_adn(t: str) -> bool:
    return any(k in t for k in [
        "adn",
        "profil genetique",
        "analyse genetique",
        "locus",
    ])


def _looks_ballistic(t: str) -> bool:
    return any(k in t for k in [
        "balistique",
        "projectile",
        "douille",
        "etude balistique",
    ])


def _looks_forensic_phone(t: str) -> bool:
    return any(k in t for k in [
        "extraction",
        "ufed",
        "cellebrite",
        "dump",
        "physical extraction",
        "logical extraction",
        "analyse telephone",
        "analyse mobile",
    ])


def _looks_odorology(t: str) -> bool:
    return "odorologie" in t or "piste olfactive" in t


def _looks_empreintes(t: str) -> bool:
    return any(k in t for k in [
        "empreinte",
        "papillaire",
        "dactylo",
        "trace papillaire",
    ])


# =========================
# Helpers DECLARATIONS
# =========================

def _looks_audition(t: str) -> bool:
    return any(k in t for k in [
        "audition de",
        "proces-verbal d audition",
        "procès-verbal d audition",
        "declare que",
        "declarant que",
        "interrogatoire",
        "ipc",
        "deposition",
        "entendu",
        "entendue",
    ])


# =========================
# Construction Evidence
# =========================

def _make_evidence(
    case_id: str,
    person: str,
    category: str,
    snippet: Dict[str, Any],
) -> Evidence:
    meta = snippet.get("meta") or {}
    return {
        "person": person,
        "case_id": case_id,
        "category": category,
        "type_label": meta.get("unit_title") or meta.get("doc_type") or "",
        "date": meta.get("date"),
        "time": meta.get("time"),
        "unit_title": meta.get("unit_title"),
        "unit_index": meta.get("unit_index"),
        "source": meta.get("source"),
        "page": meta.get("page"),
        "snippet": snippet.get("text"),
        "score": snippet.get("score", 0.0),
    }


def _get_global_snippets(
    case_id: str,
    person: str,
    aliases: Optional[List[str]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Récupère les snippets 'globaux' via person_retrieval."""
    aliases = aliases or []
    return person_retrieval.find_person_global_snippets(
        case_id,
        person,
        aliases=aliases,
        limit=limit,
    )


def _dedupe_evidences(evidences: List[Evidence]) -> List[Evidence]:
    seen = set()
    uniq: List[Evidence] = []
    for ev in evidences:
        key = (ev.get("category"), ev.get("unit_index"), ev.get("page"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(ev)
    return uniq


# =========================
# Principal — Procédure / tech
# =========================

def find_person_procedure_tech(
    case_id: str,
    person: str,
    aliases: Optional[List[str]] = None,
    limit: int = 200,
) -> List[Evidence]:
    """GAV, CRT, écoutes, géoloc, balises, sonorisations."""
    snippets = _get_global_snippets(case_id, person, aliases, limit=limit * 2)
    evidences: List[Evidence] = []

    for sn in snippets:
        t = _norm(sn.get("text", ""))
        if not t:
            continue

        if _looks_gav(t):
            evidences.append(_make_evidence(case_id, person, "gav", sn))
        if _looks_crt(t):
            evidences.append(_make_evidence(case_id, person, "crt", sn))
        if _looks_interception(t):
            evidences.append(_make_evidence(case_id, person, "ecoutes", sn))
        if _looks_geoloc(t):
            evidences.append(_make_evidence(case_id, person, "geoloc", sn))
        if _looks_balise(t):
            evidences.append(_make_evidence(case_id, person, "balise", sn))
        if _looks_sono(t):
            evidences.append(_make_evidence(case_id, person, "sono", sn))

    return _dedupe_evidences(evidences)[:limit]


# =========================
# Principal — Scientifique
# =========================

def find_person_scientific(
    case_id: str,
    person: str,
    aliases: Optional[List[str]] = None,
    limit: int = 200,
) -> List[Evidence]:
    """ADN, empreintes, balistique, forensique téléphone/PC, odorologie."""
    snippets = _get_global_snippets(case_id, person, aliases, limit=limit * 2)
    evidences: List[Evidence] = []

    for sn in snippets:
        t = _norm(sn.get("text", ""))
        if not t:
            continue

        if _looks_adn(t):
            evidences.append(_make_evidence(case_id, person, "adn", sn))
        if _looks_empreintes(t):
            evidences.append(_make_evidence(case_id, person, "empreintes", sn))
        if _looks_ballistic(t):
            evidences.append(_make_evidence(case_id, person, "balistique", sn))
        if _looks_forensic_phone(t):
            evidences.append(_make_evidence(case_id, person, "forensic_phone", sn))
        if _looks_odorology(t):
            evidences.append(_make_evidence(case_id, person, "odorologie", sn))

    return _dedupe_evidences(evidences)[:limit]


# =========================
# Principal — Déclarations
# =========================

def find_person_statements(
    case_id: str,
    person: str,
    aliases: Optional[List[str]] = None,
    limit: int = 200,
) -> List[Evidence]:
    """Auditions, interrogatoires, IPC, dépositions."""
    snippets = _get_global_snippets(case_id, person, aliases, limit=limit * 2)
    evidences: List[Evidence] = []

    for sn in snippets:
        t = _norm(sn.get("text", ""))
        if not t:
            continue

        meta = sn.get("meta") or {}
        doc_type = (meta.get("doc_type") or "").lower()
        if doc_type in {"pv_audition", "pv_gav"}:
            evidences.append(_make_evidence(case_id, person, "audition", sn))
            continue

        if _looks_audition(t):
            evidences.append(_make_evidence(case_id, person, "audition", sn))

    return _dedupe_evidences(evidences)[:limit]
