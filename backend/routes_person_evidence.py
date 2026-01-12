# backend/routes_person_evidence.py
# Routes d'extraction de fiche personne
# Auteur: Jax

from __future__ import annotations
from typing import List

from fastapi import APIRouter, Form

from .person_evidence import (
    find_person_procedure_tech,
    find_person_scientific,
    find_person_statements,
)

router = APIRouter()


def _parse_aliases(aliases: str) -> List[str]:
    return [a.strip() for a in aliases.split(",") if a.strip()]


@router.post("/api/extract/person/procedure")
async def extract_person_procedure(
    case_id: str = Form(...),
    person: str = Form(...),
    aliases: str = Form(""),
):
    """
    Retourne toutes les GAV / CRT / écoutes / géolocalisations / balises /
    sonorisations concernant une personne, sans synthèse LLM.
    """
    aliases_list = _parse_aliases(aliases)
    evidences = find_person_procedure_tech(case_id, person, aliases_list)
    return {"ok": True, "person": person, "evidences": evidences}


@router.post("/api/extract/person/science")
async def extract_person_science(
    case_id: str = Form(...),
    person: str = Form(...),
    aliases: str = Form(""),
):
    """
    Expertises scientifiques : ADN, empreintes, balistique,
    forensique téléphone/PC, odorologie, etc.
    """
    aliases_list = _parse_aliases(aliases)
    evidences = find_person_scientific(case_id, person, aliases_list)
    return {"ok": True, "person": person, "evidences": evidences}


@router.post("/api/extract/person/statements")
async def extract_person_statements_route(
    case_id: str = Form(...),
    person: str = Form(...),
    aliases: str = Form(""),
):
    """
    Auditions, interrogatoires, IPC, dépositions diverses.
    """
    aliases_list = _parse_aliases(aliases)
    evidences = find_person_statements(case_id, person, aliases_list)
    return {"ok": True, "person": person, "evidences": evidences}
