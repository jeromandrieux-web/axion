# backend/routes_extract.py
# Routes d'extraction de fiche personne
# Auteur: Jax

from __future__ import annotations
import json
import logging
import re
import asyncio
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import StreamingResponse

# utilitaires d’extraction existants
from .extractors import (
    scan_case_phones,
    scan_case_theme,
    SUPPORTED_THEMES,
)

# modules internes
from . import ollama_client
from . import person_retrieval
from .person_evidence import (
    find_person_procedure_tech,
    find_person_scientific,
    find_person_statements,
)
from identity_mapper import anonymize_prompt_for_model, deanonymize_text_for_case

router = APIRouter()
logger = logging.getLogger("axion.routes_extract")

# racine de stockage (même logique que person_retrieval)
STORAGE_ROOT = Path(os.getenv("STORAGE_DIR", "backend/storage")).resolve()

# ----------------- validations simples -----------------
# on autorise lettres/chiffres/underscore/tiret/point/espace
_CASE_ID_RE = re.compile(r'^[A-Za-z0-9_\-\. ]+$')
_NAME_RE = re.compile(r'^[\w\-\s]{1,128}$', re.UNICODE)


def _sanitize_case_id(case_id: str) -> str:
    """
    Valide le case_id : non vide, sans caractères dangereux (pas de /, pas de \).
    """
    if not case_id or not _CASE_ID_RE.fullmatch(case_id):
        raise HTTPException(status_code=400, detail="case_id manquant ou invalide")
    return case_id


def _sanitize_person(person: str) -> str:
    if not person or not _NAME_RE.fullmatch(person):
        raise HTTPException(status_code=400, detail="person invalide")
    return person


def _sse(event: Dict) -> str:
    return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"


def _assemble_sources_from_items(items: List[Dict]) -> List[Dict]:
    """Déduplique (source, page, chunk_id) pour l'event 'end'."""
    seen = set()
    out = []
    for it in items or []:
        for occ in it.get("occurrences", []):
            key = (occ.get("source"), occ.get("page"), occ.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            out.append({"source": key[0], "page": key[1], "chunk_id": key[2]})
    return out


def _assoc_badge(has_strong: bool) -> str:
    return " (associé à la personne)" if has_strong else ""


def _write_person_profile_file(case_id: str, person: str, text: str) -> str:
    """
    Sauvegarde la fiche personne dans le stockage, et renvoie une URL /storage/...
    que le front peut utiliser comme lien de téléchargement (comme pour les synthèses).
    """
    case_dir = STORAGE_ROOT / "cases" / case_id / "summaries"
    case_dir.mkdir(parents=True, exist_ok=True)

    # petit slug propre pour le nom de fichier
    slug = re.sub(r"[^a-z0-9]+", "_", person.lower())
    slug = slug.strip("_") or "personne"
    out_path = case_dir / f"fiche_personne_{slug}.md"

    out_path.write_text(text, encoding="utf-8")

    # on renvoie un chemin relatif à la racine de stockage, exposé sous /storage/...
    rel = out_path.relative_to(STORAGE_ROOT)
    return f"/storage/{rel.as_posix()}"


# =========================================================
# Helpers fiche personne : extractions techniques (phones / iban / etc.)
# =========================================================

def _summarize_theme_items(theme_label: str, items: List[Dict], limit: int = 40) -> Optional[str]:
    """
    Prend une liste d'items d'extraction (phones, iban, emails, etc.)
    et renvoie un bloc texte '### Titre\n- valeur (...) — sources'.
    """
    if not items:
        return None

    lines: List[str] = []
    for it in items[:limit]:
        key = it.get("value") or it.get("key")
        if not key:
            continue
        occs = it.get("occurrences") or []
        strong = any(o.get("linked") or o.get("assoc") == "strong" for o in occs)
        badge = "associé fortement à la personne" if strong else "mentionné"

        srcs: List[str] = []
        for o in occs[:4]:
            s = o.get("source") or "?"
            p = o.get("page")
            if p is not None:
                srcs.append(f"{s} p.{p}")
            else:
                srcs.append(str(s))
        src_str = ", ".join(srcs) if srcs else "sources non précisées"

        lines.append(f"- {key} ({badge}) — {src_str}")

    if not lines:
        return None

    return "### " + theme_label + "\n" + "\n".join(lines)


async def _gather_technical_extractions(case_id: str, person: str, aliases: List[str]) -> str:
    """
    Fait plusieurs passes sur le dossier via les extracteurs thématiques :
      - téléphones
      - iban / comptes
      - emails
      - plaques
      - IP
      - Telegram / WhatsApp / Snapchat

    et renvoie un bloc texte prêt à être injecté dans le contexte LLM.
    """
    blocks: List[str] = []

    # 1) téléphones
    try:
        phones_data = await asyncio.to_thread(
            scan_case_phones,
            case_id,
            person,
            aliases or [],
        )
        phones_items = phones_data.get("items", []) if isinstance(phones_data, dict) else []
        b = _summarize_theme_items("Téléphones", phones_items)
        if b:
            blocks.append(b)
    except Exception:
        logger.exception("scan_case_phones failed in person profile for %s / %s", case_id, person)

    # 2) autres thèmes (avec lien fort à la personne)
    theme_labels = {
        "iban": "IBAN / comptes bancaires",
        "emails": "Adresses e-mail",
        "plates": "Plaques d'immatriculation / véhicules",
        "ips": "Adresses IP",
        "telegram": "Comptes / pseudos Telegram",
        "whatsapp": "Comptes / numéros WhatsApp",
        "snapchat": "Comptes / pseudos Snapchat",
    }

    for theme, label in theme_labels.items():
        if theme not in SUPPORTED_THEMES:
            continue
        try:
            data = await asyncio.to_thread(
                scan_case_theme,
                case_id,
                person,
                theme,
                aliases or [],
                True,  # strict: on ne garde que ce qui est lié à la personne
            )
            items = data.get("items", []) if isinstance(data, dict) else []
            b = _summarize_theme_items(label, items)
            if b:
                blocks.append(b)
        except Exception:
            logger.exception(
                "scan_case_theme('%s') failed in person profile for %s / %s",
                theme,
                case_id,
                person,
            )

    if not blocks:
        return ""

    text = "\n\n".join(blocks)
    # on limite un peu la taille globale pour ne pas exploser le contexte
    return text[:8000]


def _summarize_person_evidence(
    tech: List[Dict[str, Any]],
    sci: List[Dict[str, Any]],
    stmt: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Produit à la fois :
      - un petit résumé structuré pour le LLM (texte),
      - des stats pour le front (counts par catégorie).

    tech : GAV / CRT / écoutes / géoloc / balise / sono
    sci  : ADN / balistique / empreintes / odorologie / forensic phone
    stmt : auditions / interrogatoires
    """

    def _count(cat: str, items: List[Dict[str, Any]]) -> int:
        return sum(1 for e in items if e.get("category") == cat)

    # Comptages procéduraux / techniques
    gav_count = _count("gav", tech)
    crt_count = _count("crt", tech)
    ecoutes_count = _count("ecoutes", tech)
    geoloc_count = _count("geoloc", tech)
    balise_count = _count("balise", tech)
    sono_count = _count("sono", tech)

    # Comptages scientifiques
    adn_count = _count("adn", sci)
    balistique_count = _count("balistique", sci)
    empreintes_count = _count("empreintes", sci)
    forensic_count = _count("forensic_phone", sci)
    odorologie_count = _count("odorologie", sci)

    # Comptages déclarations
    auditions_count = _count("audition", stmt)

    lines: List[str] = []

    lines.append("### Synthèse procédurale & technique (automatique)")
    if any([gav_count, crt_count, ecoutes_count, geoloc_count, balise_count, sono_count]):
        lines.append(
            f"- Garde à vue : {gav_count} occurrence(s)"
            f" — CRT : {crt_count} — Écoutes/interceptions : {ecoutes_count}"
        )
        lines.append(
            f"- Géolocalisation/bornage : {geoloc_count} — Balises véhicule : {balise_count}"
            f" — Sonorisations : {sono_count}"
        )
    else:
        lines.append(
            "- Aucun acte procédural/technique spécifique (GAV, CRT, écoutes…) "
            "n'a été détecté pour cette personne."
        )

    lines.append("")
    lines.append("### Synthèse expertises scientifiques (automatique)")
    if any([adn_count, balistique_count, empreintes_count, forensic_count, odorologie_count]):
        lines.append(
            f"- ADN : {adn_count} — Balistique : {balistique_count}"
            f" — Empreintes/traces : {empreintes_count}"
        )
        lines.append(
            f"- Forensique téléphone/PC : {forensic_count}"
            f" — Odorologie : {odorologie_count}"
        )
    else:
        lines.append(
            "- Aucune expertise scientifique (ADN, balistique, empreintes, etc.) "
            "clairement associée à cette personne n'a été détectée."
        )

    lines.append("")
    lines.append("### Synthèse déclarations / auditions (automatique)")
    if auditions_count:
        lines.append(f"- Auditions / interrogatoires / IPC : {auditions_count} acte(s) identifiés.")
    else:
        lines.append(
            "- Aucun PV d'audition / interrogatoire clairement rattaché à cette personne "
            "n'a été identifié via les patterns automatiques."
        )

    text_block = "\n".join(lines)

    return {
        "text": text_block,
        "stats": {
            "procedure": {
                "gav": gav_count,
                "crt": crt_count,
                "ecoutes": ecoutes_count,
                "geoloc": geoloc_count,
                "balise": balise_count,
                "sono": sono_count,
            },
            "science": {
                "adn": adn_count,
                "balistique": balistique_count,
                "empreintes": empreintes_count,
                "forensic_phone": forensic_count,
                "odorologie": odorologie_count,
            },
            "statements": {
                "auditions": auditions_count,
            },
        },
    }


# =========================================================
# Helpers : extraction robuste date / lieu de naissance
# =========================================================

_BIRTH_DATE_RE = re.compile(
    r"né[e]?\s+le\s+([0-3]?\d(?:[\/\.-][0-1]?\d[\/\.-]\d{2,4}|\s+[A-Za-zéèêàâôîûùçÉÈÊÀÂÔÎÛÙÇ]+\s+\d{2,4}))",
    re.IGNORECASE,
)
_BIRTH_PLACE_RE = re.compile(
    r"né[e]?\s+le\s+.+?\s+à\s+([A-ZÉÈÊÀÂÎÔÛÙÇ][^,\n;]+)",
    re.IGNORECASE,
)


def _extract_birth_info_from_snippets(snippets: List[Dict]) -> tuple[Optional[str], Optional[str]]:
    """
    Parcourt les snippets et essaie d'extraire une date et un lieu de naissance
    à partir des lignes contenant 'né le' / 'née le'.

    ⚠️ IMPORTANT :
    Cette fonction est pensée pour être appelée sur les snippets
    de type "identité" (première audition, grande identité, IPC).
    On évite ainsi de capter des dates d'événements quelconques.
    """
    dob: Optional[str] = None
    place: Optional[str] = None

    for sn in snippets or []:
        txt = (sn.get("text") or "")
        if "né" not in txt.lower():
            continue
        for line in txt.splitlines():
            lower = line.lower()
            if "né le" not in lower and "née le" not in lower:
                continue

            if dob is None:
                m_date = _BIRTH_DATE_RE.search(line)
                if m_date:
                    dob = m_date.group(1).strip(" .;,")

            if place is None:
                m_place = _BIRTH_PLACE_RE.search(line)
                if m_place:
                    place = m_place.group(1).strip(" .;,")

            if dob and place:
                return dob, place

    return dob, place


def _inject_birth_info_in_answer(answer: str, dob: Optional[str], place: Optional[str]) -> str:
    """
    Si on a pu extraire date / lieu de naissance, on remplace dans la réponse
    les lignes correspondantes si elles sont vides ou "Non mentionné".
    """
    if not answer:
        return answer

    text = answer

    if dob:
        if "Date de naissance" in text and ("Non mentionné" in text or dob not in text):
            def _repl_date(m: re.Match) -> str:
                return m.group(1) + dob
            text, _ = re.subn(
                r"(Date de naissance\s*:\s*)(.*)",
                _repl_date,
                text,
                count=1,
            )
    if place:
        if "Lieu de naissance" in text and ("Non mentionné" in text or place not in text):
            def _repl_place(m: re.Match) -> str:
                return m.group(1) + place
            text, _ = re.subn(
                r"(Lieu de naissance\s*:\s*)(.*)",
                _repl_place,
                text,
                count=1,
            )

    return text


# =========================================================
# Helpers : raffinage d'adresse via RAG ciblé
# =========================================================

def _build_address_context(snippets: List[Dict[str, Any]]) -> str:
    """
    Construit un petit contexte texte à partir des snippets d'adresse.
    """
    parts: List[str] = []
    for sn in snippets or []:
        txt = (sn.get("text") or "").strip()
        if not txt:
            continue
        meta = sn.get("meta") or {}
        src = meta.get("source") or "?"
        page = meta.get("page")
        if page is not None:
            header = f"[{src} p.{page}]"
        else:
            header = f"[{src}]"
        parts.append(header + "\n" + txt)

    ctx = "\n\n".join(parts)
    return ctx[:6000]


def _parse_address_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Essaie de récupérer un JSON { "adresse_principale": "...", "autres_adresses": [...], "sources": [...] }
    à partir de la sortie modèle (qui peut contenir du texte avant/après).
    """
    if not raw:
        return None
    raw = raw.strip()
    # tentative directe
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # fallback: on cherche le premier '{' et le dernier '}' et on tente
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        snippet = raw[start: end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _patch_address_in_answer(answer: str, addr_info: Dict[str, Any]) -> str:
    """
    Injecte l'adresse principale / autres adresses dans la fiche si le JSON est exploitable.
    On reste conservateur : si "adresse_principale" == "NON_MENTIONNE" ou vide, on ne force rien.
    """
    if not answer or not addr_info:
        return answer

    principale = (addr_info.get("adresse_principale") or "").strip()
    autres = addr_info.get("autres_adresses") or []

    # normalisation des autres adresses
    if isinstance(autres, str):
        autres_list = [autres.strip()] if autres.strip() else []
    elif isinstance(autres, list):
        autres_list = [str(a).strip() for a in autres if str(a).strip()]
    else:
        autres_list = []

    text = answer

    # Adresse principale
    if principale and principale.upper() != "NON_MENTIONNE":
        def _repl_main(m: re.Match) -> str:
            return m.group(1) + principale
        text, _ = re.subn(
            r"(-\s*Adresse principale\s*:\s*)(.*)",
            _repl_main,
            text,
            count=1,
        )

    # Autres adresses citées
    if autres_list:
        autres_str = "; ".join(autres_list)

        def _repl_other(m: re.Match) -> str:
            return m.group(1) + autres_str
        text, _ = re.subn(
            r"(-\s*Autres adresses citées\s*:\s*)(.*)",
            _repl_other,
            text,
            count=1,
        )

    return text


# ==============================
#  Helpers déterministes pour l'adresse
# ==============================

# motifs d'adresse dans un PV d'identité
_ADDRESS_LINE_RE = re.compile(
    r"(?:domicilié[e]?\s+(?:au|à|chez)\s+.+?|"
    r"demeurant\s+(?:au|à|chez)\s+.+?|"
    r"résidant\s+(?:au|à|chez)\s+.+?)",
    re.IGNORECASE,
)


def _extract_address_candidates_from_snippets(
    snippets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Parcourt les snippets (souvent PV d'identité) et extrait
    des candidats d'adresse sous forme de lignes qui commencent par
    'domicilié', 'demeurant', 'résidant'.

    Retourne une liste de dicts :
      { "adresse": "...", "source": "doc", "page": 3 }
    """
    candidates: List[Dict[str, Any]] = []
    for sn in snippets or []:
        txt = (sn.get("text") or "").replace("\n", " ")
        meta = sn.get("meta") or {}
        src = meta.get("source") or "?"
        page = meta.get("page")

        for m in _ADDRESS_LINE_RE.finditer(txt):
            addr_raw = m.group(0).strip()
            # petit nettoyage : on coupe à la première phrase si besoin
            addr_clean = re.split(r"[.;]", addr_raw, 1)[0].strip()
            if not addr_clean:
                continue
            candidates.append(
                {
                    "adresse": addr_clean,
                    "source": src,
                    "page": page,
                }
            )
    return candidates


def _summarize_address_candidates(
    candidates: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    À partir d'une liste de candidats d'adresse, fait un petit 'vote' :
      - adresse_principale = la plus fréquente
      - autres_adresses = les autres distinctes
    Retourne un dict prêt à être injecté dans la fiche :
      {
        "adresse_principale": "...",
        "autres_adresses": ["..."],
        "sources": ["doc p.3", "autre_doc p.2"]
      }
    """
    if not candidates:
        return None

    # comptage par adresse normalisée
    freq: Dict[str, int] = {}
    by_addr_sources: Dict[str, List[str]] = {}

    for c in candidates:
        addr = c["adresse"].strip()
        key = addr.lower()
        freq[key] = freq.get(key, 0) + 1

        label = c["source"]
        if c.get("page") is not None:
            label = f"{label} p.{c['page']}"
        by_addr_sources.setdefault(key, []).append(label)

    # adresse la plus fréquente
    best_key = max(freq.keys(), key=lambda k: freq[k])
    best_addr = None
    for c in candidates:
        if c["adresse"].lower().strip() == best_key:
            best_addr = c["adresse"].strip()
            break
    best_addr = best_addr or best_key

    autres_raw = sorted(
        {
            c["adresse"].strip()
            for c in candidates
            if c["adresse"].lower().strip() != best_key
        }
    )

    sources = []
    for k, labs in by_addr_sources.items():
        labs_unique = sorted(set(labs))
        if k == best_key:
            sources.extend(labs_unique[:3])
        else:
            sources.extend(labs_unique[:2])
    sources = sorted(set(sources))[:10]

    return {
        "adresse_principale": best_addr,
        "autres_adresses": autres_raw,
        "sources": sources,
    }


def _refine_address_via_rag(
    model: str,
    case_id: str,
    person: str,
    aliases: List[str],
    civil_snippets: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Utilise en priorité les PV d'identité (civil_snippets : première audition, grande identité, IPC)
    pour trouver la DOMICILIATION de la personne.
    En secours seulement : find_person_address_snippets (plus large).

    Retourne un petit JSON { "adresse_principale": ..., "autres_adresses": [...], "sources": [...] }
    ou None si rien d'exploitable.
    """
    addr_snippets: List[Dict[str, Any]] = []

    try:
        # 1) PRIORITÉ ABSOLUE : PV d'identité / grande identité / IPC
        if civil_snippets:
            addr_snippets = civil_snippets

        # 2) Si vraiment on n'a rien (affaire très pauvre), on tente le RAG adresse large
        if not addr_snippets:
            addr_snippets = person_retrieval.find_person_address_snippets(
                case_id,
                person,
                aliases=aliases,
                limit=20,
            )
    except FileNotFoundError:
        # affaire / index manquant → on laisse le haut niveau gérer
        return None
    except Exception:
        logger.exception("address retrieval failed for %s / %s", case_id, person)
        return None

    if not addr_snippets:
        return None

    ctx = _build_address_context(addr_snippets)

    prompt = f"""
Tu es un analyste judiciaire français.
On te donne des extraits de procédure concernant une personne : "{person}".

OBJECTIF : déterminer son adresse de DOMICILIATION, pas le lieu des faits.

RÈGLES TRÈS IMPORTANTES :
1) Tu CHERCHES UNIQUEMENT la domiciliation de la personne :
   - "domicilié à ..."
   - "demeurant à ..."
   - "résidant à ..."
   - "domicilié chez ..."
   - ou formulations équivalentes dans un PV d'identité (première audition / grande identité / IPC).

2) TU NE DOIS JAMAIS utiliser comme adresse de domiciliation :
   - les formulations de type :
     * "les faits se sont déroulés à ..."
     * "les faits se sont produits à ..."
     * "sur les lieux des faits ..."
     * "au lieu des faits ..."
     * "au lieu-dit ..."
     * "les policiers se rendent au ..."
   - ni les adresses qui décrivent :
     * uniquement le lieu de commission des faits,
     * un lieu de rendez-vous ou de surveillance,
     * le domicile clairement identifié d'une autre personne.

3) Si plusieurs adresses de domiciliation pour la même personne apparaissent :
   - "adresse_principale" = l'adresse la plus récente OU présentée comme actuelle.
   - les autres vont dans "autres_adresses".

SORTIE :
Tu dois renvoyer STRICTEMENT un JSON de la forme :

{{
  "adresse_principale": "texte exact de l'adresse ou 'NON_MENTIONNE'",
  "autres_adresses": ["...", "..."],
  "sources": ["nom_doc p.12", "PV première audition p.3"]
}}

- "adresse_principale" = "NON_MENTIONNE" si tu n'es pas sûr ou si aucune domiciliation claire n'apparaît.
- Pour les "sources", liste quelques références doc/page si tu peux, sinon [].

Les extraits suivants proviennent en priorité des PV d'identité (première audition, grande identité, IPC) :

---------------------
{ctx}
""".strip()

    try:
        # --- Anonymisation éventuelle du prompt avant appel LLM ---
        try:
            final_prompt_sanitized, privacy_meta = anonymize_prompt_for_model(model, prompt, case_id)
            logger.info(
                "address: privacy meta=%s",
                {
                    "sanitized": privacy_meta.get("sanitized"),
                    "reason": privacy_meta.get("reason"),
                    "orig_len": privacy_meta.get("original_len"),
                    "san_len": privacy_meta.get("sanitized_len"),
                    "repl": privacy_meta.get("replacements"),
                    "case_id": case_id,
                },
            )
        except Exception:
            logger.exception("address: anonymize_prompt_for_model failed, using original prompt")
            final_prompt_sanitized = prompt

        raw = ollama_client.generate_long(model, final_prompt_sanitized, max_tokens=600)
    except Exception:
        logger.exception("ollama_client.generate_long (address) failed for %s / %s", case_id, person)
        return None

    # --- Dé-pseudonymisation éventuelle du résultat avant parsing ---
    try:
        raw = deanonymize_text_for_case(raw or "", case_id)
    except Exception:
        logger.exception("address: deanonymize_text_for_case failed for raw result")

    addr_info = _parse_address_json(raw)
    if not addr_info:
        return None

    return addr_info


# =========================================================
# 1) ROUTE HISTORIQUE : scan / extraction thématique
# =========================================================

@router.post("/api/extract/scan/stream")
async def extract_scan_stream(
    case_id: str = Form(...),
    person: str = Form(...),
    theme: str = Form(...),
    model: str = Form(...),
    strict_person: int = Form(0),
):
    """
    Route utilisée par le front pour :
      - téléphones
      - iban
      - plaques
      - ip
    et qui streame dans le tableau à droite.
    """
    theme = (theme or "").lower().strip()
    if theme not in SUPPORTED_THEMES:
        raise HTTPException(
            status_code=400,
            detail=f"Thème non supporté. Supportés: {', '.join(sorted(SUPPORTED_THEMES))}",
        )

    _sanitize_case_id(case_id)
    _sanitize_person(person)

    strict = bool(strict_person)

    async def gen():
        # même protocole SSE qu’avant
        yield _sse({"type": "start", "task": f"scan:{theme}"})
        try:
            if theme == "phones":
                data = await asyncio.to_thread(
                    scan_case_phones,
                    case_id,
                    person,
                    [],  # aliases
                )
                items = data.get("items", []) if isinstance(data, dict) else []

                if strict:
                    # on ne garde que les occurrences très liées à la personne
                    filtered = []
                    for it in items:
                        occs = it.get("occurrences", [])
                        occs_linked = [
                            o for o in occs
                            if o.get("linked") or o.get("assoc") == "strong"
                        ]
                        if occs_linked:
                            it = dict(it)
                            it["occurrences"] = occs_linked
                            filtered.append(it)
                    items = filtered
            else:
                # tous les autres thèmes passent par l'extractor générique
                data = await asyncio.to_thread(
                    scan_case_theme,
                    case_id,
                    person,
                    theme,
                    [],
                    strict,
                )
                items = data.get("items", []) if isinstance(data, dict) else []
        except Exception:
            logger.exception(
                "scan_case_theme failed for %s / %s / %s", case_id, person, theme
            )
            yield _sse({"type": "error", "message": "erreur pendant le scan"})
            return

        # message de statut
        yield _sse(
            {
                "type": "status",
                "message": f"{len(items)} éléments '{theme}' trouvés",
            }
        )

        # stream vers le front
        for it in items[:500]:
            key = it.get("value") or it.get("key")
            occs = it.get("occurrences", [])
            strong = any(o.get("linked") or o.get("assoc") == "strong" for o in occs)
            if key:
                yield _sse(
                    {
                        "type": "token",
                        "content": f"- {key}{_assoc_badge(strong)} — {len(occs)} occurrence(s)\n",
                    }
                )

        sources = _assemble_sources_from_items(items)
        yield _sse({"type": "end", "sources": sources})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =========================================================
# 2) FICHE PERSONNE : synthèse complète + lien de téléchargement
# =========================================================

@router.post("/api/extract/person/profile")
async def extract_person_profile(
    case_id: str = Form(...),
    person: str = Form(...),
    model: str = Form(None),
    aliases: str = Form(""),
):
    """
    FICHE PERSONNE (version enrichie)

    Pipeline :
      1) Récupère un maximum de snippets centrés sur l'identité (PV grande identité,
         premières auditions, IPC) via person_retrieval.find_person_civil_snippets.
      2) Récupère un gros panel d'autres extraits où la personne apparaît
         (actes de procédure, auditions, rapports, écoutes, etc.) via
         person_retrieval.find_person_global_snippets.
      3) Appelle l'extracteur technique (_gather_technical_extractions) pour
         téléphonie, IBAN, emails, plaques, IP, messageries.
      3bis) Utilise person_evidence pour retrouver GAV / CRT / écoutes / géolocs /
         balises / sonos + expertises ADN / balistique / empreintes / etc. +
         auditions/interrogatoires.
      4) Extrait date / lieu de naissance à partir des PV d'identité uniquement.
      5) Construit un prompt riche et structuré pour produire une fiche "personne"
         très détaillée (identité, rôle, parcours procédural, relations, technique,
         points de vigilance).
      6) Optionnel : raffine la domiciliation via _refine_address_via_rag puis
         patche la réponse via _patch_address_in_answer.
      7) Sauvegarde la fiche dans backend/storage/cases/{case_id}/summaries/
         et renvoie l'URL de téléchargement + des stats procédurales/scientifiques.
    """
    # --- sanitisation de base ---
    case_id = _sanitize_case_id(case_id)
    person = _sanitize_person(person)
    model = model or os.getenv("OLLAMA_DEFAULT_MODEL", "mistral:latest")

    # Aliases : liste propre, sans doublons ni vides
    alias_list: List[str] = []
    if aliases:
        for raw in re.split(r"[;,/]", aliases):
            a = raw.strip()
            if not a:
                continue
            if a.lower() == person.lower():
                continue
            alias_list.append(a)
    # on dédoublonne en conservant l'ordre
    seen_alias = set()
    alias_clean: List[str] = []
    for a in alias_list:
        k = a.lower()
        if k in seen_alias:
            continue
        seen_alias.add(k)
        alias_clean.append(a)
    alias_list = alias_clean

    # --- 1) snippets identité & 2) snippets globaux ---
    try:
        # on élargit les limites pour exploiter davantage le dossier
        civil_snippets = person_retrieval.find_person_civil_snippets(
            case_id,
            person,
            aliases=alias_list,
            limit=80,   # 60 auparavant, on pousse un peu plus
        )
        global_snippets = person_retrieval.find_person_global_snippets(
            case_id,
            person,
            aliases=alias_list,
            limit=160,  # 80 auparavant, on élargit nettement
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Affaire introuvable ou non indexée")
    except Exception as e:
        logger.exception("person_retrieval failed for %s / %s", case_id, person)
        raise HTTPException(status_code=500, detail=f"Erreur recherche personne: {e}")

    def _fmt_snippet(sn: Dict[str, Any]) -> str:
        """
        Format lisible pour le LLM, avec titre de PV/rapport + fichier + page.
        """
        txt = (sn.get("text") or "").strip()
        meta = sn.get("meta") or {}
        unit_title = (meta.get("unit_title") or "").strip()
        source = (meta.get("source") or "").strip()
        page = meta.get("page")
        parts = []
        if unit_title:
            parts.append(unit_title)
        if source:
            parts.append(source)
        if page is not None:
            parts.append(f"p.{page}")
        header = " / ".join(parts) or "(source inconnue)"
        return f"[{header}]\n{txt}"

    civil_block = "\n\n---\n\n".join(_fmt_snippet(sn) for sn in (civil_snippets or []))
    if not civil_block:
        civil_block = "(aucun extrait explicitement centré sur l'identité n'a été retrouvé)"

    global_block = "\n\n---\n\n".join(_fmt_snippet(sn) for sn in (global_snippets or []))
    if not global_block:
        global_block = "(aucun autre extrait procédural n'a été retrouvé pour cette personne)"

    # --- 3) extractions techniques (phones / iban / emails / plaques / IP / messageries) ---
    technical_context = await _gather_technical_extractions(case_id, person, alias_list)

    # --- 3bis) Éléments procéduraux / scientifiques / déclarations via person_evidence ---
    try:
        tech_evid = find_person_procedure_tech(case_id, person, alias_list, limit=200)
        sci_evid = find_person_scientific(case_id, person, alias_list, limit=200)
        stmt_evid = find_person_statements(case_id, person, alias_list, limit=200)
    except FileNotFoundError:
        tech_evid = []
        sci_evid = []
        stmt_evid = []
    except Exception:
        logger.exception("person_evidence failed for %s / %s", case_id, person)
        tech_evid = []
        sci_evid = []
        stmt_evid = []

    evidence_summary = _summarize_person_evidence(tech_evid, sci_evid, stmt_evid)
    evidence_block = evidence_summary.get("text") if evidence_summary else ""

    # petit bloc d'instructions explicites pour le modèle
    auto_hint_block = ""
    if evidence_summary:
        stats = evidence_summary.get("stats") or {}
        proc = stats.get("procedure") or {}
        sci_s = stats.get("science") or {}
        stm = stats.get("statements") or {}

        hint_lines: List[str] = []
        hint_lines.append("INFORMATIONS AUTOMATIQUES À INTÉGRER DANS LA FICHE :")

        # Procédure / techniques
        proc_bits: List[str] = []
        if proc.get("gav"):
            proc_bits.append(f"{proc['gav']} garde(s) à vue")
        if proc.get("crt"):
            proc_bits.append(f"{proc['crt']} commission(s) rogatoire(s) technique(s)")
        if proc.get("ecoutes"):
            proc_bits.append(f"{proc['ecoutes']} interception(s) téléphonique(s)")
        if proc.get("geoloc"):
            proc_bits.append(f"{proc['geoloc']} acte(s) de géolocalisation / bornage")
        if proc.get("balise"):
            proc_bits.append(f"{proc['balise']} acte(s) de pose / suivi de balise")
        if proc.get("sono"):
            proc_bits.append(f"{proc['sono']} acte(s) de sonorisation")

        if proc_bits:
            hint_lines.append(
                "- Procédure & techniques : " + ", ".join(proc_bits) +
                " (à intégrer dans les sections 4 et 7 de la fiche)."
            )

        # Scientifique
        sci_bits: List[str] = []
        if sci_s.get("adn"):
            sci_bits.append(f"{sci_s['adn']} expertise(s) ADN")
        if sci_s.get("balistique"):
            sci_bits.append(f"{sci_s['balistique']} expertise(s) balistique(s)")
        if sci_s.get("empreintes"):
            sci_bits.append(f"{sci_s['empreintes']} acte(s) relatif(s) aux empreintes / traces")
        if sci_s.get("forensic_phone"):
            sci_bits.append(f"{sci_s['forensic_phone']} analyse(s) forensique(s) téléphone / PC")
        if sci_s.get("odorologie"):
            sci_bits.append(f"{sci_s['odorologie']} acte(s) d'odorologie")

        if sci_bits:
            hint_lines.append(
                "- Expertises scientifiques : " + ", ".join(sci_bits) +
                " (à intégrer dans les sections 4, 7 et dans les éléments techniques)."
            )

        # Déclarations
        if stm.get("auditions"):
            hint_lines.append(
                f"- Déclarations / auditions : {stm['auditions']} audition(s) / interrogatoire(s) "
                "(à intégrer dans la section 5 et dans le parcours procédural)."
            )

        if len(hint_lines) > 1:
            hint_lines.append(
                ">>> Ces informations doivent apparaître clairement dans la fiche, "
                "même si le reste du contexte est tronqué ou incomplet."
            )
            auto_hint_block = "\n".join(hint_lines) + "\n\n"

    # --- 4) date / lieu de naissance à partir des seuls PV d'identité ---
    dob, birth_place = _extract_birth_info_from_snippets(civil_snippets or [])

    birth_info_block = ""
    if dob or birth_place:
        lines = ["Informations critiques déjà repérées dans le dossier :"]
        if dob:
            lines.append(f"- Date de naissance (à recopier telle quelle) : {dob}")
        if birth_place:
            lines.append(f"- Lieu de naissance (à recopier tel quel) : {birth_place}")
        lines.append(
            ">>> Tu dois ABSOLUMENT utiliser ces valeurs dans la rubrique "
            "\"Identité / état civil\" de la fiche personne."
        )
        birth_info_block = "\n".join(lines) + "\n\n"

    # --- SECTION RÉSUMÉ PTS / SCIENTIFIQUE POUR LA FICHE ---
    summary_pts_block = ""
    if evidence_summary:
        s = evidence_summary["stats"]
        summary_pts_block = f"""
### Résumé PTS / Téléphonie / Scientifique (automatique)
- **Actes procéduraux & techniques** :
  GAV : {s['procedure']['gav']}, CRT : {s['procedure']['crt']},
  Écoutes/interceptions : {s['procedure']['ecoutes']},
  Géolocalisations/bornages : {s['procedure']['geoloc']},
  Balise(s) : {s['procedure']['balise']},
  Sonorisations : {s['procedure']['sono']}
- **Expertises scientifiques** :
  ADN : {s['science']['adn']}, Balistique : {s['science']['balistique']},
  Empreintes : {s['science']['empreintes']},
  Forensic téléphone/PC : {s['science']['forensic_phone']},
  Odorologie : {s['science']['odorologie']}
- **Déclarations & interrogatoires** :
  {s['statements']['auditions']} acte(s)
""".strip()

    # --- 5) Construction du gros contexte texte ---
    ctx_parts: List[str] = []

    ctx_parts.append(
        "=== Extraits centrés sur l'identité (PV grande identité, premières auditions, IPC) ===\n"
        f"{civil_block}"
    )

    # On insère le résumé PTS juste après l'identité
    if summary_pts_block:
        ctx_parts.append(summary_pts_block)

    ctx_parts.append(
        "=== Autres actes où la personne apparaît (PV, rapports, auditions, écoutes, etc.) ===\n"
        f"{global_block}"
    )

    if technical_context:
        ctx_parts.append(
            "=== Éléments techniques (téléphonie, finances, plaques, IP, messageries, etc.) ===\n"
            f"{technical_context}"
        )

    if evidence_block:
        ctx_parts.append(
            "=== Résumé automatique des actes concernant cette personne "
            "(GAV, écoutes, expertises, auditions) ===\n"
            f"{evidence_block}"
        )

    raw_context = "\n\n\n".join(ctx_parts)

    # on laisse la possibilité de monter assez haut en char pour les gros dossiers
    max_ctx_chars = int(os.getenv("PERSON_PROFILE_CTX_CHARS", "24000"))
    if len(raw_context) > max_ctx_chars:
        raw_context = raw_context[:max_ctx_chars] + "\n\n...[contexte tronqué pour tenir dans le modèle]"

    # --- Prompt FICHE PERSONNE très détaillé ---
    alias_str = ", ".join(alias_list) if alias_list else "(aucun alias fourni)"

    prompt = f"""
Tu es un assistant d'enquête judiciaire français, spécialisé dans la rédaction de fiches "personne"
à partir de dossiers pénaux volumineux.

Affaire : {case_id}
Personne concernée : "{person}"
Alias à traiter comme la même personne : {alias_str}

{birth_info_block}{auto_hint_block}TÂCHE :
Tu dois rédiger une FICHE PERSONNE complète, très détaillée et structurée en sections Markdown :

1. Résumé synthétique
   - En 10–15 lignes, explique qui est cette personne, son rôle dans l'affaire
     et la manière dont elle apparaît dans la procédure.

2. Identité / état civil
   - Nom(s), prénom(s), surnoms / alias.
   - Date de naissance, lieu de naissance (en reprenant exactement les formulations du dossier).
   - Nationalité, situation familiale, profession, éléments de parcours personnel.
   - Mentionne explicitement les éléments "non précisés" lorsqu'ils n'apparaissent pas dans le dossier.

3. Rôle dans le dossier
   - Qualité procédurale : mis(e) en cause, témoin, victime, autre.
   - Lien avec les faits principaux (faits initiaux, faits secondaires, contexte).
   - Évolutions éventuelles de ce rôle au fil de la procédure.

4. Parcours procédural de la personne
   - Présente, de façon chronologique, les principaux actes : auditions, GAV, perquisitions,
     confrontations, expertises, décisions, etc.
   - Pour chaque acte important : date, type d'acte, objet principal, documents concernés.
   - Mets en évidence les tournants (revirements, nouveaux éléments, changements de stratégie).

5. Déclarations et position de la personne
   - Position globale : reconnaît / conteste / minimise / se tait.
   - Évolutions de version, contradictions, points constants.
   - Éléments à charge et éléments à décharge venant de ses déclarations.

6. Relations et réseau
   - Proches, coauteurs possibles, complices, relations familiales ou professionnelles.
   - Qui parle de cette personne ? Avec quels propos (mise en cause, soutien, neutralité) ?
   - Synthétise les dynamiques de groupe (hiérarchie, influence, tensions, loyautés).

7. Éléments techniques et matériels
   - Téléphonie : lignes utilisées, bornages significatifs, contacts récurrents.
   - Financier : comptes bancaires, flux, schémas de versements.
   - Réseaux sociaux / messageries, plaques, IP, déplacements, tout élément matériel saillant.
   - Signale aussi les absences d'éléments techniques lorsqu'elles sont pertinentes.

8. Points de vigilance / questions ouvertes
   - Zones d'ombre sur l'identité ou le rôle de la personne.
   - Points à vérifier, incohérences possibles, éléments qui nécessiteraient approfondissement.
   - Hypothèses possibles, mais toujours présentées comme telles et strictement fondées
     sur les éléments du dossier.

RÈGLES :
- Tu restes STRICTEMENT fidèle aux informations présentes dans le contexte ci-dessous.
- Tu peux reformuler et structurer, mais tu n'inventes ni faits, ni dates, ni événements.
- Lorsque une information n'apparaît pas dans le contexte, indique-la comme "non précisé"
  plutôt que de l'imaginer.
- Utilise un style professionnel, neutre, lisible, adapté à un magistrat ou un enquêteur.
- Ta réponse peut être longue : n'hésite pas à détailler dès qu'il y a de la matière.

Contexte (extraits du dossier, déjà filtrés autour de cette personne) :
----------------------------------------------------------------------
{raw_context}
""".strip()

    # --- 6) Appel LLM principal ---
    max_tokens = int(os.getenv("PERSON_PROFILE_MAX_TOKENS", "3200"))
    try:
        # --- Anonymisation éventuelle du prompt avant appel LLM ---
        try:
            final_prompt_sanitized, privacy_meta = anonymize_prompt_for_model(model, prompt, case_id)
            logger.info(
                "person_profile: privacy meta=%s",
                {
                    "sanitized": privacy_meta.get("sanitized"),
                    "reason": privacy_meta.get("reason"),
                    "orig_len": privacy_meta.get("original_len"),
                    "san_len": privacy_meta.get("sanitized_len"),
                    "repl": privacy_meta.get("replacements"),
                    "case_id": case_id,
                },
            )
        except Exception:
            logger.exception("person_profile: anonymize_prompt_for_model failed, using original prompt")
            final_prompt_sanitized = prompt

        answer = ollama_client.generate_long(model, final_prompt_sanitized, max_tokens=max_tokens)
    except TypeError:
        # compat éventuelle si la signature de generate_long est différente
        answer = ollama_client.generate_long(model, final_prompt_sanitized)

    # Renforcement DOB / lieu si on les connaît
    # --- Dé-pseudonymisation éventuelle du résultat avant post‑processing ---
    try:
        answer = deanonymize_text_for_case(answer or "", case_id)
    except Exception:
        logger.exception("person_profile: deanonymize_text_for_case failed for answer")

    answer = _inject_birth_info_in_answer(answer, dob, birth_place)

    # Raffinement éventuel de l'adresse via RAG ciblé (si la fonction est dispo)
    try:
        addr_info = _refine_address_via_rag(model, case_id, person, alias_list, civil_snippets=civil_snippets or [])
        if addr_info:
            answer = _patch_address_in_answer(answer, addr_info)
    except Exception:
        logger.exception("address refinement failed for %s / %s", case_id, person)

    # --- 7) Sauvegarde de la fiche personne ---
    # le fichier contient déjà le texte dé-pseudonymisé
    download_url = _write_person_profile_file(case_id, person, answer)

    snippets_used = len(civil_snippets or []) + len(global_snippets or [])

    return {
        "ok": True,
        "person": person,
        "answer": answer,
        "snippets_used": snippets_used,
        "download_url": download_url,
        "evidence_stats": evidence_summary.get("stats") if evidence_summary else {},
    }
