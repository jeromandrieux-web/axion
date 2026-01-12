# backend/person_retrieval.py
# Recherche et extraction de snippets centrés sur une personne dans une affaire.
# Auteur: Jax

from __future__ import annotations
import os, json, re, unicodedata, logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from . import retrieval  # utilise ton retrieval.py (BM25 + embeddings)

logger = logging.getLogger(__name__)

STORAGE = Path(os.getenv("STORAGE_DIR", "backend/storage")).resolve()

# ------------ motifs généraux "état civil" ------------
CIVIL_PATTERNS = [
    r"né[e]?\s+le\s+\d{1,2}",
    r"née\s+le\s+\d{1,2}",
    r"né\s+le\s+\d{1,2}",
    r"domicilié[e]?\s+(?:au|à)\s",
    r"demeurant\s+(?:au|à)\s",
    r"résidant\s+(?:au|à)\s",
    r"de nationalité",
    r"de profession",
    r"célibataire|marié[e]?|divorcé[e]?|veuf|veuve",
    r"fils de|fille de",
]
CIVIL_REGEXES = [re.compile(p, re.IGNORECASE) for p in CIVIL_PATTERNS]

# ------------ motifs plus ciblés "adresse" ------------
ADDRESS_PATTERNS = [
    r"domicilié[e]?\s+(?:au|à|chez)\s",
    r"demeurant\s+(?:au|à|chez)\s",
    r"résidant\s+(?:au|à|chez)\s",
    r"adresse\s+(?:principale|actuelle|au)\s",
]
ADDRESS_REGEXES = [re.compile(p, re.IGNORECASE) for p in ADDRESS_PATTERNS]

AUDITION_KEYWORDS = [
    "première audition",
    "1ere audition",
    "1ère audition",
    "grande identité",
    "interrogatoire de première comparution",
    "ipc",
]

PROCEDURAL_PATTERNS = [
    r"réquisition",
    r"extraction téléphonique",
    r"exploitation du téléphone",
    r"perquisition",
    r"commission rogatoire",
    r"écoutes téléphoniques?",
    r"exploitation de scellé",
    r"facturation détaillée",
]
PROC_REGEXES = [re.compile(p, re.IGNORECASE) for p in PROCEDURAL_PATTERNS]


# =====================================================
# Utils
# =====================================================

def _norm(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).replace("\u00A0", " ")
    s = "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_name_patterns(person: str, aliases: List[str]) -> List[re.Pattern]:
    """
    Construit des patterns pour 'Jean DUPONT', 'DUPONT Jean', etc.
    sur la base du nom complet et des alias.
    """
    patterns: List[re.Pattern] = []

    def _mk_for(name: str) -> None:
        base = _norm(name)
        if not base:
            return
        parts = [p for p in re.split(r"\s+", base) if p]
        if len(parts) >= 2:
            first = parts[0]
            last = parts[-1]
            # DUPONT Jean ... ou Jean DUPONT (dans un rayon de 40 caractères)
            patterns.append(
                re.compile(rf"\b{last}\b.{0,40}\b{first}\b", re.IGNORECASE)
            )
            patterns.append(
                re.compile(rf"\b{first}\b.{0,40}\b{last}\b", re.IGNORECASE)
            )
        elif len(parts) == 1:
            # fallback très léger : un seul token complet
            p = parts[0]
            patterns.append(re.compile(rf"\b{p}\b", re.IGNORECASE))

    if person:
        _mk_for(person)
    for al in aliases:
        _mk_for(al)

    return patterns


def _person_in_text(text: str, person: str, aliases: List[str]) -> bool:
    """
    Version durcie : on cherche des occurrences explicites du couple
    NOM + prénom (ou alias) dans le texte normalisé.
    """
    if not person and not aliases:
        return False
    ntxt = _norm(text)
    if not ntxt:
        return False

    patterns = _build_name_patterns(person, aliases)
    if not patterns:
        return False

    return any(p.search(ntxt) for p in patterns)

def _score_chunk(text: str) -> int:
    """
    Score utilisé pour les bouts très 'état civil' / auditions d'identité.
    On favorise les grandes identités, on pénalise un peu les gros actes de procédure.
    """
    score = 0
    for rx in CIVIL_REGEXES:
        if rx.search(text):
            score += 4
    low = text.lower()
    if any(k in low for k in AUDITION_KEYWORDS):
        score += 8
    for rx in PROC_REGEXES:
        if rx.search(text):
            score -= 3
    if 80 < len(text) < 900:
        score += 1
    return score


def load_case_chunks(case_id: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    casedir = STORAGE / "cases" / case_id
    bm25_path = casedir / "bm25.json"
    if not bm25_path.exists():
        # bug corrigé (ancienne faute de frappe)
        raise FileNotFoundError(f"bm25.json introuvable pour {case_id}")
    data = json.loads(bm25_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks") or []
    sources = data.get("sources") or []
    return chunks, sources


# =====================================================
# Fallbacks RAG
# =====================================================

def _fallback_rag_identity(case_id: str, person: str) -> List[Dict[str, Any]]:
    """
    Si on ne voit pas de PV d'identité en lisant brut bm25.json,
    on force le RAG à chercher un PV d'identité.
    """
    query = (
        f"{person} première audition identité né le demeurant "
        f"interrogatoire de première comparution IPC grande identité"
    )
    try:
        rag_hits = retrieval.retrieve_hybrid(case_id, query, top_k=30)
    except Exception:
        logger.exception("fallback RAG identity failed")
        return []
    out: List[Dict[str, Any]] = []
    for text, meta in rag_hits:
        out.append({"text": text, "meta": meta})
    return out


def _fallback_rag_global(case_id: str, person: str) -> List[Dict[str, Any]]:
    """
    Fallback plus large : on va chercher tout ce qui concerne la personne
    (auditions, perquisitions, écoutes, finances, etc.).
    """
    query = (
        f"{person} audition perquisition réquisition téléphonique "
        f"facturation détaillée comptes bancaires mouvements financiers "
        f"écoutes téléphoniques adresses IP réquisitions bancaires"
    )
    try:
        rag_hits = retrieval.retrieve_hybrid(case_id, query, top_k=60)
    except Exception:
        logger.exception("fallback RAG global failed")
        return []
    out: List[Dict[str, Any]] = []
    for text, meta in rag_hits:
        out.append({"text": text, "meta": meta})
    return out


def _fallback_rag_address(case_id: str, person: str) -> List[Dict[str, Any]]:
    """
    Fallback ciblé sur l'adresse / domiciliation quand bm25.json ne donne rien de probant.
    """
    query = (
        f"{person} domicilié demeurant résidant adresse principale adresse actuelle "
        f"demeurant à résidant à domicilié à"
    )
    try:
        rag_hits = retrieval.retrieve_hybrid(case_id, query, top_k=40)
    except Exception:
        logger.exception("fallback RAG address failed")
        return []
    out: List[Dict[str, Any]] = []
    for text, meta in rag_hits:
        out.append({"text": text, "meta": meta})
    return out


# =====================================================
# Snippets centrés état civil / identité
# =====================================================

def find_person_civil_snippets(
    case_id: str,
    person: str,
    *,
    aliases: Optional[List[str]] = None,
    limit: int = 40,
) -> List[Dict[str, Any]]:
    """
    Focussé sur les 'grandes identités' et PV proches de l'état civil.
    Utilisé pour aller chercher la vraie fiche d'identité dans la procédure.
    """
    aliases = aliases or []
    chunks, sources = load_case_chunks(case_id)
    scored: List[Tuple[int, int, Dict[str, Any]]] = []

    for idx, txt in enumerate(chunks):
        if not txt:
            continue
        if not _person_in_text(txt, person, aliases):
            continue

        normed = _norm(txt)
        sc = _score_chunk(normed)
        meta = sources[idx] if idx < len(sources) else {}
        scored.append((sc, idx, {"text": txt, "meta": meta}))

        # on ajoute le contexte autour des mentions d'audition / grande identité
        if any(k in normed.lower() for k in AUDITION_KEYWORDS):
            if idx - 1 >= 0:
                prev_txt = chunks[idx - 1]
                prev_meta = sources[idx - 1] if idx - 1 < len(sources) else {}
                scored.append(
                    (sc - 1, idx - 1, {"text": prev_txt, "meta": prev_meta})
                )
            if idx + 1 < len(chunks):
                next_txt = chunks[idx + 1]
                next_meta = sources[idx + 1] if idx + 1 < len(sources) else {}
                scored.append(
                    (sc - 1, idx + 1, {"text": next_txt, "meta": next_meta})
                )

    # ordre décroissant
    scored.sort(key=lambda x: (-x[0], x[1]))

    # si on n’a rien avec un vrai bonus audition → fallback RAG
    has_audition_like = any(sc >= 8 for sc, _, _ in scored)
    if not has_audition_like:
        logger.info("no obvious audition PV found in bm25.json, using RAG fallback")
        rag_snippets = _fallback_rag_identity(case_id, person)
        if rag_snippets:
            return rag_snippets[:limit]

    seen_idx = set()
    final: List[Dict[str, Any]] = []
    for sc, idx, d in scored:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        final.append(d)
        if len(final) >= limit:
            break

    return final


# =====================================================
# Snippets “global” : tous les actes où la personne apparaît
# =====================================================

def find_person_global_snippets(
    case_id: str,
    person: str,
    *,
    aliases: Optional[List[str]] = None,
    limit: int = 80,
) -> List[Dict[str, Any]]:
    """
    Version plus large : on prend tous les bouts de dossier où la personne
    apparaît (auditions, perquisitions, écoutes, finances, etc.).
    Cette fonction sert à alimenter la fiche de synthèse PERSONNE.
    """
    aliases = aliases or []
    chunks, sources = load_case_chunks(case_id)
    scored: List[Tuple[float, int, Dict[str, Any]]] = []

    for idx, txt in enumerate(chunks):
        if not txt:
            continue
        if not _person_in_text(txt, person, aliases):
            continue

        normed = _norm(txt)
        low = normed.lower()
        score = 0.0

        # léger bonus si présence d'éléments d'état civil dedans
        for rx in CIVIL_REGEXES:
            if rx.search(normed):
                score += 2.0

        # bonus audition / interrogatoire
        if any(k in low for k in AUDITION_KEYWORDS):
            score += 3.0

        # ici, contrairement à _score_chunk, on donne un BONUS sur les actes
        # de procédure (perquisitions, écoutes, réquisitions...)
        for rx in PROC_REGEXES:
            if rx.search(normed):
                score += 2.0

        # longueur raisonnable
        if 80 < len(normed) < 1200:
            score += 1.0

        meta = sources[idx] if idx < len(sources) else {}
        scored.append((score, idx, {"text": txt, "meta": meta}))

    # si on ne trouve rien dans bm25.json → fallback RAG plus large
    if not scored:
        logger.info("no person snippets found in bm25.json, using global RAG fallback")
        rag_snippets = _fallback_rag_global(case_id, person)
        return rag_snippets[:limit] if rag_snippets else []

    # tri décroissant score, puis par index
    scored.sort(key=lambda x: (-x[0], x[1]))

    seen_idx = set()
    final: List[Dict[str, Any]] = []
    for score, idx, d in scored:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        final.append(d)
        if len(final) >= limit:
            break

    return final


# =====================================================
# Nouveau : snippets centrés “adresse / domiciliation”
# =====================================================

def _score_address_chunk(text: str) -> float:
    """
    Score spécifique pour repérer les lignes de type :
    'né le ... à ..., de nationalité ..., domicilié à ...'
    On favorise les mentions d'adresse proches de l'état civil,
    on évite de trop valoriser les actes purement techniques.
    """
    score = 0.0
    low = text.lower()

    # présence explicite de motifs d'adresse
    for rx in ADDRESS_REGEXES:
        if rx.search(text):
            score += 8.0

    # combo avec motifs d'état civil → typiquement PV d'identité
    if "né le" in low or "née le" in low or "né(e) le" in low:
        score += 4.0

    for rx in CIVIL_REGEXES:
        if rx.search(text):
            score += 2.0

    # petit malus si trop "procédure technique"
    for rx in PROC_REGEXES:
        if rx.search(text):
            score -= 1.0

    # longueur raisonnable
    if 50 < len(text) < 900:
        score += 1.0

    return score


def find_person_address_snippets(
    case_id: str,
    person: str,
    *,
    aliases: Optional[List[str]] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Snippets focalisés sur la DOMICILIATION de la personne.
    Idée : isoler les bouts les plus probables contenant l'adresse principale
    (domicilié à, demeurant à, résidant à).

    Utilisation typique :
      - alimenter un petit prompt spécialisé pour extraire l'adresse principale
        de la fiche personne.
    """
    aliases = aliases or []
    chunks, sources = load_case_chunks(case_id)

    scored: List[Tuple[float, int, Dict[str, Any]]] = []

    for idx, txt in enumerate(chunks):
        if not txt:
            continue
        if not _person_in_text(txt, person, aliases):
            continue

        normed = _norm(txt)

        # on ne garde que les textes qui ont au moins un motif d'adresse
        if not any(rx.search(normed) for rx in ADDRESS_REGEXES):
            continue

        sc = _score_address_chunk(normed)
        if sc <= 0:
            continue

        meta = sources[idx] if idx < len(sources) else {}
        scored.append((sc, idx, {"text": txt, "meta": meta}))

    # si rien d'évident dans bm25 → fallback RAG dédié adresse
    if not scored:
        logger.info("no address snippets found in bm25.json, using address RAG fallback")
        rag_snippets = _fallback_rag_address(case_id, person)
        return rag_snippets[:limit] if rag_snippets else []

    scored.sort(key=lambda x: (-x[0], x[1]))

    seen_idx = set()
    final: List[Dict[str, Any]] = []
    for sc, idx, d in scored:
        if idx in seen_idx:
            continue
        seen_idx.add(idx)
        final.append(d)
        if len(final) >= limit:
            break

    return final
