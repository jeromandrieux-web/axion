# backend/utils_normalize.py
# Utilitaires d'extraction & normalisation (téléphones, IBAN, emails, IP, plaques).
# Auteur: Jax

"""
Utilitaires d'extraction & normalisation (téléphones, IBAN, emails, IP, plaques).
- Regex "larges" mais raisonnables
- Normalisation téléphonique E.164 (par défaut FR +33)
- Petits helpers pour association par proximité à une personne
"""

from __future__ import annotations
import re
from typing import Iterable, List, Tuple, Dict, Any, Optional

# =========================
# Regex de détection
# =========================

# Téléphone : version renforcée pour les écritures FR / PV
# gère :
#   +33612345678
#   +33 6 12 34 56 78
#   +33 (0)6 12 34 56 78
#   0033 6 12 34 56 78
#   06 12 34 56 78
#   01.23.45.67.89
PHONE_RE = re.compile(
    r"""
    (?:(?<=\s)|(?<=^)|(?<=\())          # début plausible
    (?:
        # formes avec indicatif FR
        (?:(?:\+|00)\s?33)              # +33 ou 0033
        \s*(?:\(0\))?                   # (0) optionnel
        \s*[1-9]                        # 1er chiffre du numéro sans 0
        (?:[.\-\s]*\d{2}){4}            # 4 groupes de 2
        |
        # formes nationales FR
        0[1-9](?:[.\-\s]*\d{2}){4}
    )
    (?=\D|$)
    """,
    re.VERBOSE,
)

# IBAN (générique)
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")

# Email (simple mais efficace)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# IPv4 (basique)
IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)

# Plaques FR : SIV + (optionnel) ancien système FNI
PLATE_SIV_RE = re.compile(r"\b[A-Z]{2}-\d{3}-[A-Z]{2}\b")
PLATE_FNI_RE = re.compile(r"\b\d{1,4}\s?[A-Z]{1,3}\s?\d{2,3}\b")
PLATE_RE = re.compile(rf"(?:{PLATE_SIV_RE.pattern}|{PLATE_FNI_RE.pattern})")


# =========================
# Normalisations
# =========================

def normalize_phone(raw: str, default_cc: str = "+33") -> str:
    """
    Normalise un numéro en E.164 basique.
    Règles :
    - Supprime tout ce qui est décoratif (espaces, ., -, (0))
    - 00XX... -> +XX...
    - +33 (0)6... -> +336...
    - 0XXXXXXXXX (FR) -> +33XXXXXXXXX si default_cc='+33'
    - 33XXXXXXXXX -> +33XXXXXXXXX
    - Retourne '' si longueur trop courte (<7 chiffres après normalisation)
    """
    if not raw:
        return ""

    s = raw.strip()

    # enlever le (0) qui traîne souvent dans les PV
    s = s.replace("(0)", "")
    s = s.replace(" ", "").replace(".", "").replace("-", "")

    # on garde trace si l'utilisateur avait mis un +
    has_plus = s.startswith("+")

    # si ça commence par 00 → +…
    if s.startswith("00"):
        s = "+" + s[2:]
    elif has_plus:
        # on laisse comme ça pour l'instant
        pass
    else:
        # pas de +, pas de 00
        # 1) numéro FR classique 0X...
        if s.startswith("0"):
            cc = default_cc if default_cc.startswith("+") else f"+{default_cc}"
            s = cc + s[1:]
        # 2) commence par 33 → +33
        elif s.startswith("33"):
            s = "+" + s
        else:
            # pas d'indicatif évident : si pas trop long on préfixe
            if len(s) <= 10 and default_cc:
                cc = default_cc if default_cc.startswith("+") else f"+{default_cc}"
                s = cc + s
            else:
                s = "+" + s

    # maintenant on doit avoir du +33612345678 ou équivalent
    if not s.startswith("+"):
        s = "+" + s

    core = s[1:]
    if len(core) < 7:
        return ""

    return s


def normalize_iban(raw: str) -> str:
    if not raw:
        return ""
    s = re.sub(r"\s+", "", raw).upper()
    # Filtre très basique
    return s if IBAN_RE.search(s) else ""


def normalize_email(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    return s.lower() if EMAIL_RE.search(s) else ""


def normalize_plate(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().upper().replace(" ", "")
    # Remet au format SIV si possible (AA-123-AA)
    m = re.match(r"^([A-Z]{2})(\d{3})([A-Z]{2})$", s)
    if m:
      return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # Sinon, garde la forme originale trimée (pour FNI)
    return raw.strip().upper()


# =========================
# Association par proximité
# =========================

def window_contains_person(
    text: str, person: str, span: Tuple[int, int], window_chars: int = 80
) -> bool:
    """Retourne True si `person` apparaît dans une fenêtre ±window_chars autour du match."""
    if not person:
        return False
    i0, i1 = span
    a = max(0, i0 - window_chars)
    b = min(len(text), i1 + window_chars)
    return person.lower() in text[a:b].lower()


# =========================
# Extraction générique
# =========================

def extract_with_regex(
    text: str,
    regex: re.Pattern,
    person: Optional[str] = None,
    normalizer: Optional[Any] = None,
    default_cc: str = "+33",
) -> List[Dict[str, Any]]:
    """
    Extrait toutes les occurrences d'un regex, normalise éventuellement,
    et marque l'association (strong/indirect) selon la proximité à `person`.
    Retour :
      [{'raw':..., 'value':..., 'assoc': 'strong'|'indirect', 'span':(i0,i1)}...]
    """
    out: List[Dict[str, Any]] = []
    for m in regex.finditer(text or ""):
        raw = m.group(0)
        val = raw
        if normalizer:
            if normalizer is normalize_phone:
                val = normalizer(raw, default_cc=default_cc)
            else:
                val = normalizer(raw)
            if not val:
                continue
        assoc = "strong" if (person and window_contains_person(text, person, m.span())) else "indirect"
        out.append({"raw": raw, "value": val, "assoc": assoc, "span": m.span()})
    return out


def stable_dedup(items: Iterable[Dict[str, Any]], key=lambda x: x["value"]) -> List[Dict[str, Any]]:
    """Dé-duplication en conservant l'ordre d'apparition."""
    seen = set()
    res: List[Dict[str, Any]] = []
    for it in items:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        res.append(it)
    return res


# =========================
# Helpers d'affichage
# =========================

def format_citation(source: Optional[str], page: Optional[int]) -> str:
    """Rend 'source p.X' si page fournie, sinon 'source'."""
    src = source or "?"
    return f"{src} p.{page}" if page is not None else src


# =========================
# Si exécute en direct : mini-tests manuels
# =========================

if __name__ == "__main__":
    txt = """
    Jean Dupont a contacté le 06 12 34 56 78 et +33 6 12 34 56 78.
    Autres: +33 (0)6 12 34 56 78, 0033 1 23 45 67 89, 01.98.76.54.32
    IBAN: FR76 3000 6000 0112 3456 7890 189
    Email: jean.dupont@example.com
    IP: 192.168.0.1
    Plaque: AB-123-CD et 123 ABC 75
    """

    print("Phones:")
    for e in extract_with_regex(txt, PHONE_RE, person="Jean Dupont", normalizer=normalize_phone):
        print(e)

    print("\nIBANs:")
    for e in extract_with_regex(txt, IBAN_RE, person="Jean Dupont", normalizer=normalize_iban):
        print(e)

    print("\nEmails:")
    for e in extract_with_regex(txt, EMAIL_RE, person="Jean Dupont", normalizer=normalize_email):
        print(e)

    print("\nIPs:")
    for e in extract_with_regex(txt, IP_RE, person="Jean Dupont"):
        print(e)

    print("\nPlates:")
    for e in extract_with_regex(txt, PLATE_RE, person="Jean Dupont", normalizer=normalize_plate):
        print(e)
