# backend/identity_mapper.py
# Pseudonymisation désactivée : module "bouchon" qui ne modifie plus rien.
# Historique : ce module réalisait une pseudonymisation réversible des noms / téléphones.
# Version actuelle : il garde la même API publique, mais ne transforme plus les textes.

from __future__ import annotations

from typing import Dict, Any, Tuple, Optional


def anonymize_prompt_for_model(
    model: Optional[str],
    prompt: str,
    case_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Ancien comportement : pseudonymiser les noms/téléphones dans le prompt
    avant envoi à certains modèles "cloud", et retourner aussi des méta-données.

    Nouveau comportement (rollback complet) :
        - renvoie le prompt tel quel,
        - renvoie un meta minimal avec sanitized=False.
    """
    meta: Dict[str, Any] = {
        "sanitized": False,
        "reason": "pseudonymisation_desactivee",
        "original_len": len(prompt),
        "sanitized_len": len(prompt),
        "replacements": 0,
        "model": model,
        "case_id": case_id,
    }
    return prompt, meta


def deanonymize_text_for_case(
    text: str,
    case_id: Optional[str] = None,
) -> str:
    """
    Ancien comportement : remplacer les tags PERS_001, TEL_002, etc. par
    les valeurs originales stockées dans un mapping par affaire.

    Nouveau comportement (rollback complet) :
        - renvoie le texte tel quel, sans aucune transformation.
    """
    return text
