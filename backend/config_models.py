# backend/config_models.py
# Gestion centralisée des modèles autorisés et par défaut.
# Auteur: Jax

"""
Gestion centralisée des modèles autorisés et par défaut.

Ce module lit automatiquement les variables d'environnement
et fournit des valeurs propres et cohérentes pour tout le backend.
"""

import os

def get_allowed_models() -> list[str]:
    """
    Renvoie la liste des modèles autorisés, déduite de AXION_ALLOWED_MODELS
    ou des valeurs par défaut.
    """
    raw = os.getenv(
        "AXION_ALLOWED_MODELS",
        "mistral:latest,deepseek-v3.1:671b-cloud"  # valeurs par défaut
    )
    allowed = [m.strip() for m in raw.split(",") if m.strip()]
    # Évite les doublons tout en gardant l'ordre
    seen = set()
    models = []
    for m in allowed:
        if m not in seen:
            seen.add(m)
            models.append(m)
    return models


def get_default_model() -> str:
    """
    Renvoie le modèle par défaut, en lisant OLLAMA_DEFAULT_MODEL
    ou en prenant le premier autorisé.
    """
    env_default = os.getenv("OLLAMA_DEFAULT_MODEL")
    if env_default:
        return env_default.strip()

    allowed = get_allowed_models()
    return allowed[0] if allowed else "mistral:latest"


def is_model_allowed(name: str) -> bool:
    """Vérifie si un modèle donné est autorisé."""
    if not name:
        return False
    return name in get_allowed_models()


# Petit helper pour afficher joliment au lancement (optionnel)
if __name__ == "__main__":
    print("📦 Modèles autorisés :", get_allowed_models())
    print("⭐ Modèle par défaut :", get_default_model())

