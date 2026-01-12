# backend/power_utils.py
# Gestion simple de la mise en veille pendant les tâches longues
# Auteur: Jax

from __future__ import annotations

import logging
import platform
import subprocess
from typing import Optional, Any


def prevent_sleep(reason: str = "", logger: Optional[logging.Logger] = None) -> Optional[Any]:
    """
    Empêche la mise en veille de la machine pendant une tâche longue.

    - Sous macOS : lance `caffeinate` en arrière-plan et retourne le process.
    - Sous autres OS : ne fait rien et retourne None.

    Le "token" retourné doit être repassé à restore_sleep() à la fin.
    """
    logger = logger or logging.getLogger("axion.power_utils")
    system = platform.system().lower()

    if system == "darwin":
        # macOS : on utilise caffeinate
        try:
            # -d : empêche l'affichage de dormir
            # -i : empêche le système de dormir
            # -m : empêche le disque de dormir
            # -s : empêche le système de dormir à cause d'inactivité réseau
            # -u : déclare une activité utilisateur
            proc = subprocess.Popen(
                ["caffeinate", "-dimsu"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("prevent_sleep: lancé caffeinate pour tâche longue (%s)", reason or "sans raison")
            return proc
        except Exception:
            logger.exception("prevent_sleep: échec du lancement de caffeinate")
            return None

    # Autres OS : no-op (on ne tente rien de spécial)
    logger.info("prevent_sleep: OS %s non géré, no-op (raison=%s)", system, reason)
    return None


def restore_sleep(token: Optional[Any], logger: Optional[logging.Logger] = None) -> None:
    """
    Restaure le comportement normal de mise en veille.

    - Si `token` est un process (cas macOS/caffeinate), on le termine proprement.
    - Sinon, ne fait rien.
    """
    logger = logger or logging.getLogger("axion.power_utils")
    if token is None:
        return

    # Cas typique : token = subprocess.Popen(...)
    try:
        if hasattr(token, "terminate"):
            token.terminate()
            logger.info("restore_sleep: arrêt de caffeinate / process veille")
        else:
            logger.debug("restore_sleep: token sans terminate(), no-op")
    except Exception:
        logger.exception("restore_sleep: échec lors de l'arrêt du process caf feinate/token")
