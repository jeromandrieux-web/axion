#!/bin/zsh

# Dossier de ton projet (change-le si besoin)
APP_DIR="$HOME/axion"

cd "$APP_DIR" || exit 1

# 🧠 Initialisation comme dans run_dev.sh

# (Optionnel mais utile) crée/actualise le venv si ton scripts/activate.sh existe
if [ -f "scripts/activate.sh" ]; then
  bash scripts/activate.sh >/dev/null 2>&1
fi

# Activer le venv
if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
else
  echo "⚠️ Environnement virtuel .venv introuvable dans $APP_DIR"
  exit 1
fi

# Même truc que run_dev.sh : PYTHONPATH + .env
export PYTHONPATH="$PWD"

set -a
[ -f .env ] && source .env
set +a

# 🚀 Démarrer le serveur si pas déjà en route (port 8000)
if ! lsof -i:8000 >/dev/null 2>&1; then
  echo "🚀 Démarrage du backend FastAPI sur 127.0.0.1:8000..."
  uvicorn backend.main:app --host 127.0.0.1 --port 8000 >/tmp/assistant_ia.log 2>&1 &
fi

# ⏳ Attendre que le serveur soit prêt (max ~20 secondes)
echo "⏳ Attente de la disponibilité du serveur..."
for i in {1..40}; do
  if nc -z 127.0.0.1 8000 2>/dev/null; then
    echo "✅ Serveur prêt."
    break
  fi
  sleep 0.5
done

# 🌐 Ouvrir Safari sur l'interface
open -a "Safari" "http://127.0.0.1:8000"
