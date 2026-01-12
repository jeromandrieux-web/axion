#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Choix Python (préférence 3.11)
if command -v python3.11 >/dev/null 2>&1; then PYBIN=python3.11
elif command -v python3 >/dev/null 2>&1; then PYBIN=python3
else echo "❌ Python3.11 requis"; exit 1; fi

[ -d .venv ] || "$PYBIN" -m venv .venv
. .venv/bin/activate

pip install -U pip wheel
pip install -r requirements.txt

echo "✅ venv activé ($(python -V 2>&1))"
