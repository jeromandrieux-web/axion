#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/activate.sh >/dev/null
. .venv/bin/activate
export PYTHONPATH="$PWD"
set -a; [ -f .env ] && . .env; set +a
exec uvicorn backend.main:app --host 127.0.0.1 --port 8080 --reload
