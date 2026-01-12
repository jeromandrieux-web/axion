#!/usr/bin/env bash
set -euo pipefail
pkill -f "uvicorn.*backend.main:app" 2>/dev/null && echo "🛑 uvicorn stoppé" || echo "ℹ️ aucun uvicorn à arrêter"
