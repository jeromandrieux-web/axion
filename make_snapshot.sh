#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Axion"
SNAPDIR="$HOME/Axion_snapshots"
TS="$(date '+%Y%m%d_%H%M%S')"
OUT="$SNAPDIR/axion_snapshot_$TS"

mkdir -p "$OUT"

echo "📦 Snapshot vers: $OUT"

# 1) Code & config (sans les gros dossiers)
# On copie tout sauf: .venv, backend/storage/cases (option: on garde la structure et meta, pas les raw)
rsync -a --delete \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.mypy_cache' \
  --exclude '.DS_Store' \
  --exclude 'backend/storage/cases/*/raw' \
  "$ROOT/" "$OUT/repo/"

# 2) Env & dépendances
cp -f "$ROOT/.env" "$OUT/.env"
cp -f "$ROOT/requirements.txt" "$OUT/requirements.txt"
if [ -d "$ROOT/.venv" ]; then
  source "$ROOT/.venv/bin/activate"
  pip freeze > "$OUT/requirements.lock.txt" || true
fi

# 3) Inventaire modèles Ollama (si ollama dispo)
if command -v ollama >/dev/null 2>&1; then
  ollama list > "$OUT/ollama_models.txt" || true
fi

# 4) Infos git (si repo initialisé)
if [ -d "$ROOT/.git" ]; then
  (cd "$ROOT" && git status --porcelain) > "$OUT/git_status.txt" || true
  (cd "$ROOT" && git rev-parse HEAD) > "$OUT/git_commit.txt" || true
fi

# 5) Archive compressée
mkdir -p "$SNAPDIR"
tar -C "$SNAPDIR" -czf "$SNAPDIR/synapse_V2_snapshot_$TS.tgz" "synapse_V2_snapshot_$TS" 2>/dev/null || true

echo "✅ Snapshot terminé."
echo "• Dossier: $OUT"
echo "• Archive: $SNAPDIR/axion_snapshot_$TS.tgz"
