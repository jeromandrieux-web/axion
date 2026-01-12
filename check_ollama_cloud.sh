#!/bin/zsh
# --- Vérification Ollama Cloud (DeepSeek) ---

HOST="http://127.0.0.1:11434"
MODEL="deepseek-v3.1:671b-cloud"

say() { print -P "$*"; }

# 1) Clé présente dans l'environnement Launch Services (session utilisateur)
API_KEY="$(launchctl getenv OLLAMA_API_KEY)"
if [ -z "$API_KEY" ]; then
  say "❌  Clé OLLAMA_API_KEY absente (session)."
  say "   → Lance :  launchctl setenv OLLAMA_API_KEY \"sk-...\"  puis redémarre Ollama (open -a Ollama)"
  exit 1
fi

# 2) Daemon Ollama joignable
if ! curl -s -m 5 "$HOST/api/tags" >/dev/null; then
  say "❌  Ollama n'est pas joignable sur $HOST"
  say "   → Démarre-le :  open -a Ollama"
  exit 1
fi

# 3) Modèle cloud présent (manifest only = OK)
if ! curl -s "$HOST/api/tags" | grep -q "$MODEL"; then
  say "ℹ️  Modèle $MODEL non listé. Tentative de pull du manifest…"
  if ! ollama pull "$MODEL" >/dev/null 2>&1; then
    say "❌  Échec du pull de $MODEL. Vérifie ta connexion."
    exit 1
  fi
fi

# 4) Appel de génération direct (non-stream) pour diagnostiquer précisément
JSON_PAYLOAD='{"model":"'"$MODEL"'","prompt":"Réponds uniquement par: OK","stream":false,"options":{"temperature":0}}'
RESP="$(curl -s -m 25 -H "Content-Type: application/json" -d "$JSON_PAYLOAD" "$HOST/api/generate")"

# 5) Analyse des erreurs fréquentes
if echo "$RESP" | grep -qi '"error"'; then
  say "❌  Erreur de l'API Ollama:"
  say "$(echo "$RESP" | sed 's/\\n/\n/g')"
  if echo "$RESP" | grep -qi 'unauthorized\|invalid\|auth'; then
    say "→ Ton OLLAMA_API_KEY semble invalide/non reçu par le daemon."
    say "  - Assure-toi d'avoir fait : launchctl setenv OLLAMA_API_KEY \"sk-...\""
    say "  - Puis redémarre Ollama :  pkill ollama && open -a Ollama"
  fi
  exit 1
fi

# 6) Validation de la réponse attendue
if echo "$RESP" | grep -q '"response"\s*:\s*"OK"'; then
  say "✅  DeepSeek Cloud opérationnel (clé et réseau OK)"
  exit 0
else
  say "⚠️  Réponse inattendue. Sortie brute :"
  say "$RESP"
  say "→ Si pas d'erreur, c'est quand même joignable (probable variation de réponse)."
  exit 0
fi

