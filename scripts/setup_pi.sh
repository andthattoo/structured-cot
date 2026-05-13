#!/usr/bin/env bash
# One-shot bootstrap for the Pi + SGLang + grammar-proxy stack.
# Idempotent: safe to re-run on a fresh GPU box.
#
# What it does:
#   1. Install Pi (https://pi.dev) if not on PATH
#   2. Install aiohttp (required by scripts/grammar_proxy.py)
#   3. Write ~/.pi/agent/models.json pointing at the local grammar proxy
#      (which forwards to SGLang on 127.0.0.1:30000)
#
# Env vars (optional):
#   PROXY_URL   provider baseUrl Pi will hit (default: http://127.0.0.1:30001/v1)
#               — set to http://127.0.0.1:30000/v1 to bypass the grammar proxy
#   MODEL       model id passed to SGLang        (default: Qwen/Qwen3.6-27B)
#
# Usage:
#   bash scripts/setup_pi.sh

set -euo pipefail

PROXY_URL="${PROXY_URL:-http://127.0.0.1:30001/v1}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"

echo "[setup] target config:"
echo "        provider baseUrl = $PROXY_URL"
echo "        model            = $MODEL"
echo

# ----- 1. Pi --------------------------------------------------------------
if command -v pi >/dev/null 2>&1; then
    echo "[setup] Pi already installed:"
    pi --version 2>/dev/null | head -1 | sed 's/^/        /'
else
    echo "[setup] installing Pi via official installer..."
    curl -fsSL https://pi.dev/install.sh | sh
    if ! command -v pi >/dev/null 2>&1; then
        echo
        echo "[setup] Pi installed but 'pi' is not on PATH in this shell."
        echo "        Open a new shell (or 'source ~/.bashrc' / '~/.zshrc') and re-run:"
        echo "        bash scripts/setup_pi.sh"
        exit 1
    fi
    echo "[setup] Pi installed: $(pi --version 2>/dev/null | head -1)"
fi

# ----- 2. aiohttp for the grammar proxy -----------------------------------
if python3 -c "import aiohttp" 2>/dev/null; then
    echo "[setup] aiohttp already installed"
else
    echo "[setup] installing aiohttp for grammar proxy..."
    pip install --quiet aiohttp
    echo "[setup] aiohttp installed"
fi

# ----- 3. ~/.pi/agent/models.json ----------------------------------------
mkdir -p "$HOME/.pi/agent"
MODELS_PATH="$HOME/.pi/agent/models.json"

cat > "$MODELS_PATH" <<EOF
{
  "providers": {
    "sglang": {
      "baseUrl": "$PROXY_URL",
      "api": "openai-completions",
      "apiKey": "not-needed",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "thinkingFormat": "qwen-chat-template"
      },
      "models": [
        {
          "id": "$MODEL",
          "name": "Qwen 3.6 27B (Local SGLang)",
          "reasoning": true,
          "contextWindow": 32768,
          "maxTokens": 8192
        }
      ]
    }
  }
}
EOF
chmod 0600 "$MODELS_PATH"
echo "[setup] wrote $MODELS_PATH"

echo
echo "[setup] done. Next steps (one pane each):"
echo
echo "  pane 1 (SGLang):       bash run_sglang_server.sh"
echo "  pane 2 (grammar proxy): uv run python scripts/grammar_proxy.py"
echo "  pane 3 (Pi):            pi --provider sglang --model \"$MODEL\""
echo
echo "  To bypass the grammar proxy for an A/B test:"
echo "    PROXY_URL=http://127.0.0.1:30000/v1 bash scripts/setup_pi.sh"
