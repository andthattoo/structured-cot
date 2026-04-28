#!/usr/bin/env bash
# Run one Terminal-Bench task with the local structured-cot agent.
#
# Defaults assume a patched native llama-server is already running on port 8000:
#   REASONING_FORMAT=deepseek BACKGROUND=1 ./run_llama_server.sh
#
# Compare free vs reasoning-grammar mode:
#   GRAMMAR_MODE=none ./scripts/run_terminal_bench_smoke.sh
#   GRAMMAR_MODE=reasoning ./scripts/run_terminal_bench_smoke.sh

set -euo pipefail

DATASET="${DATASET:-terminal-bench-core==head}"
TASK_ID="${TASK_ID:-hello-world}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-ggml-org/Qwen3.6-27B-GGUF}"
GRAMMAR_MODE="${GRAMMAR_MODE:-none}"
MAX_TURNS="${MAX_TURNS:-40}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-180}"
TB_BIN="${TB_BIN:-tb}"
INSTALL_TB="${INSTALL_TB:-0}"

if ! command -v "${TB_BIN}" >/dev/null 2>&1; then
    if [ "${INSTALL_TB}" != "1" ]; then
        echo "ERROR: '${TB_BIN}' not found."
        echo "Install Terminal-Bench with:"
        echo "  uv tool install terminal-bench"
        echo
        echo "Or let this script install it:"
        echo "  INSTALL_TB=1 ./scripts/run_terminal_bench_smoke.sh"
        exit 1
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "ERROR: uv is required to install terminal-bench automatically."
        exit 1
    fi
    uv tool install terminal-bench
    if ! command -v "${TB_BIN}" >/dev/null 2>&1; then
        if [ -x "${HOME}/.local/bin/${TB_BIN}" ]; then
            TB_BIN="${HOME}/.local/bin/${TB_BIN}"
        else
            echo "ERROR: terminal-bench installed, but '${TB_BIN}' is still not on PATH."
            echo "Try adding ~/.local/bin to PATH, then rerun."
            exit 1
        fi
    fi
fi

echo "Running Terminal-Bench smoke"
echo "  dataset      = ${DATASET}"
echo "  task_id      = ${TASK_ID}"
echo "  base_url     = ${BASE_URL}"
echo "  model        = ${MODEL}"
echo "  grammar_mode = ${GRAMMAR_MODE}"
echo "  max_turns    = ${MAX_TURNS}"
echo

PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}" "${TB_BIN}" run \
    --dataset "${DATASET}" \
    --agent-import-path terminal_bench_structured_cot_agent:StructuredCotTerminalAgent \
    --task-id "${TASK_ID}" \
    --agent-kwarg "base_url=${BASE_URL}" \
    --agent-kwarg "model=${MODEL}" \
    --agent-kwarg "grammar_mode=${GRAMMAR_MODE}" \
    --agent-kwarg "max_turns=${MAX_TURNS}" \
    --agent-kwarg "max_tokens=${MAX_TOKENS}" \
    --agent-kwarg "command_timeout_sec=${COMMAND_TIMEOUT_SEC}"
