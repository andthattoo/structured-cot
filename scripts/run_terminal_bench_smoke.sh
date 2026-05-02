#!/usr/bin/env bash
# Run Terminal-Bench with the local structured-cot agent.
#
# Defaults assume a patched native llama-server is already running on port 8000:
#   REASONING_FORMAT=deepseek BACKGROUND=1 ./run_llama_server.sh
#
# Compare free vs reasoning-grammar modes:
#   GRAMMAR_MODE=none ./scripts/run_terminal_bench_smoke.sh
#   GRAMMAR_MODE=reasoning ./scripts/run_terminal_bench_smoke.sh  # STEP only
#   GRAMMAR_MODE=step_status ./scripts/run_terminal_bench_smoke.sh # STEP/STATUS
#   GRAMMAR_MODE=phase ./scripts/run_terminal_bench_smoke.sh      # PHASE/CHECK/NEXT
#   GRAMMAR_MODE=dsl ./scripts/run_terminal_bench_smoke.sh        # PLAN/STATE/RISK/NEXT
#   TOOL_MODE=qwen_xml GRAMMAR_MODE=none ./scripts/run_terminal_bench_smoke.sh
#       # prompt-only compact DSL + Qwen XML tool calls for QwenXML LoRA eval
#
# Run the full dataset by omitting --task-id:
#   TASK_ID=all GRAMMAR_MODE=none ./scripts/run_terminal_bench_smoke.sh
#
# Full runs default to serial execution because one llama-server usually cannot
# handle Terminal-Bench's default 4 concurrent trials.

set -euo pipefail

DATASET="${DATASET:-terminal-bench-core==0.1.1}"
TASK_ID="${TASK_ID:-hello-world}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-ggml-org/Qwen3.6-27B-GGUF}"
GRAMMAR_MODE="${GRAMMAR_MODE:-none}"
TOOL_MODE="${TOOL_MODE:-native}"
MQE_MODE="${MQE_MODE:-none}"
MQE_ENCODER_DIR="${MQE_ENCODER_DIR:-driaforall/code-state-embedding}"
MQE_CRITIC_DIR="${MQE_CRITIC_DIR:-driaforall/code-mqe-critic-actionchoice}"
MQE_DEVICE="${MQE_DEVICE:-auto}"
MQE_CANDIDATES="${MQE_CANDIDATES:-4}"
MQE_TEMPERATURE="${MQE_TEMPERATURE:-0.7}"
MQE_TOP_P="${MQE_TOP_P:-0.95}"
MAX_TURNS="${MAX_TURNS:-40}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
COMMAND_TIMEOUT_SEC="${COMMAND_TIMEOUT_SEC:-180}"
N_CONCURRENT="${N_CONCURRENT:-1}"
N_TASKS="${N_TASKS:-}"
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
echo "  tool_mode    = ${TOOL_MODE}"
echo "  mqe_mode     = ${MQE_MODE}"
if [ "${MQE_MODE}" != "none" ]; then
    echo "  mqe_encoder  = ${MQE_ENCODER_DIR}"
    echo "  mqe_critic   = ${MQE_CRITIC_DIR}"
    echo "  mqe_k        = ${MQE_CANDIDATES}"
fi
echo "  max_turns    = ${MAX_TURNS}"
echo "  n_concurrent = ${N_CONCURRENT}"
if [ -n "${N_TASKS}" ]; then
    echo "  n_tasks      = ${N_TASKS}"
fi
echo

TASK_ARGS=()
if [ -n "${TASK_ID}" ] && [ "${TASK_ID}" != "all" ] && [ "${TASK_ID}" != "*" ]; then
    TASK_ARGS=(--task-id "${TASK_ID}")
fi
if [ -n "${N_TASKS}" ]; then
    TASK_ARGS+=(--n-tasks "${N_TASKS}")
fi

PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}" "${TB_BIN}" run \
    --dataset "${DATASET}" \
    --n-concurrent "${N_CONCURRENT}" \
    --agent-import-path terminal_bench_structured_cot_agent:StructuredCotTerminalAgent \
    "${TASK_ARGS[@]}" \
    --agent-kwarg "base_url=${BASE_URL}" \
    --agent-kwarg "model=${MODEL}" \
    --agent-kwarg "grammar_mode=${GRAMMAR_MODE}" \
    --agent-kwarg "tool_mode=${TOOL_MODE}" \
    --agent-kwarg "mqe_mode=${MQE_MODE}" \
    --agent-kwarg "mqe_encoder_dir=${MQE_ENCODER_DIR}" \
    --agent-kwarg "mqe_critic_dir=${MQE_CRITIC_DIR}" \
    --agent-kwarg "mqe_device=${MQE_DEVICE}" \
    --agent-kwarg "mqe_candidates=${MQE_CANDIDATES}" \
    --agent-kwarg "mqe_temperature=${MQE_TEMPERATURE}" \
    --agent-kwarg "mqe_top_p=${MQE_TOP_P}" \
    --agent-kwarg "max_turns=${MAX_TURNS}" \
    --agent-kwarg "max_tokens=${MAX_TOKENS}" \
    --agent-kwarg "command_timeout_sec=${COMMAND_TIMEOUT_SEC}"
