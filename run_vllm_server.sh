#!/usr/bin/env bash
# Start vLLM's OpenAI-compatible server for the tool+grammar repro matrix.
#
# Defaults use a documented Qwen tool-call setup: Qwen2.5 with the hermes
# parser. The repro script can then hit this server with `--server vllm`.
#
# Usage:
#   source "$HOME/vllm-venv/bin/activate"
#   BACKGROUND=1 ./run_vllm_server.sh
#   python3 llama_tool_grammar_repro.py --server vllm --base-url http://127.0.0.1:8001/v1
#
# Customization:
#   MODEL=Qwen/Qwen2.5-14B-Instruct ./run_vllm_server.sh
#   TOOL_CALL_PARSER=hermes ./run_vllm_server.sh
#   PORT=8001 BACKGROUND=1 ./run_vllm_server.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-1}"
STRUCTURED_OUTPUTS_BACKEND="${STRUCTURED_OUTPUTS_BACKEND:-auto}"
BACKGROUND="${BACKGROUND:-0}"
LOG_FILE="${LOG_FILE:-vllm_server.log}"
PID_FILE="${PID_FILE:-vllm_server.pid}"

if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vllm not found on PATH."
    echo "Install/activate it first:"
    echo "  ./scripts/install_vllm.sh"
    echo "  source \"\$HOME/vllm-venv/bin/activate\""
    exit 1
fi

ARGS=(
    serve "${MODEL}"
    --host "${HOST}"
    --port "${PORT}"
    --max-model-len "${MAX_MODEL_LEN}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --structured-outputs-config.backend "${STRUCTURED_OUTPUTS_BACKEND}"
)

if [ "${ENABLE_AUTO_TOOL_CHOICE}" = "1" ]; then
    ARGS+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

echo "Starting vLLM OpenAI server"
echo "  model       = ${MODEL}"
echo "  host:port   = ${HOST}:${PORT}"
echo "  max_len     = ${MAX_MODEL_LEN}"
echo "  gpu_mem     = ${GPU_MEMORY_UTILIZATION}"
echo "  tool_parser = ${ENABLE_AUTO_TOOL_CHOICE:+${TOOL_CALL_PARSER}}"
echo "  structured  = ${STRUCTURED_OUTPUTS_BACKEND}"
echo

if [ "${BACKGROUND}" = "1" ]; then
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "Server already appears to be running with PID $(cat "${PID_FILE}")"
        echo "  log = ${LOG_FILE}"
        echo "  pid = ${PID_FILE}"
        exit 0
    fi

    echo "Starting in background"
    echo "  log = ${LOG_FILE}"
    echo "  pid = ${PID_FILE}"
    nohup vllm "${ARGS[@]}" "$@" > "${LOG_FILE}" 2>&1 &
    pid=$!
    echo "${pid}" > "${PID_FILE}"
    echo "Started vLLM PID ${pid}"
    echo "Wait for readiness:"
    echo "  tail -f ${LOG_FILE}"
    echo "  curl http://${HOST}:${PORT}/v1/models"
    exit 0
fi

exec vllm "${ARGS[@]}" "$@"
