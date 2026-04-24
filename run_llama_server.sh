#!/usr/bin/env bash
# Start native llama.cpp's OpenAI-compatible llama-server.
#
# This is the faster path for long FREE / PROMPT_TERSE generations.  It uses
# the native C++ server instead of llama-cpp-python, and enables llama.cpp's
# default speculative decoding preset by default.
#
# Customize env vars as needed:
#
#   LLAMA_SERVER_BIN — path to llama-server (default: first on PATH)
#   HF_REPO          — Hugging Face GGUF repo (default: ggml-org/Qwen3.6-27B-GGUF)
#   HF_FILE          — optional specific GGUF filename in HF_REPO
#   MODEL_PATH       — optional local .gguf path; overrides HF_REPO/HF_FILE
#   N_CTX            — context length (default: 32768)
#   PORT             — server port (default: 8000)
#   HOST             — bind address (default: 127.0.0.1)
#   N_GPU_LAYERS     — GPU layers to offload (default: 999)
#   FLASH_ATTN       — on/off/auto, or 1/0 for on/off (default: on)
#   SPEC_DEFAULT     — 1 to pass --spec-default (default: 1)
#   REASONING_FORMAT — none keeps <think> in message.content (default: none)
#   KV_TYPE          — optional KV cache type, e.g. q8_0 or q4_0
#   BACKGROUND       — 1 to start with nohup and return (default: 0)
#   LOG_FILE         — log path for BACKGROUND=1 (default: server.log)
#   PID_FILE         — pid path for BACKGROUND=1 (default: server.pid)
#
# Extra CLI args can be appended after the script:
#   ./run_llama_server.sh --parallel 2
#   BACKGROUND=1 ./run_llama_server.sh

set -euo pipefail

LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-llama-server}"
HF_REPO="${HF_REPO:-ggml-org/Qwen3.6-27B-GGUF}"
N_CTX="${N_CTX:-32768}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
FLASH_ATTN="${FLASH_ATTN:-on}"
SPEC_DEFAULT="${SPEC_DEFAULT:-1}"
REASONING_FORMAT="${REASONING_FORMAT:-none}"
BACKGROUND="${BACKGROUND:-0}"
LOG_FILE="${LOG_FILE:-server.log}"
PID_FILE="${PID_FILE:-server.pid}"

if ! command -v "${LLAMA_SERVER_BIN}" >/dev/null 2>&1; then
    echo "ERROR: llama-server not found: ${LLAMA_SERVER_BIN}"
    echo
    echo "Install native llama.cpp with CUDA on Ubuntu:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y git cmake build-essential libcurl4-openssl-dev"
    echo "  git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp"
    echo "  cmake -S ~/llama.cpp -B ~/llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release"
    echo "  cmake --build ~/llama.cpp/build --config Release -j"
    echo "  export PATH=\"\$HOME/llama.cpp/build/bin:\$PATH\""
    echo
    echo "Then rerun this script."
    exit 1
fi

MODEL_ARGS=()
if [ -n "${MODEL_PATH:-}" ]; then
    if [ ! -f "${MODEL_PATH}" ]; then
        echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"
        exit 1
    fi
    MODEL_ARGS=(-m "${MODEL_PATH}")
else
    MODEL_ARGS=(-hf "${HF_REPO}")
    if [ -n "${HF_FILE:-}" ]; then
        MODEL_ARGS+=(-hff "${HF_FILE}")
    fi
fi

ARGS=(
    "${MODEL_ARGS[@]}"
    --host "${HOST}"
    --port "${PORT}"
    -c "${N_CTX}"
    -ngl "${N_GPU_LAYERS}"
)

case "${FLASH_ATTN}" in
    1|on|ON|true|TRUE)       FLASH_ATTN_ARG="on" ;;
    0|off|OFF|false|FALSE)   FLASH_ATTN_ARG="off" ;;
    auto|AUTO)               FLASH_ATTN_ARG="auto" ;;
    *)
        echo "ERROR: FLASH_ATTN must be on/off/auto or 1/0, got '${FLASH_ATTN}'"
        exit 1
        ;;
esac

if [ "${FLASH_ATTN_ARG}" != "auto" ]; then
    ARGS+=(--flash-attn "${FLASH_ATTN_ARG}")
fi

if [ "${SPEC_DEFAULT}" = "1" ]; then
    ARGS+=(--spec-default)
fi

if [ -n "${REASONING_FORMAT}" ]; then
    ARGS+=(--reasoning-format "${REASONING_FORMAT}")
fi

if [ -n "${KV_TYPE:-}" ]; then
    ARGS+=(--cache-type-k "${KV_TYPE}" --cache-type-v "${KV_TYPE}")
fi

echo "Starting native llama-server"
if [ -n "${MODEL_PATH:-}" ]; then
    echo "  model      = ${MODEL_PATH}"
else
    echo "  hf_repo    = ${HF_REPO}${HF_FILE:+ / ${HF_FILE}}"
fi
echo "  n_ctx      = ${N_CTX}"
echo "  host:port  = ${HOST}:${PORT}"
echo "  gpu_layers = ${N_GPU_LAYERS}"
echo "  flash_attn = ${FLASH_ATTN_ARG}"
echo "  spec       = ${SPEC_DEFAULT}"
echo "  reasoning  = ${REASONING_FORMAT:-auto}"
if [ -n "${KV_TYPE:-}" ]; then
    echo "  kv_type    = ${KV_TYPE}"
fi
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
    nohup "${LLAMA_SERVER_BIN}" "${ARGS[@]}" "$@" > "${LOG_FILE}" 2>&1 &
    pid=$!
    echo "${pid}" > "${PID_FILE}"
    echo "Started llama-server PID ${pid}"
    echo "Wait for readiness:"
    echo "  tail -f ${LOG_FILE}"
    echo "  curl http://${HOST}:${PORT}/v1/models"
    exit 0
fi

exec "${LLAMA_SERVER_BIN}" "${ARGS[@]}" "$@"
