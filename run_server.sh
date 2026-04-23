#!/usr/bin/env bash
# Start llama-cpp-python's OpenAI-compatible server with Qwen3.6-35B-A3B-GGUF
# on GPU.  Customize env vars as needed:
#
#   MODEL_PATH  — explicit .gguf path (default: auto-discover from HF cache)
#   N_CTX       — context length to allocate (default: 65536)
#   PORT        — server port (default: 8000)
#   HOST        — bind address (default: 127.0.0.1)
#   KV_TYPE     — KV cache quant, e.g. q8_0 or q4_0 (default: q8_0)
#   N_GPU_LAYERS — -1 offloads everything to GPU (default: -1)
#
# Run in a tmux pane and leave it up; point longcot_mini_eval.py at
# http://$HOST:$PORT/v1 in another pane.

set -euo pipefail

N_CTX="${N_CTX:-65536}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
KV_TYPE="${KV_TYPE:-q8_0}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"

# Map KV_TYPE string → ggml_type integer (what llama-cpp-python's server expects).
# Accepts either a named type or an integer directly.
case "${KV_TYPE}" in
    f32|F32)   KV_INT=0 ;;
    f16|F16)   KV_INT=1 ;;
    q4_0|Q4_0) KV_INT=2 ;;
    q4_1|Q4_1) KV_INT=3 ;;
    q5_0|Q5_0) KV_INT=6 ;;
    q5_1|Q5_1) KV_INT=7 ;;
    q8_0|Q8_0) KV_INT=8 ;;
    [0-9]*)    KV_INT="${KV_TYPE}" ;;
    *)
        echo "ERROR: Unknown KV_TYPE='${KV_TYPE}'. Use f16/q8_0/q4_0 or an integer."
        exit 1
        ;;
esac

if [ -z "${MODEL_PATH:-}" ]; then
    # Search a few common locations for the GGUF file.
    SEARCH_DIRS=(
        "${HOME}/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF"
        "${HOME}/models/qwen3.6-gguf"
        "${HOME}/models"
        "${HOME}/huggingface"
        "${HOME}"
    )
    for dir in "${SEARCH_DIRS[@]}"; do
        if [ -d "${dir}" ]; then
            found="$(find "${dir}" -name '*Q4_K_M*.gguf' 2>/dev/null | head -1)"
            if [ -n "${found}" ]; then
                MODEL_PATH="${found}"
                break
            fi
        fi
    done
fi

if [ -z "${MODEL_PATH:-}" ] || [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: could not find Qwen3.6 GGUF."
    echo "Download it first:"
    echo "  huggingface-cli download unsloth/Qwen3.6-35B-A3B-GGUF \\"
    echo "      --include '*Q4_K_M*' --local-dir ~/models/qwen3.6-gguf"
    echo "Or set MODEL_PATH=/path/to/the.gguf explicitly and re-run."
    exit 1
fi

echo "Starting llama-cpp-python server"
echo "  model      = ${MODEL_PATH}"
echo "  n_ctx      = ${N_CTX}"
echo "  host:port  = ${HOST}:${PORT}"
echo "  kv_type    = ${KV_TYPE}  (ggml type=${KV_INT})"
echo "  gpu_layers = ${N_GPU_LAYERS}"
echo

# Pick a Python launcher.  We prefer `uv run python` so we pick up the managed
# venv without requiring it to be activated.  Fall back to python3 / python.
if command -v uv >/dev/null 2>&1; then
    PY=(uv run python)
elif command -v python3 >/dev/null 2>&1; then
    PY=(python3)
elif command -v python >/dev/null 2>&1; then
    PY=(python)
else
    echo "ERROR: no python found on PATH."
    exit 1
fi

# Put the venv's bundled nvidia lib dirs on LD_LIBRARY_PATH so
# llama-cpp-python can dlopen libcudart.so.12 / libcublas.so.12 / etc.
NVIDIA_LIBS="$("${PY[@]}" -c "
import os, site
sp = site.getsitepackages()[0]
nv = os.path.join(sp, 'nvidia')
paths = []
if os.path.isdir(nv):
    for d in os.listdir(nv):
        lib = os.path.join(nv, d, 'lib')
        if os.path.isdir(lib):
            paths.append(lib)
print(':'.join(paths))
" 2>/dev/null || true)"

if [ -n "${NVIDIA_LIBS}" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
    echo "  LD_LIBRARY_PATH prepended with: ${NVIDIA_LIBS}"
    echo
fi

exec "${PY[@]}" -m llama_cpp.server \
    --model "${MODEL_PATH}" \
    --n_gpu_layers "${N_GPU_LAYERS}" \
    --n_ctx "${N_CTX}" \
    --flash_attn True \
    --type_k "${KV_INT}" \
    --type_v "${KV_INT}" \
    --port "${PORT}" \
    --host "${HOST}"
