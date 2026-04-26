#!/usr/bin/env bash
# Install vLLM into an isolated virtual environment on an NVIDIA GPU box.
#
# Blackwell GPUs need CUDA >= 12.8 wheels. vLLM publishes prebuilt CUDA wheels,
# so this path should not require building from source.
#
# Usage:
#   ./scripts/install_vllm.sh
#   source "$HOME/vllm-venv/bin/activate"
#
# Customization:
#   VLLM_VENV=$HOME/.venvs/vllm ./scripts/install_vllm.sh
#   UV_TORCH_BACKEND=cu129 ./scripts/install_vllm.sh

set -euo pipefail

VLLM_VENV="${VLLM_VENV:-${HOME}/vllm-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python not found: ${PYTHON_BIN}"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: uv install did not put uv on PATH."
    echo "Try opening a new shell, or run:"
    echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
    exit 1
fi

if [ -d "${VLLM_VENV}" ] && { [ ! -x "${VLLM_VENV}/bin/python" ] || [ ! -r "${VLLM_VENV}/bin/activate" ]; }; then
    echo "Removing incomplete virtualenv: ${VLLM_VENV}"
    rm -rf "${VLLM_VENV}"
fi

if [ ! -d "${VLLM_VENV}" ]; then
    echo "Creating uv virtualenv: ${VLLM_VENV}"
    uv venv --python "${PYTHON_BIN}" "${VLLM_VENV}"
fi

if [ ! -r "${VLLM_VENV}/bin/activate" ]; then
    echo "ERROR: virtualenv activation script not found: ${VLLM_VENV}/bin/activate"
    echo "Try recreating the env manually:"
    echo "  rm -rf \"${VLLM_VENV}\""
    echo "  uv venv --python \"${PYTHON_BIN}\" \"${VLLM_VENV}\""
    exit 1
fi

# shellcheck disable=SC1091
source "${VLLM_VENV}/bin/activate"

echo "Installing vLLM"
uv pip install -U pip wheel setuptools
uv pip install vllm --torch-backend="${UV_TORCH_BACKEND}"

echo
echo "Done. Activate with:"
echo "  source \"${VLLM_VENV}/bin/activate\""
echo
echo "Verify with:"
echo "  python -m vllm.entrypoints.openai.api_server --help | head"
