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
#   TORCH_INDEX=https://download.pytorch.org/whl/cu129 ./scripts/install_vllm.sh

set -euo pipefail

VLLM_VENV="${VLLM_VENV:-${HOME}/vllm-venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python not found: ${PYTHON_BIN}"
    exit 1
fi

if [ ! -d "${VLLM_VENV}" ]; then
    echo "Creating virtualenv: ${VLLM_VENV}"
    "${PYTHON_BIN}" -m venv "${VLLM_VENV}"
fi

# shellcheck disable=SC1091
source "${VLLM_VENV}/bin/activate"

echo "Installing vLLM"
python -m pip install -U pip wheel setuptools
python -m pip install vllm --extra-index-url "${TORCH_INDEX}"

echo
echo "Done. Activate with:"
echo "  source \"${VLLM_VENV}/bin/activate\""
echo
echo "Verify with:"
echo "  python -m vllm.entrypoints.openai.api_server --help | head"
