#!/usr/bin/env bash
# Fetch and apply the experimental llama.cpp pre-trigger grammar patch.
#
# This patch was shared in:
#   https://github.com/ggml-org/llama.cpp/discussions/22408
#
# It adds a pre-trigger / reasoning-phase grammar slot that can coexist with
# llama.cpp's internally generated lazy tool grammar.
#
# Usage:
#   ./scripts/apply_llama_cpp_pre_trigger_grammar_patch.sh
#   cmake --build "$HOME/llama.cpp/build" --config Release -j
#
# Customization:
#   LLAMA_CPP_DIR=$HOME/src/llama.cpp ./scripts/apply_llama_cpp_pre_trigger_grammar_patch.sh
#   PATCH_URL=https://.../patch ./scripts/apply_llama_cpp_pre_trigger_grammar_patch.sh

set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${HOME}/llama.cpp}"
PATCH_URL="${PATCH_URL:-https://github.com/user-attachments/files/27124806/llama-cpp-pre-trigger-grammar-78433f6.patch}"
PATCH_PATH="${PATCH_PATH:-/tmp/llama-cpp-pre-trigger-grammar.patch}"

if [ ! -d "${LLAMA_CPP_DIR}/.git" ]; then
    echo "ERROR: llama.cpp checkout not found: ${LLAMA_CPP_DIR}"
    echo "Run ./scripts/install_llama_cpp_cuda.sh first, or set LLAMA_CPP_DIR."
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl not found. Install curl or download the patch manually."
    exit 1
fi

echo "Downloading patch"
echo "  url  = ${PATCH_URL}"
echo "  path = ${PATCH_PATH}"
curl -L "${PATCH_URL}" -o "${PATCH_PATH}"

echo "Checking patch against ${LLAMA_CPP_DIR}"
if git -C "${LLAMA_CPP_DIR}" apply --check "${PATCH_PATH}"; then
    echo "Applying patch"
    git -C "${LLAMA_CPP_DIR}" apply "${PATCH_PATH}"
else
    echo "ERROR: patch does not apply cleanly."
    echo "Check llama.cpp revision and inspect with:"
    echo "  git -C \"${LLAMA_CPP_DIR}\" status"
    echo "  git -C \"${LLAMA_CPP_DIR}\" apply --reject \"${PATCH_PATH}\""
    exit 1
fi

echo
echo "Patch applied. Rebuild llama.cpp:"
echo "  cmake --build \"${LLAMA_CPP_DIR}/build\" --config Release -j"
echo
echo "Then verify reasoning flags:"
echo "  \"${LLAMA_CPP_DIR}/build/bin/llama-server\" --help | grep -i reasoning"
