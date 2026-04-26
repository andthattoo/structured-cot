#!/usr/bin/env bash
# Build native llama.cpp with CUDA support on an Ubuntu GPU box.
#
# Usage:
#   ./scripts/install_llama_cpp_cuda.sh
#   export PATH="$HOME/llama.cpp/build/bin:$PATH"
#
# Customization:
#   LLAMA_CPP_DIR=$HOME/src/llama.cpp ./scripts/install_llama_cpp_cuda.sh
#   BUILD_DIR=$HOME/src/llama.cpp/build-cuda ./scripts/install_llama_cpp_cuda.sh
#   BUILD_JOBS=8 ./scripts/install_llama_cpp_cuda.sh

set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${HOME}/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_CPP_DIR}/build}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"

if ! command -v sudo >/dev/null 2>&1; then
    echo "ERROR: sudo not found. Install dependencies manually, then rerun:"
    echo "  apt-get update"
    echo "  apt-get install -y git cmake build-essential libcurl4-openssl-dev"
    exit 1
fi

echo "Installing llama.cpp build dependencies"
sudo apt-get update
sudo apt-get install -y git cmake build-essential libcurl4-openssl-dev

if [ -d "${LLAMA_CPP_DIR}/.git" ]; then
    echo "Updating existing llama.cpp checkout: ${LLAMA_CPP_DIR}"
    git -C "${LLAMA_CPP_DIR}" pull --ff-only
else
    if [ -e "${LLAMA_CPP_DIR}" ]; then
        echo "ERROR: ${LLAMA_CPP_DIR} exists but is not a git checkout."
        echo "Set LLAMA_CPP_DIR to a different path or remove the existing path."
        exit 1
    fi
    echo "Cloning llama.cpp into ${LLAMA_CPP_DIR}"
    git clone https://github.com/ggml-org/llama.cpp "${LLAMA_CPP_DIR}"
fi

echo "Configuring CUDA build"
cmake -S "${LLAMA_CPP_DIR}" -B "${BUILD_DIR}" \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release

echo "Building llama.cpp with ${BUILD_JOBS} jobs"
cmake --build "${BUILD_DIR}" --config Release -j "${BUILD_JOBS}"

echo
echo "Done. Add llama.cpp binaries to PATH:"
echo "  export PATH=\"${BUILD_DIR}/bin:\$PATH\""
echo
echo "Then verify:"
echo "  llama-server --help | head"
