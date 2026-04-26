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
#   CUDA_TOOLKIT_PACKAGE=cuda-toolkit-12-4 ./scripts/install_llama_cpp_cuda.sh
#   INSTALL_CUDA_TOOLKIT=0 ./scripts/install_llama_cpp_cuda.sh

set -euo pipefail

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${HOME}/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_CPP_DIR}/build}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
CUDA_TOOLKIT_PACKAGE="${CUDA_TOOLKIT_PACKAGE:-cuda-toolkit-12-4}"
INSTALL_CUDA_TOOLKIT="${INSTALL_CUDA_TOOLKIT:-1}"

if ! command -v sudo >/dev/null 2>&1; then
    echo "ERROR: sudo not found. Install dependencies manually, then rerun:"
    echo "  apt-get update"
    echo "  apt-get install -y git cmake build-essential libcurl4-openssl-dev"
    exit 1
fi

echo "Installing llama.cpp build dependencies"
sudo apt-get update
sudo apt-get install -y git cmake build-essential libcurl4-openssl-dev ca-certificates wget

if ! command -v nvcc >/dev/null 2>&1 && [ "${INSTALL_CUDA_TOOLKIT}" = "1" ]; then
    echo "nvcc not found; installing CUDA toolkit from NVIDIA apt repository"

    if [ ! -r /etc/os-release ]; then
        echo "ERROR: cannot detect Ubuntu release from /etc/os-release"
        exit 1
    fi

    # shellcheck disable=SC1091
    . /etc/os-release
    ubuntu_version="${VERSION_ID//./}"
    distro="ubuntu${ubuntu_version}"
    arch="$(dpkg --print-architecture)"
    case "${arch}" in
        amd64) cuda_arch="x86_64" ;;
        arm64) cuda_arch="sbsa" ;;
        *)
            echo "ERROR: unsupported architecture for CUDA repo: ${arch}"
            exit 1
            ;;
    esac

    cuda_repo_url="https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${cuda_arch}"
    keyring_deb="/tmp/cuda-keyring_1.1-1_all.deb"

    echo "Using CUDA repo: ${cuda_repo_url}"
    wget -O "${keyring_deb}" "${cuda_repo_url}/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i "${keyring_deb}"
    sudo apt-get update

    if apt-cache show "${CUDA_TOOLKIT_PACKAGE}" >/dev/null 2>&1; then
        sudo apt-get install -y "${CUDA_TOOLKIT_PACKAGE}"
    else
        echo "Package ${CUDA_TOOLKIT_PACKAGE} not available; falling back to cuda-toolkit"
        sudo apt-get install -y cuda-toolkit
    fi
fi

if ! command -v nvcc >/dev/null 2>&1; then
    for cuda_bin in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        if [ -x "${cuda_bin}/nvcc" ]; then
            export PATH="${cuda_bin}:${PATH}"
            export CUDAToolkit_ROOT="${cuda_bin%/bin}"
            break
        fi
    done
fi

if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc still not found."
    echo "Try manually installing a CUDA toolkit package, then rerun:"
    echo "  sudo apt-get install -y cuda-toolkit-12-4"
    echo "  export PATH=\"/usr/local/cuda/bin:\$PATH\""
    echo "  export CUDAToolkit_ROOT=/usr/local/cuda"
    exit 1
fi

echo "Using nvcc: $(command -v nvcc)"
nvcc --version | tail -n 1

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
