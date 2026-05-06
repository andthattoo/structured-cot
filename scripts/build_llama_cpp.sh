#!/usr/bin/env bash
# Build native llama.cpp's llama-server for this repo.
#
# Modes:
#   native              build unmodified upstream llama.cpp
#   pre-trigger-grammar build llama.cpp with the local tools+grammar patch
#
# Examples:
#   ./scripts/build_llama_cpp.sh native
#   LLAMA_CPP_REF=master ./scripts/build_llama_cpp.sh pre-trigger-grammar
#   CUDA=OFF ./scripts/build_llama_cpp.sh native

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-${MODE:-native}}"
if [ "$#" -gt 0 ]; then
    shift
fi

LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-}"
BUILD_DIR="${BUILD_DIR:-}"
INSTALL_DIR="${INSTALL_DIR:-${ROOT_DIR}/.local/bin}"
PATCH_FILE="${PATCH_FILE:-${ROOT_DIR}/third_party/patches/llama.cpp/pre-trigger-grammar-78433f6.patch}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA="${CUDA:-ON}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

usage() {
    cat <<'EOF'
Usage: scripts/build_llama_cpp.sh [native|pre-trigger-grammar] [extra cmake args...]

Environment:
  LLAMA_CPP_REPO  upstream repo URL
  LLAMA_CPP_DIR   checkout path, default .deps/llama.cpp or .deps/llama.cpp-pre-trigger
  LLAMA_CPP_REF   optional branch/tag/SHA to checkout before building
  BUILD_DIR       CMake build dir, default $LLAMA_CPP_DIR/build
  INSTALL_DIR     binary output dir, default .local/bin
  PATCH_FILE      patch used by pre-trigger-grammar mode
  CUDA            ON/OFF for -DGGML_CUDA, default ON
  BUILD_TYPE      CMake build type, default Release
  JOBS            parallel build jobs

Extra arguments are passed to cmake configure, e.g.
  ./scripts/build_llama_cpp.sh native -DGGML_METAL=ON
EOF
}

case "${MODE}" in
    native|pre-trigger-grammar|patched)
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "ERROR: unknown mode '${MODE}'" >&2
        usage >&2
        exit 2
        ;;
esac

if [ "${MODE}" = "patched" ]; then
    MODE="pre-trigger-grammar"
fi

if [ -z "${LLAMA_CPP_DIR}" ]; then
    if [ "${MODE}" = "pre-trigger-grammar" ]; then
        LLAMA_CPP_DIR="${ROOT_DIR}/.deps/llama.cpp-pre-trigger"
    else
        LLAMA_CPP_DIR="${ROOT_DIR}/.deps/llama.cpp"
    fi
fi
BUILD_DIR="${BUILD_DIR:-${LLAMA_CPP_DIR}/build}"

if ! command -v git >/dev/null 2>&1; then
    echo "ERROR: git is required" >&2
    exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
    echo "ERROR: cmake is required" >&2
    exit 1
fi

mkdir -p "$(dirname "${LLAMA_CPP_DIR}")" "${INSTALL_DIR}"

if [ ! -d "${LLAMA_CPP_DIR}/.git" ]; then
    echo "Cloning llama.cpp"
    echo "  repo = ${LLAMA_CPP_REPO}"
    echo "  dir  = ${LLAMA_CPP_DIR}"
    git clone "${LLAMA_CPP_REPO}" "${LLAMA_CPP_DIR}"
fi

if [ -n "${LLAMA_CPP_REF}" ]; then
    echo "Checking out llama.cpp ref"
    echo "  ref = ${LLAMA_CPP_REF}"
    git -C "${LLAMA_CPP_DIR}" fetch --tags origin
    git -C "${LLAMA_CPP_DIR}" checkout "${LLAMA_CPP_REF}"
else
    echo "Using existing llama.cpp checkout"
    echo "  dir = ${LLAMA_CPP_DIR}"
    echo "  ref = $(git -C "${LLAMA_CPP_DIR}" rev-parse --short HEAD)"
fi

apply_pre_trigger_patch() {
    if [ ! -f "${PATCH_FILE}" ]; then
        echo "ERROR: patch file not found: ${PATCH_FILE}" >&2
        exit 1
    fi

    if git -C "${LLAMA_CPP_DIR}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
        echo "Applying pre-trigger grammar patch"
        git -C "${LLAMA_CPP_DIR}" apply "${PATCH_FILE}"
        return
    fi

    if git -C "${LLAMA_CPP_DIR}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
        echo "Pre-trigger grammar patch already appears to be applied"
        return
    fi

    echo "ERROR: could not apply ${PATCH_FILE}" >&2
    echo "       The patch is version-sensitive; try a different LLAMA_CPP_REF." >&2
    exit 1
}

if [ "${MODE}" = "pre-trigger-grammar" ]; then
    apply_pre_trigger_patch
fi

echo "Configuring llama.cpp"
echo "  mode       = ${MODE}"
echo "  build_dir  = ${BUILD_DIR}"
echo "  build_type = ${BUILD_TYPE}"
echo "  cuda       = ${CUDA}"

cmake -S "${LLAMA_CPP_DIR}" -B "${BUILD_DIR}" \
    -DGGML_CUDA="${CUDA}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    "$@"

echo "Building llama-server"
cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" -j "${JOBS}" --target llama-server

SERVER_BIN="${BUILD_DIR}/bin/llama-server"
if [ ! -x "${SERVER_BIN}" ]; then
    SERVER_BIN="$(find "${BUILD_DIR}" -type f -perm -111 -name llama-server | head -1)"
fi
if [ -z "${SERVER_BIN}" ] || [ ! -x "${SERVER_BIN}" ]; then
    echo "ERROR: built llama-server binary was not found under ${BUILD_DIR}" >&2
    exit 1
fi

if [ "${MODE}" = "pre-trigger-grammar" ]; then
    OUT_BIN="${INSTALL_DIR}/llama-server-pre-trigger"
else
    OUT_BIN="${INSTALL_DIR}/llama-server"
fi

cp "${SERVER_BIN}" "${OUT_BIN}"
chmod +x "${OUT_BIN}"

echo "Installed:"
echo "  ${OUT_BIN}"
