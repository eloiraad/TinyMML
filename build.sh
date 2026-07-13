#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/tinytensor/build"
BUILD_TYPE="Release"
ENABLE_CUDA="OFF"
CLEAN_BUILD="OFF"

usage() {
    echo "Usage: ./build.sh [--clean] [--debug] [--cuda]"
}

for argument in "$@"; do
    case "${argument}" in
        --clean) CLEAN_BUILD="ON" ;;
        --debug) BUILD_TYPE="Debug" ;;
        --cuda) ENABLE_CUDA="ON" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: ${argument}" >&2; usage >&2; exit 2 ;;
    esac
done

PYTHON_BIN="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found at ${PYTHON_BIN}. Create .venv first." >&2
    exit 1
fi

CMAKE_BIN="${CMAKE:-${ROOT_DIR}/.venv/bin/cmake}"
if [[ ! -x "${CMAKE_BIN}" ]]; then
    CMAKE_BIN="$(command -v cmake || true)"
fi
if [[ -z "${CMAKE_BIN}" ]]; then
    echo "CMake not found. Install requirements.txt in .venv first." >&2
    exit 1
fi

if [[ "${CLEAN_BUILD}" == "ON" ]]; then
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

PYBIND11_DIR="$("${PYTHON_BIN}" -m pybind11 --cmakedir)"

echo "[TinyMML] Python: ${PYTHON_BIN}"
echo "[TinyMML] Build: ${BUILD_TYPE}, CUDA=${ENABLE_CUDA}"

"${CMAKE_BIN}" -S "${ROOT_DIR}/tinytensor" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DPython_EXECUTABLE="${PYTHON_BIN}" \
    -Dpybind11_DIR="${PYBIND11_DIR}" \
    -DTINYTENSOR_ENABLE_CUDA="${ENABLE_CUDA}"
"${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel

echo "[TinyMML] Built extension in ${ROOT_DIR}"
