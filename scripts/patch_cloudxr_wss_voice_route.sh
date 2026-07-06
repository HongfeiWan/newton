#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -x "/home/whf/anaconda3/envs/newton/bin/python3" ]]; then
    PYTHON_BIN="/home/whf/anaconda3/envs/newton/bin/python3"
fi
if [[ -z "${PYTHON_BIN}" ]]; then
    PYTHON_BIN="$(command -v python3)"
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" tools/patch_installed_cloudxr_wss_voice_route.py
