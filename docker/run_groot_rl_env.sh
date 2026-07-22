#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${REPO_DIR}/.." && pwd)"

IMAGE_NAME="${NEWTON_GROOT_RL_IMAGE:-${NEWTON_GROOT_RTC_IMAGE:-newton-direct-gpu-groot:latest}}"
GPU_INDEX="${NEWTON_GROOT_RL_GPU:-${NEWTON_GROOT_GPU:-0}}"

resolve_newton_conda_env() {
    if [[ "${CONDA_DEFAULT_ENV:-}" == "newton" && -x "${CONDA_PREFIX:-}/bin/python" ]]; then
        printf '%s\n' "${CONDA_PREFIX}"
        return 0
    fi
    command -v conda >/dev/null 2>&1 || return 1
    conda env list 2>/dev/null | awk '$1 == "newton" {print $NF; exit}'
}

NEWTON_CONDA_ENV="${NEWTON_CONDA_ENV:-$(resolve_newton_conda_env || true)}"
if [[ -n "${NEWTON_CONDA_ENV}" && -x "${NEWTON_CONDA_ENV}/bin/python" ]]; then
    PYTHON_BIN="${NEWTON_GROOT_RL_PYTHON:-${NEWTON_CONDA_ENV}/bin/python}"
else
    PYTHON_BIN="${NEWTON_GROOT_RL_PYTHON:-/workspace/newton/.venv/bin/python}"
fi

docker_args=(
    --rm
    --name newton-groot-rl-env
    --gpus all
    --ipc host
    --ulimit stack=33554432
    -e "CUDA_VISIBLE_DEVICES=${GPU_INDEX}"
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}"
    -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
    -e "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}"
    -e "HOME=${HOME}"
    -e "PYTHONPATH=${REPO_DIR}:/workspace/newton"
    -v "${PROJECT_DIR}:${PROJECT_DIR}:rw"
    -v "${HOME}/.cache:${HOME}/.cache:rw"
    -w "${REPO_DIR}"
)
if [[ -n "${NEWTON_CONDA_ENV}" && -x "${NEWTON_CONDA_ENV}/bin/python" ]]; then
    docker_args+=(-v "${NEWTON_CONDA_ENV}:${NEWTON_CONDA_ENV}:ro")
fi

docker run "${docker_args[@]}" \
    "${IMAGE_NAME}" \
    "${PYTHON_BIN}" tools/run_newton_groot_rl_env.py \
    --device cuda:0 \
    "$@"
