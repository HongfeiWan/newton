#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${REPO_DIR}/.." && pwd)"

IMAGE_NAME="${NEWTON_GROOT_RL_IMAGE:-${NEWTON_GROOT_RTC_IMAGE:-newton-direct-gpu-groot:latest}}"
GPU_INDEX="${NEWTON_GROOT_RL_GPU:-${NEWTON_GROOT_GPU:-0}}"
CONDA_PYTHON="${REPO_DIR}/conda_envs/newton/bin/python"
if [[ -x "${CONDA_PYTHON}" ]]; then
    PYTHON_BIN="${NEWTON_GROOT_RL_PYTHON:-${CONDA_PYTHON}}"
else
    PYTHON_BIN="${NEWTON_GROOT_RL_PYTHON:-/workspace/newton/.venv/bin/python}"
fi

docker run --rm \
    --name newton-groot-rl-env \
    --gpus all \
    --ipc host \
    --ulimit stack=33554432 \
    -e "CUDA_VISIBLE_DEVICES=${GPU_INDEX}" \
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}" \
    -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}" \
    -e "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}" \
    -e "HOME=${HOME}" \
    -e "PYTHONPATH=${REPO_DIR}:/workspace/newton" \
    -v "${PROJECT_DIR}:${PROJECT_DIR}:rw" \
    -v "${REPO_DIR}/conda_envs/newton:${REPO_DIR}/conda_envs/newton:ro" \
    -v "${HOME}/.cache:${HOME}/.cache:rw" \
    -w "${REPO_DIR}" \
    "${IMAGE_NAME}" \
    "${PYTHON_BIN}" tools/run_newton_groot_rl_env.py \
    --device cuda:0 \
    "$@"
