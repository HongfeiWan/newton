#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${REPO_DIR}/.." && pwd)"

IMAGE_NAME="${NEWTON_GROOT_RTC_IMAGE:-newton-direct-gpu-groot:latest}"
GPU_INDEX="${NEWTON_GROOT_GPU:-${NEWTON_VR_GPU:-0}}"
DISPLAY_ARG="${DISPLAY:-:0}"
ISAAC_GROOT_ROOT="${ISAAC_GROOT_ROOT:-${PROJECT_DIR}/Isaac-GR00T}"
XAUTHORITY_PATH="${XAUTHORITY:-${HOME}/.Xauthority}"
CONDA_PYTHON="${REPO_DIR}/conda_envs/newton/bin/python"
if [[ -x "${CONDA_PYTHON}" ]]; then
    PYTHON_BIN="${NEWTON_GROOT_PYTHON:-${CONDA_PYTHON}}"
else
    PYTHON_BIN="${NEWTON_GROOT_PYTHON:-/workspace/newton/.venv/bin/python}"
fi

if [[ ! -d "${ISAAC_GROOT_ROOT}/gr00t" ]]; then
    printf 'Missing Isaac-GR00T source: %s/gr00t\n' "${ISAAC_GROOT_ROOT}" >&2
    exit 2
fi

docker_args=(
    --rm
    --gpus all
    --privileged
    --network host
    --ipc host
    --ulimit stack=33554432
    -e "CUDA_VISIBLE_DEVICES=${GPU_INDEX}"
    -e "DISPLAY=${DISPLAY_ARG}"
    -e "HOME=${HOME}"
    -e "ISAAC_GROOT_ROOT=${ISAAC_GROOT_ROOT}"
    -e "NO_ALBUMENTATIONS_UPDATE=1"
    -e "PYTHONPATH=${REPO_DIR}:${ISAAC_GROOT_ROOT}:/workspace/newton:/camera_streamer:/camera_streamer/build"
    -v "${PROJECT_DIR}:${PROJECT_DIR}:rw"
    -v "${REPO_DIR}/conda_envs/newton:${REPO_DIR}/conda_envs/newton:ro"
    -v "${HOME}/.cache:${HOME}/.cache:rw"
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v /dev:/dev
    -w "${REPO_DIR}"
)

if [[ -f "${XAUTHORITY_PATH}" ]]; then
    docker_args+=(
        -e "XAUTHORITY=${XAUTHORITY_PATH}"
        -v "${XAUTHORITY_PATH}:${XAUTHORITY_PATH}:ro"
    )
    if command -v xhost >/dev/null 2>&1; then
        DISPLAY="${DISPLAY_ARG}" XAUTHORITY="${XAUTHORITY_PATH}" \
            xhost +SI:localuser:root >/dev/null 2>&1 || true
    fi
fi

exec docker run "${docker_args[@]}" "${IMAGE_NAME}" \
    "${PYTHON_BIN}" tools/run_newton_groot_rtc_control.py \
    --device cuda:0 \
    --policy-device cuda:0 \
    "$@"
