#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${REPO_DIR}/.." && pwd)"
ORIGINAL_ARGS=("$@")
RUN_USER="${SUDO_USER:-${USER:-$(id -un)}}"
RUN_HOME="$(getent passwd "${RUN_USER}" | cut -d: -f6 || true)"
RUN_HOME="${RUN_HOME:-${HOME}}"
RUN_UID="$(id -u "${RUN_USER}" 2>/dev/null || printf '%s' "$(id -u)")"
HOST_HOME="${HOST_HOME:-${RUN_HOME}}"
CONTAINER_HOME="${CONTAINER_HOME:-${HOST_HOME}}"
IMAGE_NAME="${IMAGE_NAME:-newton:latest}"
DISPLAY_ARG="${DISPLAY:-:0}"
MODEL_PATH="${MODEL_PATH:-${CONTAINER_HOME}/.cache/teleop_stack/vosk/vosk-model-small-cn-0.22}"
ISAAC_TELEOP_ROOT="${ISAAC_TELEOP_ROOT:-${PROJECT_DIR}/IsaacTeleop}"
IMPORTED_WEBXR_DIR="${IMPORTED_WEBXR_DIR:-${CONTAINER_HOME}/.cache/teleop_stack/cloudxr_web_client_remote/webxr_client}"
CAMERA_STREAMER_LITE_IMAGE="${NEWTON_CAMERA_STREAMER_LITE_IMAGE:-harness-camera-streamer-lite:latest}"
CLOUDXR_NATIVE_DIR="${NEWTON_CLOUDXR_NATIVE_DIR:-${REPO_DIR}/docker/cloudxr-native-6.1}"
CLOUDXR_NATIVE_CONTAINER_DIR="${NEWTON_CLOUDXR_NATIVE_CONTAINER_DIR:-/workspace/newton/.venv/lib/python3.12/site-packages/isaacteleop/cloudxr/native}"

shell_quote() {
    printf '%q' "$1"
}

reexec_with_docker_group_if_possible() {
    if [[ "${NEWTON_RUN_VR_STACK_REEXEC_DOCKER:-0}" == "1" ]]; then
        return 1
    fi
    if ! command -v sg >/dev/null 2>&1; then
        return 1
    fi
    if ! getent group docker | grep -Eq "(^|[:,])${RUN_USER}([,]|$)"; then
        return 1
    fi
    if ! sg docker -c "docker ps >/dev/null 2>&1"; then
        return 1
    fi

    cmd="cd $(shell_quote "${REPO_DIR}") && NEWTON_RUN_VR_STACK_REEXEC_DOCKER=1 DISPLAY=$(shell_quote "${DISPLAY_ARG}") exec $(shell_quote "$0")"
    for arg in "${ORIGINAL_ARGS[@]}"; do
        cmd+=" $(shell_quote "${arg}")"
    done
    printf '[vr-docker] current shell cannot access docker.sock; re-executing with sg docker\n' >&2
    exec sg docker -c "${cmd}"
}

if command -v docker >/dev/null 2>&1 && ! docker ps >/dev/null 2>&1; then
    reexec_with_docker_group_if_possible || true
fi

HOST_XAUTHORITY=""
if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    HOST_XAUTHORITY="${XAUTHORITY}"
elif [[ -f "${HOST_HOME}/.Xauthority" ]]; then
    HOST_XAUTHORITY="${HOST_HOME}/.Xauthority"
elif [[ -f "/run/user/${RUN_UID}/gdm/Xauthority" ]]; then
    HOST_XAUTHORITY="/run/user/${RUN_UID}/gdm/Xauthority"
fi

display_socket() {
    local display_name="$1"
    local display_num
    display_num="${display_name%%.*}"
    display_num="${display_num#:}"
    if [[ "${display_num}" =~ ^[0-9]+$ ]]; then
        printf '/tmp/.X11-unix/X%s\n' "${display_num}"
    fi
}

DISPLAY_SOCKET="$(display_socket "${DISPLAY_ARG}")"
if [[ -n "${DISPLAY_SOCKET}" && ! -S "${DISPLAY_SOCKET}" ]]; then
    shopt -s nullglob
    x_sockets=(/tmp/.X11-unix/X*)
    shopt -u nullglob
    if [[ "${#x_sockets[@]}" -eq 1 ]]; then
        detected_display=":${x_sockets[0]##*/X}"
        printf '[vr-docker] display %s has no X socket; using detected display %s\n' "${DISPLAY_ARG}" "${detected_display}" >&2
        DISPLAY_ARG="${detected_display}"
    fi
fi

if [[ ! -e /dev/video44 ]]; then
    sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2 max_width=1920 max_height=1080
fi

if command -v xhost >/dev/null 2>&1; then
    if [[ -n "${HOST_XAUTHORITY}" ]]; then
        XAUTHORITY="${HOST_XAUTHORITY}" xhost +SI:localuser:root >/dev/null 2>&1 || true
        XAUTHORITY="${HOST_XAUTHORITY}" xhost +local:root >/dev/null 2>&1 || true
    else
        xhost +SI:localuser:root >/dev/null 2>&1 || true
        xhost +local:root >/dev/null 2>&1 || true
    fi
fi

docker_args=(
    --rm
    --name newton-vr-stack
    --gpus all
    --privileged
    --network=host
    --ipc=host
    -e "DISPLAY=${DISPLAY_ARG}"
    -e "HOME=${CONTAINER_HOME}"
    -e "USER=${RUN_USER}"
    -e "REPO_DIR=${REPO_DIR}"
    -e "PYTHONPATH=${REPO_DIR}"
    -e "PYTHON_BIN=/workspace/newton/.venv/bin/python"
    -e "SCENE_PYTHON_BIN=/workspace/newton/.venv/bin/python"
    -e "TELEOP_PYTHON_BIN=/workspace/newton/.venv/bin/python"
    -e "ISAAC_TELEOP_ROOT=${ISAAC_TELEOP_ROOT}"
    -e "MODEL_PATH=${MODEL_PATH}"
    -e "IMPORTED_IMAGE=cloudxr-web-app:latest"
    -e "IMPORTED_WEBXR_DIR=${IMPORTED_WEBXR_DIR}"
    -e "NEWTON_CAMERA_STREAMER_LITE_IMAGE=${CAMERA_STREAMER_LITE_IMAGE}"
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}"
    -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-graphics,video,compute,utility,display}"
    -e "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
    -v "${PROJECT_DIR}:${PROJECT_DIR}:rw"
    -v "${HOST_HOME}/.cache:${CONTAINER_HOME}/.cache:rw"
    -v "${HOST_HOME}/.cloudxr:${CONTAINER_HOME}/.cloudxr:rw"
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v /dev:/dev
    -v /run/udev:/run/udev:rw
    -v /var/run/docker.sock:/var/run/docker.sock
)

if [[ -x /usr/bin/docker ]]; then
    docker_args+=(-v /usr/bin/docker:/usr/bin/docker:ro)
fi

if [[ -d "${CLOUDXR_NATIVE_DIR}" ]]; then
    docker_args+=(
        -e "NEWTON_CLOUDXR_NATIVE_DIR=${CLOUDXR_NATIVE_DIR}"
        -v "${CLOUDXR_NATIVE_DIR}:${CLOUDXR_NATIVE_CONTAINER_DIR}:ro"
    )
fi

if [[ -t 0 && -t 1 ]]; then
    docker_args+=(-it)
fi

if [[ -n "${HOST_XAUTHORITY}" ]]; then
    docker_args+=(-e "XAUTHORITY=${HOST_XAUTHORITY}" -v "${HOST_XAUTHORITY}:${HOST_XAUTHORITY}:ro")
fi

if [[ -d /usr/share/vulkan/icd.d ]]; then
    docker_args+=(-v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro)
fi
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
    docker_args+=(-v /etc/vulkan/icd.d/nvidia_icd.json:/etc/vulkan/icd.d/nvidia_icd.json:ro)
fi
if [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
    docker_args+=(-v /usr/share/glvnd/egl_vendor.d/10_nvidia.json:/usr/share/glvnd/egl_vendor.d/10_nvidia.json:ro)
fi

exec docker run "${docker_args[@]}" "${IMAGE_NAME}" \
    bash -lc 'cd "${REPO_DIR}" && exec scripts/run_newton_vr_prereqs.sh --display "${DISPLAY}" "$@"' \
    bash "$@"
