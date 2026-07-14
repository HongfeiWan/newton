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
IMAGE_NAME_PROVIDED="${IMAGE_NAME+x}"
IMAGE_NAME="${IMAGE_NAME:-newton:latest}"
VR_OUTPUT_MODE="${NEWTON_VR_OUTPUT_MODE:-legacy-v4l2}"
NEWTON_VR_GPU="${NEWTON_VR_GPU:-0}"
DISPLAY_ARG="${DISPLAY:-:0}"
MODEL_PATH="${MODEL_PATH:-${CONTAINER_HOME}/.cache/teleop_stack/vosk/vosk-model-small-cn-0.22}"
ISAAC_TELEOP_ROOT="${ISAAC_TELEOP_ROOT:-${PROJECT_DIR}/IsaacTeleop}"
IMPORTED_WEBXR_DIR="${IMPORTED_WEBXR_DIR:-${CONTAINER_HOME}/.cache/teleop_stack/cloudxr_web_client_remote/webxr_client}"
CAMERA_STREAMER_LITE_IMAGE="${NEWTON_CAMERA_STREAMER_LITE_IMAGE:-harness-camera-streamer-lite:latest}"
CLOUDXR_NATIVE_DIR="${NEWTON_CLOUDXR_NATIVE_DIR:-${REPO_DIR}/docker/cloudxr-native-6.1}"
CLOUDXR_NATIVE_CONTAINER_DIR="${NEWTON_CLOUDXR_NATIVE_CONTAINER_DIR:-/workspace/newton/.venv/lib/python3.12/site-packages/isaacteleop/cloudxr/native}"

parse_vr_output_mode() {
    local idx=0
    local arg
    local mode="${VR_OUTPUT_MODE}"
    while (( idx < ${#ORIGINAL_ARGS[@]} )); do
        arg="${ORIGINAL_ARGS[$idx]}"
        case "${arg}" in
            --)
                break
                ;;
            --vr-output-mode)
                idx=$((idx + 1))
                if (( idx < ${#ORIGINAL_ARGS[@]} )); then
                    mode="${ORIGINAL_ARGS[$idx]}"
                fi
                ;;
            --vr-output-mode=*)
                mode="${arg#--vr-output-mode=}"
                ;;
            --skip-vr-output)
                mode="off"
                ;;
            --with-vr-output)
                mode="legacy-v4l2"
                ;;
        esac
        idx=$((idx + 1))
    done
    printf '%s\n' "${mode}"
}

VR_OUTPUT_MODE="$(parse_vr_output_mode)"
case "${VR_OUTPUT_MODE}" in
    direct-gpu|legacy-v4l2|off) ;;
    *) printf '[vr-docker] invalid --vr-output-mode: %s\n' "${VR_OUTPUT_MODE}" >&2; exit 2 ;;
esac
if [[ "${VR_OUTPUT_MODE}" == "direct-gpu" && -z "${IMAGE_NAME_PROVIDED}" ]]; then
    IMAGE_NAME="${NEWTON_DIRECT_GPU_IMAGE:-newton-direct-gpu:latest}"
fi

shell_quote() {
    printf '%q' "$1"
}

append_env_assignment_if_set() {
    local var_name="$1"
    if [[ -n "${!var_name+x}" ]]; then
        cmd+=" ${var_name}=$(shell_quote "${!var_name}")"
    fi
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

    cmd="cd $(shell_quote "${REPO_DIR}") && NEWTON_RUN_VR_STACK_REEXEC_DOCKER=1 DISPLAY=$(shell_quote "${DISPLAY_ARG}")"
    append_env_assignment_if_set NEWTON_VR_GPU
    append_env_assignment_if_set NEWTON_VR_OUTPUT_MODE
    append_env_assignment_if_set NEWTON_DYNAMIC_OBJECT_SHAPE
    append_env_assignment_if_set NEWTON_DYNAMIC_BOTTLE_SPEC
    append_env_assignment_if_set NEWTON_SCENE_PHYSICS_CONFIG
    append_env_assignment_if_set IMAGE_NAME
    append_env_assignment_if_set NEWTON_DIRECT_GPU_IMAGE
    cmd+=" exec $(shell_quote "$0")"
    for arg in "${ORIGINAL_ARGS[@]}"; do
        cmd+=" $(shell_quote "${arg}")"
    done
    printf '[vr-docker] current shell cannot access docker.sock; re-executing this command inside docker group via sg docker\n' >&2
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

ensure_sim_screen_device() {
    local device="/dev/video44"
    local needs_reload=0

    if [[ ! -e "${device}" ]]; then
        needs_reload=1
    elif command -v v4l2-ctl >/dev/null 2>&1; then
        if ! v4l2-ctl -d "${device}" --get-fmt-video-out >/dev/null 2>&1; then
            needs_reload=1
        fi
    fi

    if [[ "${needs_reload}" -eq 1 ]]; then
        sudo modprobe -r v4l2loopback >/dev/null 2>&1 || true
        sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2 max_width=1920 max_height=1080
    fi
}

if [[ "${VR_OUTPUT_MODE}" == "legacy-v4l2" ]]; then
    ensure_sim_screen_device
fi

NVIDIA_VISIBLE_DEVICES_ARG="${NVIDIA_VISIBLE_DEVICES:-all}"
CUDA_VISIBLE_DEVICES_ARG="${CUDA_VISIBLE_DEVICES:-}"
CONTAINER_PYTHONPATH="${REPO_DIR}"
if [[ "${VR_OUTPUT_MODE}" == "direct-gpu" ]]; then
    NVIDIA_VISIBLE_DEVICES_ARG="${NVIDIA_VISIBLE_DEVICES:-all}"
    CUDA_VISIBLE_DEVICES_ARG="${CUDA_VISIBLE_DEVICES:-${NEWTON_VR_GPU}}"
    CONTAINER_PYTHONPATH="${REPO_DIR}:/workspace/newton:/camera_streamer:/camera_streamer/build:/camera_streamer/python:/camera_streamer/python/lib:/camera_streamer/build/python:/camera_streamer/build/python/lib:/opt/nvidia/holoscan/python/lib"
    PYOPENGL_PLATFORM_ARG="${PYOPENGL_PLATFORM:-egl}"
    EGL_VENDOR_LIBRARY_FILENAMES_ARG="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
    if [[ -n "${PYGLET_HEADLESS_DEVICE:-}" ]]; then
        PYGLET_HEADLESS_DEVICE_ARG="${PYGLET_HEADLESS_DEVICE}"
    elif [[ "${NEWTON_VR_GPU}" == "0" ]]; then
        PYGLET_HEADLESS_DEVICE_ARG="1"
    elif [[ "${NEWTON_VR_GPU}" == "1" ]]; then
        PYGLET_HEADLESS_DEVICE_ARG="0"
    else
        PYGLET_HEADLESS_DEVICE_ARG="0"
    fi
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
    --ulimit stack=33554432
    -e "DISPLAY=${DISPLAY_ARG}"
    -e "HOME=${CONTAINER_HOME}"
    -e "USER=${RUN_USER}"
    -e "REPO_DIR=${REPO_DIR}"
    -e "PYTHONPATH=${CONTAINER_PYTHONPATH}"
    -e "PYTHON_BIN=/workspace/newton/.venv/bin/python"
    -e "SCENE_PYTHON_BIN=/workspace/newton/.venv/bin/python"
    -e "TELEOP_PYTHON_BIN=/workspace/newton/.venv/bin/python"
    -e "ISAAC_TELEOP_ROOT=${ISAAC_TELEOP_ROOT}"
    -e "MODEL_PATH=${MODEL_PATH}"
    -e "IMPORTED_IMAGE=cloudxr-web-app:latest"
    -e "IMPORTED_WEBXR_DIR=${IMPORTED_WEBXR_DIR}"
    -e "NEWTON_CAMERA_STREAMER_LITE_IMAGE=${CAMERA_STREAMER_LITE_IMAGE}"
    -e "NEWTON_VR_OUTPUT_MODE=${VR_OUTPUT_MODE}"
    -e "NEWTON_VR_GPU=${NEWTON_VR_GPU}"
    -e "NEWTON_DYNAMIC_OBJECT_SHAPE=${NEWTON_DYNAMIC_OBJECT_SHAPE:-cylinder}"
    -e "NEWTON_SCENE_PHYSICS_CONFIG=${NEWTON_SCENE_PHYSICS_CONFIG:-${REPO_DIR}/configs/scene_physics/groot_rtc.json}"
    -e "NV_DEVICE_PROFILE=${NV_DEVICE_PROFILE:-Quest3}"
    -e "NV_CXR_ENABLE_PUSH_DEVICES=${NV_CXR_ENABLE_PUSH_DEVICES:-0}"
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES_ARG}"
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

if [[ -n "${NEWTON_DYNAMIC_BOTTLE_SPEC:-}" ]]; then
    docker_args+=(-e "NEWTON_DYNAMIC_BOTTLE_SPEC=${NEWTON_DYNAMIC_BOTTLE_SPEC}")
fi

if [[ -n "${CUDA_VISIBLE_DEVICES_ARG}" ]]; then
    docker_args+=(-e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_ARG}")
fi
if [[ "${VR_OUTPUT_MODE}" == "direct-gpu" ]]; then
    docker_args+=(
        -e "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM_ARG}"
        -e "__EGL_VENDOR_LIBRARY_FILENAMES=${EGL_VENDOR_LIBRARY_FILENAMES_ARG}"
        -e "PYGLET_HEADLESS_DEVICE=${PYGLET_HEADLESS_DEVICE_ARG}"
    )
fi

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
