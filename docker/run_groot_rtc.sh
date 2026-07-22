#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${REPO_DIR}/.." && pwd)"

IMAGE_NAME="${NEWTON_GROOT_RTC_IMAGE:-newton-direct-gpu-groot:latest}"
GPU_INDEX="${NEWTON_GROOT_GPU:-${NEWTON_VR_GPU:-0}}"
DISPLAY_ARG="${DISPLAY:-:0}"
RENDER_MODE="${NEWTON_GROOT_RENDER_MODE:-direct-gpu}"
PREVIEW_WIDTH="${NEWTON_GROOT_PREVIEW_WIDTH:-1600}"
PREVIEW_HEIGHT="${NEWTON_GROOT_PREVIEW_HEIGHT:-720}"
PREVIEW_FPS="${NEWTON_GROOT_PREVIEW_FPS:-15}"
INPUT_PREVIEW_WIDTH="${NEWTON_GROOT_INPUT_PREVIEW_WIDTH:-320}"
RENDER_FPS="${NEWTON_GROOT_RENDER_FPS:-60}"
ISAAC_GROOT_ROOT="${ISAAC_GROOT_ROOT:-${PROJECT_DIR}/Isaac-GR00T}"
XAUTHORITY_PATH="${XAUTHORITY:-${HOME}/.Xauthority}"

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
    PYTHON_BIN="${NEWTON_GROOT_PYTHON:-${NEWTON_CONDA_ENV}/bin/python}"
else
    PYTHON_BIN="${NEWTON_GROOT_PYTHON:-/workspace/newton/.venv/bin/python}"
fi

if [[ ! -d "${ISAAC_GROOT_ROOT}/gr00t" ]]; then
    printf 'Missing Isaac-GR00T source: %s/gr00t\n' "${ISAAC_GROOT_ROOT}" >&2
    exit 2
fi

case "${RENDER_MODE}" in
    direct-gpu|window) ;;
    *) printf 'Invalid NEWTON_GROOT_RENDER_MODE: %s (expected direct-gpu or window)\n' "${RENDER_MODE}" >&2; exit 2 ;;
esac

VIEWER_KIND="gl"
args=("$@")
for ((idx = 0; idx < ${#args[@]}; idx++)); do
    case "${args[$idx]}" in
        --viewer)
            if ((idx + 1 < ${#args[@]})); then
                VIEWER_KIND="${args[$((idx + 1))]}"
            fi
            ;;
        --viewer=*) VIEWER_KIND="${args[$idx]#--viewer=}" ;;
    esac
done

preview_dir=""
preview_fifo=""
preview_pid=""
preview_log=""

cleanup_preview() {
    if [[ -n "${preview_pid}" ]]; then
        kill "${preview_pid}" >/dev/null 2>&1 || true
        wait "${preview_pid}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${preview_dir}" ]]; then
        rm -rf "${preview_dir}"
    fi
}

if [[ "${RENDER_MODE}" == "direct-gpu" && "${VIEWER_KIND}" == "gl" ]]; then
    for value in \
        "${PREVIEW_WIDTH}" \
        "${PREVIEW_HEIGHT}" \
        "${PREVIEW_FPS}" \
        "${INPUT_PREVIEW_WIDTH}" \
        "${RENDER_FPS}"; do
        if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
            printf 'Direct-GPU preview dimensions/FPS must be positive integers, got %s\n' "${value}" >&2
            exit 2
        fi
    done
    if ((INPUT_PREVIEW_WIDTH >= PREVIEW_WIDTH)); then
        printf 'NEWTON_GROOT_INPUT_PREVIEW_WIDTH must be smaller than NEWTON_GROOT_PREVIEW_WIDTH.\n' >&2
        exit 2
    fi
    display_num="${DISPLAY_ARG%%.*}"
    display_num="${display_num#:}"
    if [[ ! "${display_num}" =~ ^[0-9]+$ || ! -S "/tmp/.X11-unix/X${display_num}" ]]; then
        printf 'Direct-GPU preview cannot open DISPLAY=%s (missing X socket).\n' "${DISPLAY_ARG}" >&2
        exit 2
    fi
    if ! command -v gst-launch-1.0 >/dev/null 2>&1; then
        printf 'Direct-GPU preview requires host gst-launch-1.0.\n' >&2
        exit 2
    fi

    preview_dir="$(mktemp -d /tmp/newton-groot-preview.XXXXXX)"
    preview_fifo="${preview_dir}/viewer.rgb"
    preview_log="${preview_dir}/gstreamer.log"
    mkfifo "${preview_fifo}"
    trap cleanup_preview EXIT INT TERM
    DISPLAY="${DISPLAY_ARG}" XAUTHORITY="${XAUTHORITY_PATH}" \
        gst-launch-1.0 -q \
        filesrc location="${preview_fifo}" blocksize="$((PREVIEW_WIDTH * PREVIEW_HEIGHT * 3))" ! \
        rawvideoparse format=rgb width="${PREVIEW_WIDTH}" height="${PREVIEW_HEIGHT}" \
            framerate="${PREVIEW_FPS}/1" ! \
        queue max-size-buffers=1 leaky=downstream ! videoconvert ! \
        ximagesink sync=false force-aspect-ratio=true >"${preview_log}" 2>&1 &
    preview_pid=$!
    sleep 0.2
    if ! kill -0 "${preview_pid}" >/dev/null 2>&1; then
        cat "${preview_log}" >&2 || true
        printf 'Failed to start the host direct-GPU preview window.\n' >&2
        exit 2
    fi
fi

docker_args=(
    --rm
    --name newton-groot-rtc
    --gpus all
    --privileged
    --network host
    --ipc host
    --ulimit stack=33554432
    -e "CUDA_VISIBLE_DEVICES=${GPU_INDEX}"
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}"
    -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-graphics,video,compute,utility,display}"
    -e "CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-LAZY}"
    -e "DISPLAY=${DISPLAY_ARG}"
    -e "HOME=${HOME}"
    -e "ISAAC_GROOT_ROOT=${ISAAC_GROOT_ROOT}"
    -e "NO_ALBUMENTATIONS_UPDATE=1"
    -e "PYTHONPATH=${REPO_DIR}:${ISAAC_GROOT_ROOT}:/workspace/newton:/camera_streamer:/camera_streamer/build"
    -v "${PROJECT_DIR}:${PROJECT_DIR}:rw"
    -v "${HOME}/.cache:${HOME}/.cache:rw"
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
    -v /dev:/dev
    -w "${REPO_DIR}"
)
if [[ -n "${NEWTON_CONDA_ENV}" && -x "${NEWTON_CONDA_ENV}/bin/python" ]]; then
    docker_args+=(-v "${NEWTON_CONDA_ENV}:${NEWTON_CONDA_ENV}:ro")
fi

python_args=(
    --device cuda:0
    --policy-device cuda:0
    --capture-graph
    --async-policy
)

if [[ "${RENDER_MODE}" == "direct-gpu" ]]; then
    if [[ "${GPU_INDEX}" == "0" ]]; then
        PYGLET_HEADLESS_DEVICE_ARG="${PYGLET_HEADLESS_DEVICE:-1}"
    elif [[ "${GPU_INDEX}" == "1" ]]; then
        PYGLET_HEADLESS_DEVICE_ARG="${PYGLET_HEADLESS_DEVICE:-0}"
    else
        PYGLET_HEADLESS_DEVICE_ARG="${PYGLET_HEADLESS_DEVICE:-0}"
    fi
    docker_args+=(
        -e "PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}"
        -e "__EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
        -e "PYGLET_HEADLESS_DEVICE=${PYGLET_HEADLESS_DEVICE_ARG}"
        -e "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
        -v /run/udev:/run/udev:rw
    )
    python_args+=(--render-fps "${RENDER_FPS}")
    if [[ "${VIEWER_KIND}" == "gl" ]]; then
        docker_args+=(-v "${preview_dir}:${preview_dir}:rw")
        python_args+=(
            --headless
            --viewer-fifo-preview "${preview_fifo}"
            --viewer-fifo-preview-width "${PREVIEW_WIDTH}"
            --viewer-fifo-preview-height "${PREVIEW_HEIGHT}"
            --viewer-fifo-preview-fps "${PREVIEW_FPS}"
            --viewer-fifo-preview-input-width "${INPUT_PREVIEW_WIDTH}"
        )
    fi
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

status=0
docker run "${docker_args[@]}" "${IMAGE_NAME}" \
    "${PYTHON_BIN}" tools/run_newton_groot_rtc_control.py \
    "${python_args[@]}" \
    "$@" || status=$?
exit "${status}"
