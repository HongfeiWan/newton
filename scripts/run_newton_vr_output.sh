#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEVICE="/dev/video44"
CAPTURE_DISPLAY="${DISPLAY:-}"
CAPTURE_OFFSET="0,0"
CAPTURE_SIZE="1280x720"
CAPTURE_FPS="20"
PLANE_DISTANCE="1.6"
PLANE_WIDTH="1.2"
PLANE_OFFSET_X="0.0"
PLANE_OFFSET_Y="0.0"
ISAAC_TELEOP_ROOT_ARG="${ISAAC_TELEOP_ROOT:-}"
XR_ENV_PATH="${HOME}/.cloudxr/run/cloudxr.env"
PYTHON_BIN="${PYTHON_BIN:-}"
DOCKERFILE_SYNTAX_IMAGE="${TELEOP_DOCKERFILE_SYNTAX_IMAGE:-}"
CAMERA_STREAMER_LITE_IMAGE="${NEWTON_CAMERA_STREAMER_LITE_IMAGE:-}"
if [[ -z "$CAMERA_STREAMER_LITE_IMAGE" ]]; then
    if docker image inspect harness-camera-streamer-lite:latest >/dev/null 2>&1; then
        CAMERA_STREAMER_LITE_IMAGE="harness-camera-streamer-lite:latest"
    else
        CAMERA_STREAMER_LITE_IMAGE="newton-camera-streamer-lite:latest"
    fi
fi
XR_HAND_LOG_PATH="${TELEOP_XR_HAND_LOG_PATH:-$REPO_ROOT/logs/xr_debug/camera_overlay_hand.jsonl}"
XR_HAND_LOG_STRIDE="${TELEOP_XR_HAND_LOG_STRIDE:-10}"
XR_STATUS_PATH="${TELEOP_XR_STATUS_PATH:-}"
XR_RECOVERY_MAX_RETRIES="${TELEOP_CAMERA_XR_RECOVERY_MAX_RETRIES:-3600}"
XR_RECOVERY_DELAY_S="${TELEOP_CAMERA_XR_RECOVERY_DELAY_S:-2.0}"
CHECK_ONLY="false"
DISABLE_HAND_OVERLAY="false"
SKIP_PATCH="false"
USE_LITE_CAMERA_STREAMER="true"

usage() {
    cat <<'EOF'
Usage: scripts/run_newton_vr_output.sh [options]

Streams the Newton viewer into Quest/CloudXR as an XR plane and enables the
IsaacTeleop hand skeleton overlay.

Options:
  --device /dev/videoX        V4L2 loopback output device (default: /dev/video44)
  --display :N[.S]            X11 display containing the Newton viewer
  --offset X,Y                X11 capture offset (default: 0,0)
  --size WIDTHxHEIGHT         Capture size (default: 1280x720)
  --fps FPS                   ffmpeg capture FPS (default: 20)
  --plane-distance M          XR plane distance in meters (default: 1.6)
  --plane-width M             XR plane width in meters (default: 1.2)
  --plane-offset-x M          XR plane horizontal offset (default: 0.0)
  --plane-offset-y M          XR plane vertical offset (default: 0.0)
  --isaac-teleop-root PATH    IsaacTeleop checkout (default: ISAAC_TELEOP_ROOT or ../IsaacTeleop)
  --cloudxr-env PATH          CloudXR env file (default: ~/.cloudxr/run/cloudxr.env)
  --python PATH               Python executable (default: python3, then python)
  --dockerfile-syntax-image IMAGE
                              Override Dockerfile BuildKit frontend image, e.g.
                              docker.1ms.run/docker/dockerfile:1
  --lite-image TAG            Lite camera_streamer image tag
                              (default: existing harness-camera-streamer-lite, else newton-camera-streamer-lite)
  --hand-log-path PATH        XR hand joint log path (default: logs/xr_debug/camera_overlay_hand.jsonl)
  --hand-log-stride N         Log every N rendered frames (default: 10)
  --xr-status-path PATH       Teleop status JSON path shown in VR
                              (default: $NV_CXR_RUNTIME_DIR/teleop_xr_status.json)
  --xr-recovery-max-retries N Number of XR-session retries before giving up
                              (default: 3600, about 2 hours at 2s delay)
  --xr-recovery-delay-s SEC   Delay between XR-session retries (default: 2.0)
  --use-upstream-camera-streamer
                              Use IsaacTeleop camera_streamer.sh instead of the migrated lite image
  --disable-hand-overlay      Disable XR hand skeleton overlay
  --skip-patch                Do not patch camera_streamer before running
  --check-only                Print preflight status and exit
  -h, --help                  Show this help

If the V4L2 device is missing, create it once with:
  sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --display) CAPTURE_DISPLAY="$2"; shift 2 ;;
        --offset) CAPTURE_OFFSET="$2"; shift 2 ;;
        --size) CAPTURE_SIZE="$2"; shift 2 ;;
        --fps) CAPTURE_FPS="$2"; shift 2 ;;
        --plane-distance) PLANE_DISTANCE="$2"; shift 2 ;;
        --plane-width) PLANE_WIDTH="$2"; shift 2 ;;
        --plane-offset-x) PLANE_OFFSET_X="$2"; shift 2 ;;
        --plane-offset-y) PLANE_OFFSET_Y="$2"; shift 2 ;;
        --isaac-teleop-root) ISAAC_TELEOP_ROOT_ARG="$2"; shift 2 ;;
        --cloudxr-env) XR_ENV_PATH="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        --dockerfile-syntax-image) DOCKERFILE_SYNTAX_IMAGE="$2"; shift 2 ;;
        --lite-image) CAMERA_STREAMER_LITE_IMAGE="$2"; shift 2 ;;
        --hand-log-path) XR_HAND_LOG_PATH="$2"; shift 2 ;;
        --hand-log-stride) XR_HAND_LOG_STRIDE="$2"; shift 2 ;;
        --xr-status-path) XR_STATUS_PATH="$2"; shift 2 ;;
        --xr-recovery-max-retries) XR_RECOVERY_MAX_RETRIES="$2"; shift 2 ;;
        --xr-recovery-delay-s) XR_RECOVERY_DELAY_S="$2"; shift 2 ;;
        --use-upstream-camera-streamer) USE_LITE_CAMERA_STREAMER="false"; shift ;;
        --disable-hand-overlay) DISABLE_HAND_OVERLAY="true"; shift ;;
        --skip-patch) SKIP_PATCH="true"; shift ;;
        --check-only) CHECK_ONLY="true"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

failures=0

if [[ -f "$XR_ENV_PATH" ]]; then
    # shellcheck disable=SC1090
    source "$XR_ENV_PATH"
fi
if [[ -z "$XR_STATUS_PATH" ]]; then
    XR_STATUS_PATH="${NV_CXR_RUNTIME_DIR:-$HOME/.cloudxr/run}/teleop_xr_status.json"
fi

status_ok() { printf '[vr-output] ok: %s\n' "$*"; }
status_warn() { printf '[vr-output] warn: %s\n' "$*" >&2; }
status_error() { printf '[vr-output] error: %s\n' "$*" >&2; failures=$((failures + 1)); }

require_command() {
    if command -v "$1" >/dev/null 2>&1; then
        status_ok "$1=$(command -v "$1")"
    else
        status_error "missing command: $1"
    fi
}

v4l2_reload_hint() {
    local video_nr="${DEVICE#/dev/video}"
    status_warn "reload the loopback device with:"
    status_warn "sudo modprobe -r v4l2loopback"
    status_warn "sudo modprobe v4l2loopback video_nr=${video_nr} card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2 max_width=1920 max_height=1080"
}

cloudxr_ipc_ready() {
    local socket_path="$1"
    [[ -S "$socket_path" ]] || return 1
    [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || return 0
    "$PYTHON_BIN" - "$socket_path" >/dev/null 2>&1 <<'PY'
import socket
import sys

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    sock.connect(sys.argv[1])
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}

check_v4l2_output_device() {
    if [[ ! -e "$DEVICE" ]]; then
        status_error "sim screen device missing: $DEVICE"
        v4l2_reload_hint
        return
    fi
    status_ok "sim screen device=$DEVICE"

    if ! command -v v4l2-ctl >/dev/null 2>&1; then
        status_warn "v4l2-ctl not found; cannot validate $DEVICE capabilities"
        return
    fi

    local device_info output_fmt
    output_fmt="$(v4l2-ctl -d "$DEVICE" --get-fmt-video-out 2>&1 || true)"
    device_info="$(v4l2-ctl -d "$DEVICE" --all 2>&1 || true)"
    if grep -Eq "Width/Height|Pixel Format" <<<"$output_fmt" \
        || grep -Eq "Video Output|Video Output Multiplanar" <<<"$device_info"; then
        status_ok "sim screen output capability=V4L2 output"
    else
        status_error "$DEVICE is not advertising V4L2 output capability; ffmpeg cannot feed the sim screen"
        status_warn "current output format query:"
        sed -n '1,12p' <<<"$output_fmt" >&2
        status_warn "current v4l2-ctl summary:"
        sed -n '1,35p' <<<"$device_info" >&2
        v4l2_reload_hint
    fi
}

resolve_isaac_root() {
    if [[ -n "$ISAAC_TELEOP_ROOT_ARG" ]]; then
        printf '%s\n' "$ISAAC_TELEOP_ROOT_ARG"
    elif [[ -d "$REPO_ROOT/../IsaacTeleop" ]]; then
        printf '%s\n' "$REPO_ROOT/../IsaacTeleop"
    elif [[ -d "$REPO_ROOT/external/IsaacTeleop" ]]; then
        printf '%s\n' "$REPO_ROOT/external/IsaacTeleop"
    else
        printf '%s\n' ""
    fi
}

ISAAC_TELEOP_ROOT_RESOLVED="$(resolve_isaac_root)"
CAMERA_STREAMER_ROOT="$ISAAC_TELEOP_ROOT_RESOLVED/examples/camera_streamer"

require_command ffmpeg
if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    fi
fi
if [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]]; then
    status_ok "python=$PYTHON_BIN"
else
    status_error "missing Python executable. Pass --python /path/to/python or set PYTHON_BIN."
fi

if command -v docker >/dev/null 2>&1; then
    status_ok "docker=$(command -v docker)"
    if docker ps >/dev/null 2>&1; then
        status_ok "docker daemon is accessible"
    else
        status_error "docker daemon is not accessible by the current user. Run: sudo usermod -aG docker \$USER && newgrp docker"
    fi
else
    status_error "docker command not found; IsaacTeleop camera_streamer.sh run requires Docker"
fi

if [[ -n "${NV_CXR_RUNTIME_DIR:-}" ]] && cloudxr_ipc_ready "${NV_CXR_RUNTIME_DIR}/ipc_cloudxr"; then
    status_ok "CloudXR IPC=${NV_CXR_RUNTIME_DIR}/ipc_cloudxr"
else
    status_error "CloudXR runtime is not ready. Start: python -m isaacteleop.cloudxr --accept-eula"
fi

if [[ -n "$CAPTURE_DISPLAY" ]]; then
    status_ok "capture display=$CAPTURE_DISPLAY"
else
    status_error "DISPLAY is empty. Pass --display :N after confirming the Newton viewer display with echo \$DISPLAY."
fi

check_v4l2_output_device
status_ok "xr status path=$XR_STATUS_PATH"
status_ok "XR recovery retries=${XR_RECOVERY_MAX_RETRIES} delay=${XR_RECOVERY_DELAY_S}s"

if [[ -d "$CAMERA_STREAMER_ROOT" && -x "$CAMERA_STREAMER_ROOT/camera_streamer.sh" ]]; then
    status_ok "camera_streamer=$CAMERA_STREAMER_ROOT"
else
    status_error "IsaacTeleop camera_streamer not found. Set --isaac-teleop-root or ISAAC_TELEOP_ROOT."
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
    if [[ "$failures" -eq 0 ]]; then
        status_ok "preflight passed"
        exit 0
    fi
    status_error "preflight failed with $failures problem(s)"
    exit 1
fi

if [[ "$failures" -ne 0 ]]; then
    status_error "fix preflight errors before starting VR output"
    exit 1
fi

if [[ "$USE_LITE_CAMERA_STREAMER" == "false" && "$SKIP_PATCH" != "true" ]]; then
    patch_args=(--camera-streamer-root "$CAMERA_STREAMER_ROOT")
    if [[ -n "$DOCKERFILE_SYNTAX_IMAGE" ]]; then
        patch_args+=(--dockerfile-syntax-image "$DOCKERFILE_SYNTAX_IMAGE")
    fi
    "$PYTHON_BIN" "$REPO_ROOT/tools/apply_camera_streamer_overlay.py" "${patch_args[@]}"
fi

CONFIG_PATH="$("$PYTHON_BIN" "$REPO_ROOT/tools/generate_camera_streamer_sim_config.py" \
    --isaac-teleop-root "$ISAAC_TELEOP_ROOT_RESOLVED" \
    --device "$DEVICE" \
    --width "${CAPTURE_SIZE%x*}" \
    --height "${CAPTURE_SIZE#*x}" \
    --fps 0 \
    --plane-distance "$PLANE_DISTANCE" \
    --plane-width "$PLANE_WIDTH" \
    --plane-offset-x "$PLANE_OFFSET_X" \
    --plane-offset-y "$PLANE_OFFSET_Y")"

FFMPEG_PID=""
cleanup() {
    if [[ -n "$FFMPEG_PID" ]] && kill -0 "$FFMPEG_PID" >/dev/null 2>&1; then
        status_warn "stopping ffmpeg pid=$FFMPEG_PID"
        kill "$FFMPEG_PID" >/dev/null 2>&1 || true
        wait "$FFMPEG_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

capture_input="${CAPTURE_DISPLAY}+${CAPTURE_OFFSET}"
status_ok "streaming ${capture_input} ${CAPTURE_SIZE}@${CAPTURE_FPS}fps -> ${DEVICE}"
ffmpeg -nostdin -hide_banner -loglevel warning \
    -f x11grab -draw_mouse 1 \
    -framerate "$CAPTURE_FPS" \
    -video_size "$CAPTURE_SIZE" \
    -i "$capture_input" \
    -codec:v rawvideo -pix_fmt yuyv422 \
    -f v4l2 "$DEVICE" &
FFMPEG_PID="$!"

sleep 1
if ! kill -0 "$FFMPEG_PID" >/dev/null 2>&1; then
    status_error "ffmpeg exited during startup"
    wait "$FFMPEG_PID" || true
    exit 1
fi

if [[ "$DISABLE_HAND_OVERLAY" == "true" ]]; then
    export TELEOP_CAMERA_DISABLE_HAND_OVERLAY=1
else
    unset TELEOP_CAMERA_DISABLE_HAND_OVERLAY || true
fi

status_ok "starting camera_streamer XR config=$CONFIG_PATH"
if [[ "$USE_LITE_CAMERA_STREAMER" == "true" ]]; then
    status_ok "lite camera_streamer image=$CAMERA_STREAMER_LITE_IMAGE"
    if ! docker image inspect "$CAMERA_STREAMER_LITE_IMAGE" >/dev/null 2>&1; then
        status_warn "lite camera_streamer image not found: $CAMERA_STREAMER_LITE_IMAGE"
        "$REPO_ROOT/scripts/build_camera_streamer_lite.sh" \
            --isaac-teleop-root "$ISAAC_TELEOP_ROOT_RESOLVED" \
            --image-tag "$CAMERA_STREAMER_LITE_IMAGE" \
            --python "$PYTHON_BIN"
    fi

    config_basename="$(basename "$CONFIG_PATH")"
    cxr_host_volume_path="${CXR_HOST_VOLUME_PATH:-$HOME/.cloudxr}"
    xr_runtime_json="${XR_RUNTIME_JSON:-${cxr_host_volume_path}/openxr_cloudxr.json}"
    nv_cxr_runtime_dir="${NV_CXR_RUNTIME_DIR:-${cxr_host_volume_path}/run}"
    hand_log_dir="$(dirname "$XR_HAND_LOG_PATH")"
    xr_status_dir="$(dirname "$XR_STATUS_PATH")"
    mkdir -p "$hand_log_dir"
    mkdir -p "$xr_status_dir"
    docker_args=(
        --rm
        --gpus all
        --runtime=nvidia
        --privileged
        --network=host
        --ulimit stack=33554432
        -e "XR_RUNTIME_JSON=${xr_runtime_json}"
        -e "NV_CXR_RUNTIME_DIR=${nv_cxr_runtime_dir}"
        -e "TELEOP_XR_HAND_LOG_PATH=${XR_HAND_LOG_PATH}"
        -e "TELEOP_XR_HAND_LOG_STRIDE=${XR_HAND_LOG_STRIDE}"
        -e "TELEOP_XR_STATUS_PATH=${XR_STATUS_PATH}"
        -e "TELEOP_CAMERA_XR_RECOVERY_MAX_RETRIES=${XR_RECOVERY_MAX_RETRIES}"
        -e "TELEOP_CAMERA_XR_RECOVERY_DELAY_S=${XR_RECOVERY_DELAY_S}"
        -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-graphics,video,compute,utility,display}"
        -v /dev:/dev
        -v /run/udev:/run/udev:rw
        -v "${cxr_host_volume_path}:${cxr_host_volume_path}:ro"
        -v "${CONFIG_PATH}:/config/${config_basename}:ro"
        -v "${hand_log_dir}:${hand_log_dir}:rw"
    )
    if [[ "${xr_status_dir}" != "${cxr_host_volume_path}" && "${xr_status_dir}" != "${cxr_host_volume_path}/"* ]]; then
        docker_args+=(-v "${xr_status_dir}:${xr_status_dir}:ro")
    fi
    if [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
        docker_args+=(
            -e "VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json"
            -v /usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro
        )
    fi
    if [[ -n "${TELEOP_CAMERA_DISABLE_HAND_OVERLAY:-}" ]]; then
        docker_args+=(-e "TELEOP_CAMERA_DISABLE_HAND_OVERLAY=${TELEOP_CAMERA_DISABLE_HAND_OVERLAY}")
    fi

    docker run "${docker_args[@]}" \
        "$CAMERA_STREAMER_LITE_IMAGE" \
        python3 /camera_streamer/teleop_camera_app.py \
        --config "/config/${config_basename}" \
        --source local \
        --mode xr
else
    cd "$CAMERA_STREAMER_ROOT"
    ./camera_streamer.sh run -- --config "$CONFIG_PATH" --source local --mode xr
fi
