#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCENE_PHYSICS_CONFIG="${NEWTON_SCENE_PHYSICS_CONFIG:-${REPO_ROOT}/configs/scene_physics/groot_rtc.json}"

export ISAAC_TELEOP_ROOT="${ISAAC_TELEOP_ROOT:-${REPO_ROOT}/../IsaacTeleop}"
DISPLAY_ARG="${DISPLAY:-:0}"
MODEL_PATH="${MODEL_PATH:-/home/whf/.cache/teleop_stack/vosk/vosk-model-small-cn-0.22}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.5}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/vr_stack}"
XR_ENV_PATH="${XR_ENV_PATH:-${HOME}/.cloudxr/run/cloudxr.env}"
PYTHON_BIN="${PYTHON_BIN:-}"
SCENE_PYTHON_BIN="${SCENE_PYTHON_BIN:-${PYTHON_BIN}}"
TELEOP_PYTHON_BIN="${TELEOP_PYTHON_BIN:-${PYTHON_BIN}}"
WEB_MODE_WAS_SET="${WEB_MODE+x}"
WEB_MODE="${WEB_MODE:-image}"
START_CLOUDXR=1
START_WEB=1
START_VOICE=1
START_VR_OUTPUT="${NEWTON_START_VR_OUTPUT:-1}"
VR_OUTPUT_MODE="${NEWTON_VR_OUTPUT_MODE:-legacy-v4l2}"
CHECK_ONLY=0
REEXEC_DOCKER_GROUP="${NEWTON_VR_PREREQS_REEXEC_DOCKER:-0}"
ORIGINAL_ARGS=("$@")
WITH_SCENE=1
SCENE_DEVICE="${SCENE_DEVICE:-cuda:0}"
TELEOP_INPUT_SOURCE="${TELEOP_INPUT_SOURCE:-quest}"
SCENE_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/run_newton_vr_prereqs.sh [options]

Starts the Newton Quest teleop stack in one terminal:
  1. python -m isaacteleop.cloudxr --accept-eula
  2. scripts/run_cloudxr_web_client.sh
  3. scripts/run_quest_voice_command_bridge.sh
  4. debug/import_dual_nero_linker_l10.py --quest-teleop

The default path starts the Quest sim-screen/XR-plane and hand skeleton overlay,
but teleop reads Quest/OpenXR hands directly instead of the overlay JSONL log.

Options:
  --display :N[.S]       X11 display to capture for Newton/VR output (default: $DISPLAY or :0)
  --model-path PATH      Vosk model path
  --min-confidence N     Voice command min confidence (default: 0.5)
  --log-dir PATH         Log directory (default: logs/vr_stack)
  --cloudxr-env PATH     CloudXR env file (default: ~/.cloudxr/run/cloudxr.env)
  --python PATH          Python executable for both scene and teleop helpers
  --scene-python PATH    Python executable for Newton scene (default: conda env newton)
  --teleop-python PATH   Python executable for CloudXR/voice helpers (default: Python with isaacteleop)
  --web-mode MODE        Pass --mode MODE to run_cloudxr_web_client.sh (default: image)
  --skip-cloudxr         Do not start CloudXR runtime
  --skip-web             Do not start CloudXR web client
  --skip-voice           Do not start Quest voice bridge
  --vr-output-mode MODE  VR output mode: direct-gpu, legacy-v4l2, or off
                         (default: legacy-v4l2)
  --with-vr-output       Start legacy-v4l2 VR sim-screen/XR output via camera_streamer
  --skip-vr-output       Do not start VR sim-screen/XR output
  --teleop-input-source overlay-log|quest
                         Quest hand input source (default: quest)
  --with-scene           Start the Newton scene in this terminal (default)
  --no-scene             Only start VR prerequisites/output; run Newton separately
  --scene-device DEVICE  Newton device passed to --device (default: cuda:0)
  --scene-backend cpu|gpu
                         Compatibility alias: cpu -> cpu, gpu -> cuda:0
  --check-only           Check basic prerequisites and exit
  --                     Extra arguments passed to debug/import_dual_nero_linker_l10.py
  -h, --help             Show this help

Default one-command startup:
  scripts/run_newton_vr_prereqs.sh --display :0

Then open https://<host-ip>:8443/ in Quest, enable voice, enter XR, and say 开始.
EOF
}

log() { printf '[vr-prereqs] %s\n' "$*"; }
ok() { log "ok: $*"; }
warn() { log "warn: $*" >&2; }
err() { log "error: $*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --display) DISPLAY_ARG="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --min-confidence) MIN_CONFIDENCE="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --cloudxr-env) XR_ENV_PATH="$2"; shift 2 ;;
        --python) PYTHON_BIN="$2"; SCENE_PYTHON_BIN="$2"; TELEOP_PYTHON_BIN="$2"; shift 2 ;;
        --scene-python) SCENE_PYTHON_BIN="$2"; shift 2 ;;
        --teleop-python) TELEOP_PYTHON_BIN="$2"; shift 2 ;;
        --web-mode) WEB_MODE="$2"; WEB_MODE_WAS_SET=1; shift 2 ;;
        --skip-cloudxr) START_CLOUDXR=0; shift ;;
        --skip-web) START_WEB=0; shift ;;
        --skip-voice) START_VOICE=0; shift ;;
        --vr-output-mode)
            case "$2" in
                direct-gpu|legacy-v4l2|off) VR_OUTPUT_MODE="$2" ;;
                *) err "--vr-output-mode must be direct-gpu, legacy-v4l2, or off"; usage >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --with-vr-output) START_VR_OUTPUT=1; VR_OUTPUT_MODE="legacy-v4l2"; shift ;;
        --skip-vr-output) START_VR_OUTPUT=0; VR_OUTPUT_MODE="off"; shift ;;
        --teleop-input-source)
            case "$2" in
                overlay-log|quest) TELEOP_INPUT_SOURCE="$2" ;;
                *) err "--teleop-input-source must be overlay-log or quest"; usage >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --with-scene) WITH_SCENE=1; shift ;;
        --no-scene) WITH_SCENE=0; shift ;;
        --scene-device) SCENE_DEVICE="$2"; shift 2 ;;
        --scene-backend)
            case "$2" in
                cpu) SCENE_DEVICE="cpu" ;;
                gpu) SCENE_DEVICE="cuda:0" ;;
                *) err "--scene-backend must be cpu or gpu"; usage >&2; exit 2 ;;
            esac
            shift 2
            ;;
        --check-only) CHECK_ONLY=1; shift ;;
        --) shift; SCENE_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) err "unknown argument: $1"; usage >&2; exit 2 ;;
    esac
done

case "${VR_OUTPUT_MODE}" in
    direct-gpu|legacy-v4l2) START_VR_OUTPUT=1 ;;
    off) START_VR_OUTPUT=0 ;;
    *) err "--vr-output-mode must be direct-gpu, legacy-v4l2, or off"; exit 2 ;;
esac
if [[ "${VR_OUTPUT_MODE}" == "direct-gpu" && -z "${WEB_MODE_WAS_SET}" && "${WEB_MODE}" == "image" ]]; then
    WEB_MODE="image-static"
fi

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

failures=0
check_ok() { ok "$*"; }
check_warn() { warn "$*"; }
check_error() { err "$*"; failures=$((failures + 1)); }

shell_quote() {
    printf '%q' "$1"
}

resolve_conda_env_python() {
    local env_name="$1"
    if [[ "${CONDA_DEFAULT_ENV:-}" == "${env_name}" && -x "${CONDA_PREFIX:-}/bin/python3" ]]; then
        printf '%s\n' "${CONDA_PREFIX}/bin/python3"
        return 0
    fi

    local env_path
    env_path="$(conda env list 2>/dev/null | awk -v name="${env_name}" '$1 == name {print $NF; exit}')"
    if [[ -n "${env_path}" && -x "${env_path}/bin/python3" ]]; then
        printf '%s\n' "${env_path}/bin/python3"
        return 0
    fi
    return 1
}

python_can_import() {
    local python_path="$1"
    local module_name="$2"
    [[ -n "${python_path}" && -x "${python_path}" ]] || return 1
    "${python_path}" -c "import ${module_name}" >/dev/null 2>&1
}

if [[ -z "${SCENE_PYTHON_BIN}" ]]; then
    SCENE_PYTHON_BIN="$(resolve_conda_env_python newton || true)"
fi
if [[ -z "${TELEOP_PYTHON_BIN}" ]]; then
    for candidate in \
        "$(resolve_conda_env_python genesis || true)" \
        "${SCENE_PYTHON_BIN}" \
        "$(command -v python3 || true)"
    do
        if python_can_import "${candidate}" isaacteleop; then
            TELEOP_PYTHON_BIN="${candidate}"
            break
        fi
    done
fi
if [[ -z "${SCENE_PYTHON_BIN}" ]]; then
    SCENE_PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "${TELEOP_PYTHON_BIN}" ]]; then
    TELEOP_PYTHON_BIN="$(command -v python3 || true)"
fi

reexec_with_docker_group_if_possible() {
    if [[ "${REEXEC_DOCKER_GROUP}" == "1" ]]; then
        return 1
    fi
    if ! command -v sg >/dev/null 2>&1; then
        return 1
    fi
    if ! getent group docker | grep -Eq "(^|[:,])${USER}([,]|$)"; then
        return 1
    fi
    if ! sg docker -c "docker ps >/dev/null 2>&1"; then
        return 1
    fi

    local cmd
    cmd="cd $(shell_quote "${REPO_ROOT}") && NEWTON_VR_PREREQS_REEXEC_DOCKER=1 exec $(shell_quote "$0")"
    for arg in "$@"; do
        cmd+=" $(shell_quote "${arg}")"
    done
    warn "current shell has not refreshed docker group; re-executing this script with 'sg docker'"
    exec sg docker -c "${cmd}"
}

if [[ -n "${SCENE_PYTHON_BIN}" && -x "${SCENE_PYTHON_BIN}" ]]; then
    check_ok "scene python=${SCENE_PYTHON_BIN}"
else
    check_error "missing scene Python executable; pass --scene-python /path/to/python"
fi

if [[ -n "${TELEOP_PYTHON_BIN}" && -x "${TELEOP_PYTHON_BIN}" ]]; then
    check_ok "teleop python=${TELEOP_PYTHON_BIN}"
    if python_can_import "${TELEOP_PYTHON_BIN}" isaacteleop; then
        check_ok "teleop python imports isaacteleop"
    else
        check_error "teleop Python cannot import isaacteleop; pass --teleop-python /path/to/python"
    fi
else
    check_error "missing teleop Python executable; pass --teleop-python /path/to/python"
fi

SCENE_PYTHONPATH_DIR=""
if [[ -n "${SCENE_PYTHON_BIN}" && -x "${SCENE_PYTHON_BIN}" ]] && ! python_can_import "${SCENE_PYTHON_BIN}" isaacteleop; then
    isaacteleop_package_dir="$(
        "${TELEOP_PYTHON_BIN}" - <<'PY' 2>/dev/null || true
import pathlib
import isaacteleop

print(pathlib.Path(isaacteleop.__file__).resolve().parent)
PY
    )"
    if [[ -n "${isaacteleop_package_dir}" && -d "${isaacteleop_package_dir}" ]]; then
        SCENE_PYTHONPATH_DIR="${LOG_DIR}/pythonpath"
        mkdir -p "${SCENE_PYTHONPATH_DIR}"
        ln -sfn "${isaacteleop_package_dir}" "${SCENE_PYTHONPATH_DIR}/isaacteleop"
        check_ok "scene isaacteleop shim=${SCENE_PYTHONPATH_DIR}/isaacteleop"
    else
        check_error "scene Python cannot import isaacteleop and no teleop isaacteleop package was found"
    fi
fi

if [[ ! -f "${MODEL_PATH}" && ! -d "${MODEL_PATH}" ]]; then
    check_error "voice model path not found: ${MODEL_PATH}"
else
    check_ok "voice model=${MODEL_PATH}"
fi

if [[ -z "${DISPLAY_ARG}" ]]; then
    check_error "display is empty; pass --display :0"
else
    check_ok "capture display=${DISPLAY_ARG}"
fi

check_ok "vr output mode=${VR_OUTPUT_MODE}"

if [[ "${START_VR_OUTPUT}" -eq 1 && "${VR_OUTPUT_MODE}" == "legacy-v4l2" ]]; then
    if command -v ffmpeg >/dev/null 2>&1; then
        check_ok "ffmpeg=$(command -v ffmpeg)"
    else
        check_error "missing ffmpeg"
    fi
fi

if command -v docker >/dev/null 2>&1; then
    check_ok "docker=$(command -v docker)"
    if docker ps >/dev/null 2>&1; then
        check_ok "docker daemon is accessible"
    else
        if reexec_with_docker_group_if_possible "${ORIGINAL_ARGS[@]}"; then
            :
        else
            check_error "docker daemon is not accessible by current shell"
            check_warn "if you just joined the docker group, run: newgrp docker"
            check_warn "or fully log out and log back in, then retry this script"
        fi
    fi
else
    check_error "missing docker"
fi

if [[ "${START_VR_OUTPUT}" -eq 1 && "${VR_OUTPUT_MODE}" == "legacy-v4l2" ]]; then
    if [[ -e /dev/video44 ]]; then
        check_ok "sim screen device=/dev/video44"
        if command -v v4l2-ctl >/dev/null 2>&1; then
            video44_output_fmt="$(v4l2-ctl -d /dev/video44 --get-fmt-video-out 2>&1 || true)"
            video44_info="$(v4l2-ctl -d /dev/video44 --all 2>&1 || true)"
            if grep -Eq "Width/Height|Pixel Format" <<<"${video44_output_fmt}" \
                || grep -Eq "Video Output|Video Output Multiplanar" <<<"${video44_info}"; then
                check_ok "sim screen output capability=V4L2 output"
            else
                check_error "/dev/video44 is present but is not a V4L2 output device"
                check_warn "current output format query:"
                sed -n '1,12p' <<<"${video44_output_fmt}" >&2
                check_warn "reload it with:"
                check_warn "sudo modprobe -r v4l2loopback"
                check_warn "sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2 max_width=1920 max_height=1080"
            fi
        else
            check_warn "v4l2-ctl not found; cannot validate /dev/video44 capabilities"
        fi
    else
        check_error "sim screen device missing: /dev/video44"
        check_warn "create once with: sudo modprobe v4l2loopback video_nr=44 card_label=teleop_sim_screen exclusive_caps=1 max_buffers=2 max_width=1920 max_height=1080"
    fi
elif [[ "${START_VR_OUTPUT}" -eq 1 && "${VR_OUTPUT_MODE}" == "direct-gpu" ]]; then
    if [[ "${WITH_SCENE}" -ne 1 ]]; then
        check_error "direct-gpu VR output is in-process and requires --with-scene"
    fi
    if [[ -n "${SCENE_PYTHON_BIN}" && -x "${SCENE_PYTHON_BIN}" ]]; then
        direct_gpu_import_log="${LOG_DIR}/direct_gpu_import_check.log"
        if PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${SCENE_PYTHON_BIN}" - <<'PY' >"${direct_gpu_import_log}" 2>&1
from tools.newton_xr_direct_bridge import _load_direct_gpu_modules

_load_direct_gpu_modules()
PY
        then
            check_ok "direct-gpu runtime imports Holoscan/CuPy/XR renderer"
        else
            check_error "direct-gpu runtime imports failed; see ${direct_gpu_import_log}"
            tail -40 "${direct_gpu_import_log}" >&2 || true
        fi
    fi
else
    check_ok "sim-screen camera_streamer output disabled"
fi

if [[ ! -x "${SCRIPT_DIR}/run_cloudxr_web_client.sh" ]]; then
    check_error "missing executable: scripts/run_cloudxr_web_client.sh"
fi
if [[ ! -x "${SCRIPT_DIR}/run_quest_voice_command_bridge.sh" ]]; then
    check_error "missing executable: scripts/run_quest_voice_command_bridge.sh"
fi
if [[ "${START_VR_OUTPUT}" -eq 1 && "${VR_OUTPUT_MODE}" == "legacy-v4l2" && ! -x "${SCRIPT_DIR}/run_newton_vr_output.sh" ]]; then
    check_error "missing executable: scripts/run_newton_vr_output.sh"
fi
if [[ "${WITH_SCENE}" -eq 1 && ! -f "${REPO_ROOT}/debug/import_dual_nero_linker_l10.py" ]]; then
    check_error "missing debug/import_dual_nero_linker_l10.py"
fi
if [[ "${WITH_SCENE}" -eq 1 && ! -f "${SCENE_PHYSICS_CONFIG}" ]]; then
    check_error "missing scene physics config: ${SCENE_PHYSICS_CONFIG}"
elif [[ "${WITH_SCENE}" -eq 1 ]]; then
    check_ok "scene physics config=${SCENE_PHYSICS_CONFIG}"
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    if [[ "${failures}" -eq 0 ]]; then
        check_ok "preflight passed"
        exit 0
    fi
    check_error "preflight failed with ${failures} problem(s)"
    exit 1
fi

if [[ "${failures}" -ne 0 ]]; then
    err "fix preflight errors before starting VR prerequisites"
    exit 1
fi

PIDS=()
NAMES=()

start_bg() {
    local name="$1"
    shift
    local log_path="${LOG_DIR}/${name}.log"
    : > "${log_path}"
    log "starting ${name}; log=${log_path}"
    "$@" >"${log_path}" 2>&1 &
    local pid=$!
    PIDS+=("${pid}")
    NAMES+=("${name}")
    ok "${name} pid=${pid}"
}

last_bg_pid() {
    printf '%s\n' "${PIDS[$((${#PIDS[@]} - 1))]}"
}

require_bg_alive() {
    local name="$1"
    local pid="$2"
    local log_path="${LOG_DIR}/${name}.log"
    if kill -0 "${pid}" >/dev/null 2>&1; then
        return 0
    fi
    err "${name} exited during startup"
    err "recent ${name} log:"
    tail -80 "${log_path}" >&2 || true
    exit 1
}

cloudxr_ipc_ready() {
    local socket_path="$1"
    [[ -S "${socket_path}" ]] || return 1
    [[ -n "${TELEOP_PYTHON_BIN}" && -x "${TELEOP_PYTHON_BIN}" ]] || return 0
    "${TELEOP_PYTHON_BIN}" - "${socket_path}" >/dev/null 2>&1 <<'PY'
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

stop_stale_cloudxr_runtimes() {
    local socket_path="$1"
    local runtime_user="${USER:-$(id -un)}"
    local pgrep_user="${runtime_user}"
    local pids=()
    local pid
    if ! id -u "${pgrep_user}" >/dev/null 2>&1; then
        pgrep_user="$(id -u)"
    fi
    mapfile -t pids < <(pgrep -u "${pgrep_user}" -f 'python.*-m isaacteleop.cloudxr' || true)
    if [[ "${#pids[@]}" -eq 0 ]]; then
        return
    fi

    warn "stopping stale CloudXR runtime process(es): ${pids[*]}"
    for pid in "${pids[@]}"; do
        kill "${pid}" >/dev/null 2>&1 || true
    done

    for _ in {1..20}; do
        local alive=0
        for pid in "${pids[@]}"; do
            if kill -0 "${pid}" >/dev/null 2>&1; then
                alive=1
                break
            fi
        done
        [[ "${alive}" -eq 0 ]] && break
        sleep 0.25
    done

    if [[ -e "${socket_path}" ]]; then
        if ! cloudxr_ipc_ready "${socket_path}"; then
            warn "removing stale CloudXR IPC socket: ${socket_path}"
            rm -f "${socket_path}"
        fi
    fi
}

stop_all() {
    local exit_code=$?
    trap - EXIT INT TERM
    if [[ "${#PIDS[@]}" -gt 0 ]]; then
        warn "stopping background services..."
        for idx in "${!PIDS[@]}"; do
            local pid="${PIDS[$idx]}"
            local name="${NAMES[$idx]}"
            if kill -0 "${pid}" >/dev/null 2>&1; then
                warn "stopping ${name} pid=${pid}"
                kill "${pid}" >/dev/null 2>&1 || true
            fi
        done
        for pid in "${PIDS[@]}"; do
            wait "${pid}" >/dev/null 2>&1 || true
        done
    fi
    exit "${exit_code}"
}
trap stop_all EXIT INT TERM

wait_for_socket() {
    local socket_path="$1"
    local label="$2"
    local timeout_s="$3"
    local pid="${4:-}"
    local log_path="${5:-}"
    local started
    started="$(date +%s)"
    while true; do
        if cloudxr_ipc_ready "${socket_path}"; then
            ok "${label} is ready: ${socket_path}"
            return 0
        fi
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
            err "${label} exited before becoming ready"
            if [[ -n "${log_path}" ]]; then
                err "recent ${label} log:"
                tail -80 "${log_path}" >&2 || true
            fi
            return 1
        fi
        if (( $(date +%s) - started >= timeout_s )); then
            err "${label} did not become ready within ${timeout_s}s: ${socket_path}"
            return 1
        fi
        sleep 1
    done
}

wait_for_port() {
    local port="$1"
    local label="$2"
    local timeout_s="$3"
    local started
    started="$(date +%s)"
    while true; do
        if ss -ltn | awk '{print $4}' | grep -Eq "[:.]${port}$"; then
            ok "${label} is listening on port ${port}"
            return 0
        fi
        if (( $(date +%s) - started >= timeout_s )); then
            err "${label} did not listen on port ${port} within ${timeout_s}s"
            return 1
        fi
        sleep 1
    done
}

wait_for_https() {
    local url="$1"
    local label="$2"
    local timeout_s="$3"
    local pid="${4:-}"
    local log_path="${5:-}"
    local started
    started="$(date +%s)"
    while true; do
        if curl -kfsS --connect-timeout 2 "${url}" >/dev/null 2>&1; then
            ok "${label} is serving ${url}"
            return 0
        fi
        if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
            err "${label} exited before serving ${url}"
            if [[ -n "${log_path}" ]]; then
                err "recent ${label} log:"
                tail -80 "${log_path}" >&2 || true
            fi
            return 1
        fi
        if (( $(date +%s) - started >= timeout_s )); then
            err "${label} did not serve ${url} within ${timeout_s}s"
            return 1
        fi
        sleep 1
    done
}

if [[ "${START_CLOUDXR}" -eq 1 ]]; then
    if [[ -f "${XR_ENV_PATH}" ]]; then
        set -a
        # shellcheck disable=SC1090
        source "${XR_ENV_PATH}"
        set +a
        ok "CloudXR env loaded: ${XR_ENV_PATH}"
    fi
    export NV_CXR_RUNTIME_DIR="${NV_CXR_RUNTIME_DIR:-${HOME}/.cloudxr/run}"
    cloudxr_ipc_path="${NV_CXR_RUNTIME_DIR}/ipc_cloudxr"
    if cloudxr_ipc_ready "${cloudxr_ipc_path}"; then
        ok "CloudXR runtime already ready: ${cloudxr_ipc_path}"
    else
        stop_stale_cloudxr_runtimes "${cloudxr_ipc_path}"
        start_bg "cloudxr_runtime" "${TELEOP_PYTHON_BIN}" -m isaacteleop.cloudxr --accept-eula
        cloudxr_pid="$(last_bg_pid)"
        wait_for_socket "${cloudxr_ipc_path}" "CloudXR runtime" 90 "${cloudxr_pid}" "${LOG_DIR}/cloudxr_runtime.log"
        require_bg_alive "cloudxr_runtime" "${cloudxr_pid}"
    fi
else
    warn "skipping CloudXR runtime"
fi

if [[ "${START_WEB}" -eq 1 ]]; then
    start_bg "cloudxr_web_client" env PYTHON_BIN="${TELEOP_PYTHON_BIN}" "${SCRIPT_DIR}/run_cloudxr_web_client.sh" --mode "${WEB_MODE}"
    web_pid="$(last_bg_pid)"
    wait_for_https "https://127.0.0.1:8443/" "CloudXR web client" 120 "${web_pid}" "${LOG_DIR}/cloudxr_web_client.log"
    require_bg_alive "cloudxr_web_client" "${web_pid}"
else
    warn "skipping CloudXR web client"
fi

if [[ "${START_VOICE}" -eq 1 ]]; then
    start_bg "quest_voice_bridge" \
        env PYTHON_BIN="${TELEOP_PYTHON_BIN}" "${SCRIPT_DIR}/run_quest_voice_command_bridge.sh" \
        --model-path "${MODEL_PATH}" \
        --no-tls \
        --min-confidence "${MIN_CONFIDENCE}"
    voice_pid="$(last_bg_pid)"
    wait_for_port 8766 "Quest voice bridge" 60
    require_bg_alive "quest_voice_bridge" "${voice_pid}"
else
    warn "skipping Quest voice bridge"
fi

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-127.0.0.1}"
ok "Quest web page: https://${HOST_IP}:8443/"
ok "CloudXR certificate page if needed: https://${HOST_IP}:48322/"
if [[ "${WITH_SCENE}" -eq 1 ]]; then
    ok "scene will start in this terminal"
else
    ok "Next terminal for scene: conda activate newton && python debug/import_dual_nero_linker_l10.py --device ${SCENE_DEVICE} --quest-teleop"
fi
ok "teleop input source=${TELEOP_INPUT_SOURCE}"

if [[ "${WITH_SCENE}" -eq 1 ]]; then
    if [[ "${START_VR_OUTPUT}" -eq 1 && "${VR_OUTPUT_MODE}" == "legacy-v4l2" ]]; then
        vr_output_args=(
            --display "${DISPLAY_ARG}"
            --python "${TELEOP_PYTHON_BIN}"
        )
        if [[ "${TELEOP_INPUT_SOURCE}" == "quest" ]]; then
            vr_output_args+=(--disable-hand-log)
        fi
        start_bg "newton_vr_output" "${SCRIPT_DIR}/run_newton_vr_output.sh" "${vr_output_args[@]}"
        vr_output_pid="${PIDS[$((${#PIDS[@]} - 1))]}"
        sleep 3
        if ! kill -0 "${vr_output_pid}" >/dev/null 2>&1; then
            err "newton_vr_output exited during startup; Quest hand overlay will be unavailable"
            err "recent newton_vr_output log:"
            tail -80 "${LOG_DIR}/newton_vr_output.log" >&2 || true
            exit 1
        fi
    elif [[ "${START_VR_OUTPUT}" -eq 1 && "${VR_OUTPUT_MODE}" == "direct-gpu" ]]; then
        ok "direct-gpu VR output will run inside the Newton scene process"
    else
        warn "skipping camera_streamer VR output; sim-screen XR plane and overlay hand skeleton are disabled"
        if [[ "${TELEOP_INPUT_SOURCE}" == "overlay-log" ]]; then
            err "--teleop-input-source overlay-log requires --with-vr-output to produce hand samples"
            exit 2
        fi
    fi
    XR_STATUS_PATH="${TELEOP_XR_STATUS_PATH:-${NV_CXR_RUNTIME_DIR:-${HOME}/.cloudxr/run}/teleop_xr_status.json}"
    XR_HAND_LOG_PATH="${TELEOP_XR_HAND_LOG_PATH:-${REPO_ROOT}/logs/xr_debug/camera_overlay_hand.jsonl}"
    export TELEOP_XR_STATUS_PATH="${XR_STATUS_PATH}"
    export TELEOP_XR_HAND_LOG_PATH="${XR_HAND_LOG_PATH}"
    scene_mode_args=()
    if [[ "${VR_OUTPUT_MODE}" == "direct-gpu" ]]; then
        scene_mode_args+=(
            --direct-gpu-capture-fps "${NEWTON_DIRECT_GPU_CAPTURE_FPS:-60}"
            --viewer-camera-source "${NEWTON_VIEWER_CAMERA_SOURCE:-d455}"
            --no-d455-preview
            --no-d405-preview
            --no-d455-opencv-window
            --no-d405-opencv-window
        )
        if [[ "${NEWTON_DIRECT_GPU_FLIP_X:-1}" == "0" ]]; then
            scene_mode_args+=(--no-direct-gpu-flip-x)
        else
            scene_mode_args+=(--direct-gpu-flip-x)
        fi
        if [[ "${NEWTON_DIRECT_GPU_HEADLESS:-1}" == "0" ]]; then
            scene_mode_args+=(--no-headless)
            warn "direct-gpu visible viewer requested; NVIDIA EGL/CUDA interop may fall back to the slower X11 path"
        else
            scene_mode_args+=(--headless)
        fi
    fi
    scene_perf_args=()
    if [[ -n "${NEWTON_SCENE_FPS:-}" ]]; then
        scene_perf_args+=(--fps "${NEWTON_SCENE_FPS}")
    fi
    if [[ -n "${NEWTON_SCENE_SUBSTEPS:-}" ]]; then
        scene_perf_args+=(--substeps "${NEWTON_SCENE_SUBSTEPS}")
    fi
    if [[ -n "${NEWTON_SOLVER_ITERATIONS:-}" ]]; then
        scene_perf_args+=(--solver-iterations "${NEWTON_SOLVER_ITERATIONS}")
    fi
    if [[ -n "${NEWTON_TELEOP_LOOP_HZ:-}" ]]; then
        scene_perf_args+=(--teleop-loop-hz "${NEWTON_TELEOP_LOOP_HZ}")
    fi
    scene_bottle_args=(
        --scene-physics-config "${SCENE_PHYSICS_CONFIG}"
        --dynamic-object-shape "${NEWTON_DYNAMIC_OBJECT_SHAPE:-cylinder}"
    )
    if [[ -n "${NEWTON_DYNAMIC_BOTTLE_SPEC:-}" ]]; then
        scene_bottle_args+=(--dynamic-bottle-spec "${NEWTON_DYNAMIC_BOTTLE_SPEC}")
    fi
    scene_contact_args=()
    if [[ -n "${NEWTON_DRIVE_HAND_TARGET_KE:-}" ]]; then
        scene_contact_args+=(--drive-hand-target-ke "${NEWTON_DRIVE_HAND_TARGET_KE}")
    fi
    if [[ -n "${NEWTON_DRIVE_HAND_TARGET_KD:-}" ]]; then
        scene_contact_args+=(--drive-hand-target-kd "${NEWTON_DRIVE_HAND_TARGET_KD}")
    fi
    if [[ -n "${NEWTON_DRIVE_HAND_EFFORT_LIMIT:-}" ]]; then
        scene_contact_args+=(--drive-hand-effort-limit "${NEWTON_DRIVE_HAND_EFFORT_LIMIT}")
    fi
    if [[ -n "${NEWTON_L10_FRICTION:-}" ]]; then
        scene_contact_args+=(--l10-friction "${NEWTON_L10_FRICTION}")
    fi
    if [[ -n "${NEWTON_L10_TORSIONAL_FRICTION:-}" ]]; then
        scene_contact_args+=(--l10-torsional-friction "${NEWTON_L10_TORSIONAL_FRICTION}")
    fi
    if [[ -n "${NEWTON_L10_ROLLING_FRICTION:-}" ]]; then
        scene_contact_args+=(--l10-rolling-friction "${NEWTON_L10_ROLLING_FRICTION}")
    fi
    if [[ -n "${NEWTON_L10_BOTTLE_CONTACT_STOP_ACTIVATION:-}" ]]; then
        scene_contact_args+=(--l10-bottle-contact-stop-activation "${NEWTON_L10_BOTTLE_CONTACT_STOP_ACTIVATION}")
    fi
    if [[ -n "${NEWTON_L10_BOTTLE_CONTACT_STOP_PENETRATION:-}" ]]; then
        scene_contact_args+=(--l10-bottle-contact-stop-penetration "${NEWTON_L10_BOTTLE_CONTACT_STOP_PENETRATION}")
    fi
    if [[ -n "${NEWTON_L10_BOTTLE_CONTACT_STOP_RELEASE:-}" ]]; then
        scene_contact_args+=(--l10-bottle-contact-stop-release "${NEWTON_L10_BOTTLE_CONTACT_STOP_RELEASE}")
    fi
    if [[ -n "${NEWTON_L10_BOTTLE_CONTACT_STOP_RETREAT_RAD:-}" ]]; then
        scene_contact_args+=(--l10-bottle-contact-stop-retreat-rad "${NEWTON_L10_BOTTLE_CONTACT_STOP_RETREAT_RAD}")
    fi
    if [[ -n "${NEWTON_L10_BOTTLE_CONTACT_RELEASE_RETREAT_RAD:-}" ]]; then
        scene_contact_args+=(--l10-bottle-contact-release-retreat-rad "${NEWTON_L10_BOTTLE_CONTACT_RELEASE_RETREAT_RAD}")
    fi
    if [[ "${NEWTON_TELEOP_HAND_PUBLISH_KINEMATIC_VELOCITY:-0}" == "1" ]]; then
        scene_contact_args+=(--teleop-hand-publish-kinematic-velocity)
    else
        scene_contact_args+=(--no-teleop-hand-publish-kinematic-velocity)
    fi
    log "starting Newton scene in foreground; press Ctrl+C here to stop the whole stack"
    scene_pythonpath="${PYTHONPATH:-}"
    if [[ -n "${SCENE_PYTHONPATH_DIR}" ]]; then
        scene_pythonpath="${SCENE_PYTHONPATH_DIR}${scene_pythonpath:+:${scene_pythonpath}}"
    fi
    DISPLAY="${DISPLAY_ARG}" PYTHONPATH="${scene_pythonpath}" "${SCENE_PYTHON_BIN}" "${REPO_ROOT}/debug/import_dual_nero_linker_l10.py" \
        --device "${SCENE_DEVICE}" \
        --quest-teleop \
        --teleop-input-source "${TELEOP_INPUT_SOURCE}" \
        --vr-output-mode "${VR_OUTPUT_MODE}" \
        --teleop-overlay-hand-log-path "${XR_HAND_LOG_PATH}" \
        --teleop-startup-timeout-s "${TELEOP_STARTUP_TIMEOUT_S:-300}" \
        --teleop-xr-status-path "${XR_STATUS_PATH}" \
        --no-capture-graph \
        --no-viewer-contacts \
        --no-viewer-hydro-contact-surface \
        "${scene_mode_args[@]}" \
        "${scene_perf_args[@]}" \
        "${scene_bottle_args[@]}" \
        "${scene_contact_args[@]}" \
        "${SCENE_ARGS[@]}"
elif [[ "${START_VR_OUTPUT}" -eq 1 && "${VR_OUTPUT_MODE}" == "legacy-v4l2" ]]; then
    log "starting VR output in foreground; press Ctrl+C here to stop the whole stack"
    "${SCRIPT_DIR}/run_newton_vr_output.sh" --display "${DISPLAY_ARG}" --python "${TELEOP_PYTHON_BIN}"
elif [[ "${START_VR_OUTPUT}" -eq 1 && "${VR_OUTPUT_MODE}" == "direct-gpu" ]]; then
    err "direct-gpu VR output is in-process and cannot run with --no-scene"
    exit 2
else
    warn "skipping VR output; background services are running, press Ctrl+C to stop"
    while true; do
        sleep 3600
    done
fi
