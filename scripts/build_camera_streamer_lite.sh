#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_TELEOP_ROOT_ARG="${ISAAC_TELEOP_ROOT:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
IMAGE_TAG="${NEWTON_CAMERA_STREAMER_LITE_IMAGE:-newton-camera-streamer-lite:latest}"
BASE_TAG="${NEWTON_CAMERA_STREAMER_LITE_BASE_IMAGE:-newton-camera-streamer-lite:base}"
SKIP_DEPTHAI="true"
NO_CACHE="false"

usage() {
    cat <<'EOF'
Usage: scripts/build_camera_streamer_lite.sh [options]

Builds a V4L2-only IsaacTeleop camera_streamer image with the migrated XR hand
skeleton overlay. This avoids the upstream Dockerfile's BuildKit frontend and
unconditional ZED SDK install.

Options:
  --isaac-teleop-root PATH    IsaacTeleop checkout (default: ISAAC_TELEOP_ROOT or ../IsaacTeleop)
  --image-tag TAG             Output image tag (default: newton-camera-streamer-lite:latest)
  --base-tag TAG              Temporary base tag (default: newton-camera-streamer-lite:base)
  --with-depthai              Keep depthai Python dependency
  --no-cache                  Build Docker base image without cache
  --python PATH               Python executable (default: python3, then python)
  -h, --help                  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --isaac-teleop-root) ISAAC_TELEOP_ROOT_ARG="$2"; shift 2 ;;
        --image-tag) IMAGE_TAG="$2"; shift 2 ;;
        --base-tag) BASE_TAG="$2"; shift 2 ;;
        --with-depthai) SKIP_DEPTHAI="false"; shift ;;
        --no-cache) NO_CACHE="true"; shift ;;
        --python) PYTHON_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$PYTHON_BIN" ]]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        echo "missing Python executable" >&2
        exit 1
    fi
fi

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
if [[ ! -d "$CAMERA_STREAMER_ROOT" ]]; then
    echo "camera_streamer directory not found: $CAMERA_STREAMER_ROOT" >&2
    exit 1
fi

case "$(uname -m)" in
    x86_64)
        TARGETARCH="amd64"
        BASE_IMAGE="nvcr.io/nvidia/clara-holoscan/holoscan:v3.11.0-cuda12-dgpu"
        ;;
    aarch64|arm64)
        TARGETARCH="arm64"
        BASE_IMAGE="nvcr.io/nvidia/clara-holoscan/holoscan:v3.11.0-cuda13"
        ;;
    *)
        echo "Unsupported host architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

if ! docker ps >/dev/null 2>&1; then
    echo "docker daemon is not accessible by current user" >&2
    exit 1
fi

if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    echo "Docker NVIDIA runtime is not configured" >&2
    exit 1
fi

TEMP_CONTEXT_DIR="$(mktemp -d)"
TEMP_DOCKERFILE="$(mktemp)"
cleanup() {
    rm -rf "$TEMP_CONTEXT_DIR"
    rm -f "$TEMP_DOCKERFILE"
}
trap cleanup EXIT

echo "[camera-lite] preparing temporary camera_streamer context"
cp -a "$CAMERA_STREAMER_ROOT/." "$TEMP_CONTEXT_DIR/"
rm -rf "$TEMP_CONTEXT_DIR/build"

"$PYTHON_BIN" "$REPO_ROOT/tools/apply_camera_streamer_overlay.py" \
    --camera-streamer-root "$TEMP_CONTEXT_DIR" >/dev/null

"$PYTHON_BIN" "$REPO_ROOT/tools/generate_camera_streamer_lite_dockerfile.py" \
    --upstream-dockerfile "$TEMP_CONTEXT_DIR/Dockerfile" \
    --output "$TEMP_DOCKERFILE" \
    --base-image "$BASE_IMAGE" \
    --target-arch "$TARGETARCH" \
    --skip-zed >/dev/null
cp "$TEMP_DOCKERFILE" "$TEMP_CONTEXT_DIR/Dockerfile"

pyproject_args=(--input "$TEMP_CONTEXT_DIR/pyproject.toml" --output "$TEMP_CONTEXT_DIR/pyproject.toml")
if [[ "$SKIP_DEPTHAI" == "true" ]]; then
    pyproject_args+=(--skip-depthai)
fi
"$PYTHON_BIN" "$REPO_ROOT/tools/generate_camera_streamer_lite_pyproject.py" "${pyproject_args[@]}" >/dev/null

docker_build_args=(-t "$BASE_TAG")
if [[ "$NO_CACHE" == "true" ]]; then
    docker_build_args+=(--no-cache)
fi

echo "[camera-lite] building base image: $BASE_TAG"
DOCKER_BUILDKIT=0 docker build "${docker_build_args[@]}" "$TEMP_CONTEXT_DIR"

BUILD_CONTAINER="newton-camera-streamer-lite-build"
HOST_BUILD_DIR="$CAMERA_STREAMER_ROOT/build/newton_lite"
mkdir -p "$HOST_BUILD_DIR"
docker rm "$BUILD_CONTAINER" >/dev/null 2>&1 || true

echo "[camera-lite] compiling C++ operators"
docker run --gpus all --name "$BUILD_CONTAINER" \
    --network=host \
    --user "$(id -u):$(id -g)" \
    -e "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}" \
    -e "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-graphics,video,compute,utility,display}" \
    -v "$HOST_BUILD_DIR:/camera_streamer/build" \
    "$BASE_TAG" \
    bash -lc "
        set -e
        find /camera_streamer/build -mindepth 1 -maxdepth 1 -exec rm -rf {} +
        cd /camera_streamer/build
        cmake /camera_streamer -GNinja -Wno-dev \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
            -DBUILD_ENCODER=ON \
            -DBUILD_DECODER=ON \
            -DBUILD_XR=ON \
            -DPYTHON_LIB_OUTPUT_DIR=/camera_streamer/build/python
        ninja
        for lib_path in \
            /camera_streamer/build/operators/nv_stream_decoder/libnv_stream_decoder.so \
            /camera_streamer/build/operators/xr_plane_renderer/libxr_plane_renderer.so \
            /camera_streamer/build/_deps/openxr-sdk-source-build/src/loader/libopenxr_loader.so \
            /camera_streamer/build/_deps/openxr-sdk-source-build/src/loader/libopenxr_loader.so.1 \
            /camera_streamer/build/_deps/openxr-sdk-source-build/src/loader/libopenxr_loader.so.1.0 \
            /camera_streamer/build/_deps/openxr-sdk-source-build/src/loader/libopenxr_loader.so.1.0.26; do
            if [[ -f \"\${lib_path}\" ]]; then
                cp -a \"\${lib_path}\" /camera_streamer/build/python/
            fi
        done
        cp -a /camera_streamer/build/python /camera_streamer/python
    "

docker commit \
    --change 'USER root' \
    --change 'ENTRYPOINT ["/usr/local/bin/camera-sender-entrypoint"]' \
    --change 'CMD ["/bin/bash"]' \
    "$BUILD_CONTAINER" "$IMAGE_TAG" >/dev/null
docker rm "$BUILD_CONTAINER" >/dev/null
docker rmi "$BASE_TAG" >/dev/null 2>&1 || true

echo "[camera-lite] ready: $IMAGE_TAG"
