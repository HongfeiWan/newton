#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAAC_TELEOP_ROOT="${ISAAC_TELEOP_ROOT:-${REPO_ROOT}/../IsaacTeleop}"
CHECK_ONLY=0
SKIP_BUILD=0
MODE="image"
IMPORTED_IMAGE="${IMPORTED_IMAGE:-cloudxr-web-app:latest}"
IMPORTED_WEBXR_DIR="${IMPORTED_WEBXR_DIR:-${HOME}/.cache/teleop_stack/cloudxr_web_client_remote/webxr_client}"

usage() {
    cat <<'EOF'
Usage: scripts/run_cloudxr_web_client.sh [--check-only] [--mode image|image-static|local|docker] [--skip-build]

Starts the full CloudXR Web Client on https://<host-ip>:8443/.
Port 48322 is only the CloudXR certificate/WSS proxy endpoint.
EOF
}

log() { printf '[cloudxr-web] %s\n' "$*"; }
ok() { log "ok: $*"; }
warn() { log "warn: $*"; }
err() { log "error: $*" >&2; }

WEBPACK_WATCH_ENV=(
    CHOKIDAR_USEPOLLING="${CHOKIDAR_USEPOLLING:-true}"
    WATCHPACK_POLLING="${WATCHPACK_POLLING:-true}"
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --check-only) CHECK_ONLY=1; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        --image) IMPORTED_IMAGE="$2"; shift 2 ;;
        --webxr-dir) IMPORTED_WEBXR_DIR="$2"; shift 2 ;;
        --skip-build) SKIP_BUILD=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) err "unknown argument: $1"; usage; exit 2 ;;
    esac
done

if [[ "${MODE}" != "image" && "${MODE}" != "image-static" && "${MODE}" != "local" && "${MODE}" != "docker" ]]; then
    err "--mode must be image, image-static, local, or docker"
    exit 2
fi

if [[ ! -d "${ISAAC_TELEOP_ROOT}" ]]; then
    err "IsaacTeleop root not found: ${ISAAC_TELEOP_ROOT}"
    err "Set ISAAC_TELEOP_ROOT=/path/to/IsaacTeleop and retry."
    exit 1
fi

CLOUDXR_DIR="${ISAAC_TELEOP_ROOT}/deps/cloudxr"
if [[ ! -f "${CLOUDXR_DIR}/docker-compose.yaml" ]]; then
    err "CloudXR docker-compose.yaml not found under ${CLOUDXR_DIR}"
    exit 1
fi

if [[ "${MODE}" == "image" || "${MODE}" == "image-static" || "${MODE}" == "docker" ]]; then
    if ! command -v docker >/dev/null 2>&1; then
        err "docker is not available"
        exit 1
    fi

    if ! docker info >/dev/null 2>&1; then
        err "docker daemon is not accessible by the current user"
        err "Run: sudo usermod -aG docker \$USER && newgrp docker"
        exit 1
    fi
    ok "docker daemon is accessible"
fi

if [[ "${MODE}" == "docker" ]]; then
    if docker compose version >/dev/null 2>&1; then
        COMPOSE=(docker compose)
    elif command -v docker-compose >/dev/null 2>&1; then
        COMPOSE=(docker-compose)
    else
        err "docker compose is not available"
        exit 1
    fi
elif [[ "${MODE}" == "local" ]]; then
    if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
        err "node/npm are required for local mode"
        exit 1
    fi
    ok "node=$(command -v node) $(node --version)"
    ok "npm=$(command -v npm) $(npm --version)"
else
    if ! docker image inspect "${IMPORTED_IMAGE}" >/dev/null 2>&1; then
        err "imported CloudXR web image not found: ${IMPORTED_IMAGE}"
        err "Copy it from the remote workstation with:"
        err "  sshpass -p '<password>' ssh zhangbt@192.168.8.109 \"docker save cloudxr-web-app:latest\" | docker load"
        exit 1
    fi
    if [[ ! -f "${IMPORTED_WEBXR_DIR}/package.json" || ! -d "${IMPORTED_WEBXR_DIR}/node_modules" ]]; then
        err "imported webxr_client cache is missing: ${IMPORTED_WEBXR_DIR}"
        err "Copy it from the running remote container with:"
        err "  mkdir -p ${HOME}/.cache/teleop_stack/cloudxr_web_client_remote"
        err "  sshpass -p '<password>' ssh zhangbt@192.168.8.109 \"docker exec cloudxr-web-app-6.1.0 sh -lc 'cd /app && tar -cf - webxr_client'\" | tar -xf - -C ${HOME}/.cache/teleop_stack/cloudxr_web_client_remote"
        exit 1
    fi
    ok "imported image=${IMPORTED_IMAGE}"
    ok "imported webxr_client=${IMPORTED_WEBXR_DIR}"
fi

GIT_ROOT="${ISAAC_TELEOP_ROOT}"
export GIT_ROOT
# shellcheck disable=SC1091
source "${ISAAC_TELEOP_ROOT}/scripts/setup_cloudxr_env.sh" >/dev/null
ok "CloudXR env: ${CXR_ENV_FILE}"

"${REPO_ROOT}/scripts/apply_isaac_teleop_cloudxr_overlay.sh" >/dev/null
ok "CloudXR web overlay applied"

CXR_WEB_SDK_VERSION="${CXR_WEB_SDK_VERSION:-6.2.0}"
SDK_TGZ="${CLOUDXR_DIR}/nvidia-cloudxr-${CXR_WEB_SDK_VERSION}.tgz"
if [[ "${MODE}" != "image" && "${MODE}" != "image-static" && ! -f "${SDK_TGZ}" ]]; then
    warn "missing CloudXR Web SDK: ${SDK_TGZ}"
    if [[ "${CHECK_ONLY}" -eq 0 ]]; then
        log "downloading/preparing CloudXR Web SDK with IsaacTeleop helper..."
        (cd "${ISAAC_TELEOP_ROOT}" && GIT_ROOT="${ISAAC_TELEOP_ROOT}" scripts/download_cloudxr_sdk.sh)
    fi
fi

if [[ "${MODE}" != "image" && "${MODE}" != "image-static" && ! -f "${SDK_TGZ}" ]]; then
    err "CloudXR Web SDK is still missing: ${SDK_TGZ}"
    err "Put nvidia-cloudxr-${CXR_WEB_SDK_VERSION}.tgz in ${CLOUDXR_DIR}, then rerun this script."
    exit 1
fi
if [[ "${MODE}" != "image" && "${MODE}" != "image-static" ]]; then
    ok "CloudXR Web SDK=${SDK_TGZ}"
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
    if ss -ltn | grep -q ':8443 '; then
        ok "8443 is already listening"
    else
        warn "8443 is not listening yet"
    fi
    exit 0
fi

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-127.0.0.1}"

if [[ "${MODE}" == "image" || "${MODE}" == "image-static" ]]; then
    CONTAINER_NAME="${CLOUDXR_WEB_CONTAINER_NAME:-cloudxr-web-app-newton}"
    IMAGE_CERT_DIR="${IMPORTED_WEBXR_DIR}/.newton_certs"
    IMAGE_CERT_FILE="${IMAGE_CERT_DIR}/web_client.crt"
    IMAGE_KEY_FILE="${IMAGE_CERT_DIR}/web_client.key"
    IMAGE_CONFIG_FILE="${IMPORTED_WEBXR_DIR}/webpack.newton.local.js"
    mkdir -p "${IMAGE_CERT_DIR}"
    if [[ ! -f "${IMAGE_CERT_FILE}" || ! -f "${IMAGE_KEY_FILE}" ]]; then
        log "generating image-mode HTTPS certificate..."
        openssl req -x509 -newkey rsa:2048 \
            -keyout "${IMAGE_KEY_FILE}" \
            -out "${IMAGE_CERT_FILE}" \
            -days 365 \
            -nodes \
            -subj "/CN=${HOST_IP}" >/dev/null 2>&1
    fi
    cat > "${IMAGE_CONFIG_FILE}" <<'EOF'
const fs = require('fs');
const { merge } = require('webpack-merge');
const dev = require('./webpack.dev.js');

module.exports = merge(dev, {
  devServer: {
    host: '0.0.0.0',
    port: 8443,
    allowedHosts: 'all',
    server: {
      type: 'https',
      options: {
        key: fs.readFileSync('/app/webxr_client/.newton_certs/web_client.key'),
        cert: fs.readFileSync('/app/webxr_client/.newton_certs/web_client.crt'),
      },
    },
    proxy: [
      {
        context: ['/quest-voice'],
        target: 'ws://127.0.0.1:8766',
        ws: true,
        changeOrigin: true,
      },
    ],
  },
});
EOF
    if [[ "${MODE}" == "image-static" ]]; then
        IMAGE_STATIC_SERVER="${IMPORTED_WEBXR_DIR}/newton_static_server.js"
        cat > "${IMAGE_STATIC_SERVER}" <<'EOF'
const fs = require('fs');
const https = require('https');
const net = require('net');
const path = require('path');

const root = '/app/webxr_client/build';
const certDir = '/app/webxr_client/.newton_certs';
const mime = {
  '.css': 'text/css',
  '.html': 'text/html; charset=utf-8',
  '.ico': 'image/x-icon',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.map': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
};

function sendFile(res, filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('not found');
      return;
    }
    res.writeHead(200, {'content-type': mime[path.extname(filePath)] || 'application/octet-stream'});
    res.end(data);
  });
}

const server = https.createServer({
  key: fs.readFileSync(path.join(certDir, 'web_client.key')),
  cert: fs.readFileSync(path.join(certDir, 'web_client.crt')),
}, (req, res) => {
  const rawPath = decodeURIComponent((req.url || '/').split('?')[0]);
  let filePath = path.normalize(path.join(root, rawPath === '/' ? 'index.html' : rawPath));
  if (!filePath.startsWith(root)) {
    res.writeHead(403);
    res.end('forbidden');
    return;
  }
  fs.stat(filePath, (err, stat) => {
    if (!err && stat.isDirectory()) {
      filePath = path.join(filePath, 'index.html');
    }
    if (!err && stat.isFile()) {
      sendFile(res, filePath);
      return;
    }
    sendFile(res, path.join(root, 'index.html'));
  });
});

server.on('upgrade', (req, socket, head) => {
  if (!req.url || !req.url.startsWith('/quest-voice')) {
    socket.destroy();
    return;
  }
  const upstream = net.connect(8766, '127.0.0.1', () => {
    upstream.write(`${req.method} ${req.url} HTTP/${req.httpVersion}\r\n`);
    for (const [key, value] of Object.entries(req.headers)) {
      upstream.write(`${key}: ${value}\r\n`);
    }
    upstream.write('\r\n');
    if (head.length) {
      upstream.write(head);
    }
    upstream.pipe(socket);
    socket.pipe(upstream);
  });
  upstream.on('error', () => socket.destroy());
});

server.listen(8443, '0.0.0.0', () => {
  console.log('[cloudxr-web-static] serving https://0.0.0.0:8443/');
});
EOF
    fi
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    ok "open this in Quest: https://${HOST_IP}:8443/"
    log "48322 is only for certificate/WSS. If needed, accept: https://${HOST_IP}:48322/"
    if [[ "${MODE}" == "image-static" ]]; then
        log "starting imported CloudXR Web Client image in static mode; press Ctrl+C to stop"
        rebuild_cmd=""
        if [[ "${NEWTON_WEB_STATIC_REBUILD:-1}" != "0" || ! -f "${IMPORTED_WEBXR_DIR}/build/index.html" ]]; then
            rebuild_cmd="npx webpack --config webpack.newton.local.js >/tmp/newton_static_webpack.log 2>&1 && "
        fi
        exec docker run --rm \
            --name "${CONTAINER_NAME}" \
            --network host \
            --user "$(id -u):$(id -g)" \
            -v "${IMPORTED_WEBXR_DIR}:/app/webxr_client" \
            --entrypoint sh \
            "${IMPORTED_IMAGE}" \
            -lc "cd /app/webxr_client && ${rebuild_cmd}exec node newton_static_server.js"
    fi
    log "starting imported CloudXR Web Client image; press Ctrl+C to stop"
    exec docker run --rm \
        --name "${CONTAINER_NAME}" \
        --network host \
        --user "$(id -u):$(id -g)" \
        -e "CHOKIDAR_USEPOLLING=${CHOKIDAR_USEPOLLING:-true}" \
        -e "WATCHPACK_POLLING=${WATCHPACK_POLLING:-true}" \
        -v "${IMPORTED_WEBXR_DIR}:/app/webxr_client" \
        --entrypoint sh \
        "${IMPORTED_IMAGE}" \
        -lc "cd /app/webxr_client && exec npx webpack serve --config webpack.newton.local.js --no-open"
fi

if [[ "${MODE}" == "docker" ]]; then
    export CXR_UID="${CXR_UID:-$(id -u)}"
    export CXR_GID="${CXR_GID:-$(id -g)}"
    export CXR_WEB_SDK_VERSION

    compose_args=(-f docker-compose.yaml up -d web-app)
    if [[ "${SKIP_BUILD}" -eq 0 ]]; then
        compose_args=(-f docker-compose.yaml up -d --build web-app)
    fi

    log "starting CloudXR Web Client container..."
    (cd "${CLOUDXR_DIR}" && "${COMPOSE[@]}" "${compose_args[@]}")

    for _ in {1..30}; do
        if curl -kfsS https://127.0.0.1:8443/ >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! curl -kfsS https://127.0.0.1:8443/ >/dev/null 2>&1; then
        err "CloudXR Web Client did not become ready on https://127.0.0.1:8443/"
        (cd "${CLOUDXR_DIR}" && "${COMPOSE[@]}" -f docker-compose.yaml logs --tail 80 web-app) || true
        exit 1
    fi

    ok "open this in Quest: https://${HOST_IP}:8443/"
    log "48322 is only for certificate/WSS. If needed, accept: https://${HOST_IP}:48322/"
    exit 0
fi

WEBXR_DIR="${CLOUDXR_DIR}/webxr_client"
CERT_DIR="${HOME}/.cloudxr/certs"
CERT_FILE="${CERT_DIR}/web_client.crt"
KEY_FILE="${CERT_DIR}/web_client.key"
mkdir -p "${CERT_DIR}"
if [[ ! -f "${CERT_FILE}" || ! -f "${KEY_FILE}" ]]; then
    log "generating local HTTPS certificate..."
    openssl req -x509 -newkey rsa:2048 \
        -keyout "${KEY_FILE}" \
        -out "${CERT_FILE}" \
        -days 365 \
        -nodes \
        -subj "/CN=${HOST_IP}" >/dev/null 2>&1
fi

CONFIG_FILE="${WEBXR_DIR}/webpack.newton.local.js"
cat > "${CONFIG_FILE}" <<EOF
const fs = require('fs');
const { merge } = require('webpack-merge');
const dev = require('./webpack.dev.js');

module.exports = merge(dev, {
  devServer: {
    host: '0.0.0.0',
    port: 8443,
    server: {
      type: 'https',
      options: {
        key: fs.readFileSync('${KEY_FILE}'),
        cert: fs.readFileSync('${CERT_FILE}'),
      },
    },
    proxy: [
      {
        context: ['/quest-voice'],
        target: 'ws://127.0.0.1:8766',
        ws: true,
        changeOrigin: true,
      },
    ],
  },
});
EOF

DEPS_HASH="$(cd "${WEBXR_DIR}" && sha256sum package.json "../nvidia-cloudxr-${CXR_WEB_SDK_VERSION}.tgz" | sha256sum | cut -d' ' -f1)"
PREVIOUS_HASH_FILE="${WEBXR_DIR}/node_modules/.newton-deps-hash"
if [[ ! -d "${WEBXR_DIR}/node_modules" || ! -f "${PREVIOUS_HASH_FILE}" || "$(cat "${PREVIOUS_HASH_FILE}" 2>/dev/null)" != "${DEPS_HASH}" ]]; then
    log "installing CloudXR Web Client npm dependencies..."
    (cd "${WEBXR_DIR}" && npm install --ignore-scripts --no-save "../nvidia-cloudxr-${CXR_WEB_SDK_VERSION}.tgz" && npm install --ignore-scripts)
    mkdir -p "${WEBXR_DIR}/node_modules"
    printf '%s\n' "${DEPS_HASH}" > "${PREVIOUS_HASH_FILE}"
fi

ok "open this in Quest: https://${HOST_IP}:8443/"
log "48322 is only for certificate/WSS. If needed, accept: https://${HOST_IP}:48322/"
log "starting local CloudXR Web Client; press Ctrl+C to stop"
exec env "${WEBPACK_WATCH_ENV[@]}" npx --prefix "${WEBXR_DIR}" webpack serve --config "${CONFIG_FILE}" --no-open
