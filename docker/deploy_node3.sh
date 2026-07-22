#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REMOTE="${REMOTE:-user@node3}"
REMOTE_DIR="${REMOTE_DIR:-~/project/newton}"

rsync -az --delete \
    --exclude .venv \
    --exclude docker/wheelhouse \
    "${REPO_DIR}/" "${REMOTE}:${REMOTE_DIR}/"

ssh "${REMOTE}" "cd ${REMOTE_DIR} && docker/create_local_ubuntu_base.sh && docker/download_wheels.sh && docker/build_and_test.sh"
