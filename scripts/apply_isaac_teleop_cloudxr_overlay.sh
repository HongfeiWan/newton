#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -n "${ISAAC_TELEOP_ROOT:-}" ]]; then
    ISAAC_ROOT="${ISAAC_TELEOP_ROOT}"
elif [[ -d "${REPO_ROOT}/../IsaacTeleop" ]]; then
    ISAAC_ROOT="${REPO_ROOT}/../IsaacTeleop"
else
    echo "IsaacTeleop root not found. Set ISAAC_TELEOP_ROOT=/path/to/IsaacTeleop" >&2
    exit 1
fi

python3 "${REPO_ROOT}/tools/apply_isaac_teleop_cloudxr_overlay.py" \
    --isaac-teleop-root "${ISAAC_ROOT}"
