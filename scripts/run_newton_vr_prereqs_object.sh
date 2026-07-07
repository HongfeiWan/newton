#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OBJECT="cube"
CUBE_SIZE_X="0.05"
CUBE_SIZE_Y="0.05"
CUBE_SIZE_Z="0.15"
PREREQ_ARGS=()
SCENE_ARGS=()

usage() {
    cat <<'EOF'
Usage: scripts/run_newton_vr_prereqs_object.sh [options] [-- scene-options]

Starts the same stack as run_newton_vr_prereqs.sh, but selects the dynamic
grasp object for debug scenes.

Object options:
  --object bottle|cube   Dynamic object to load (default: cube)
  --cube-size-x M        Cube/box X size [m] (default: 0.05)
  --cube-size-y M        Cube/box Y size [m] (default: 0.05)
  --cube-size-z M        Cube/box Z size [m] (default: 0.15)

All other options before -- are passed to run_newton_vr_prereqs.sh.
All options after -- are passed to debug/import_dual_nero_linker_l10.py.

Examples:
  scripts/run_newton_vr_prereqs_object.sh --display :0
  scripts/run_newton_vr_prereqs_object.sh --object bottle --display :0
  scripts/run_newton_vr_prereqs_object.sh --display :0 -- --dynamic-bottle-mass 0.5
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --object)
            OBJECT="$2"
            shift 2
            ;;
        --cube-size-x)
            CUBE_SIZE_X="$2"
            shift 2
            ;;
        --cube-size-y)
            CUBE_SIZE_Y="$2"
            shift 2
            ;;
        --cube-size-z)
            CUBE_SIZE_Z="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            SCENE_ARGS=("$@")
            break
            ;;
        *)
            PREREQ_ARGS+=("$1")
            shift
            ;;
    esac
done

case "${OBJECT}" in
    bottle)
        OBJECT_SCENE_ARGS=(--dynamic-object-shape cylinder)
        ;;
    cube)
        OBJECT_SCENE_ARGS=(
            --dynamic-object-shape box
            --dynamic-box-size-x "${CUBE_SIZE_X}"
            --dynamic-box-size-y "${CUBE_SIZE_Y}"
            --dynamic-box-size-z "${CUBE_SIZE_Z}"
        )
        ;;
    *)
        printf '[vr-prereqs-object] error: --object must be bottle or cube, got %q\n' "${OBJECT}" >&2
        exit 2
        ;;
esac

exec "${SCRIPT_DIR}/run_newton_vr_prereqs.sh" "${PREREQ_ARGS[@]}" -- "${SCENE_ARGS[@]}" "${OBJECT_SCENE_ARGS[@]}"
