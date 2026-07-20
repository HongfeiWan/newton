from __future__ import annotations

import argparse
from pathlib import Path

DEPTHAI_DEPENDENCY_LINE = '    "depthai",\n'


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a patched camera_streamer pyproject for RealSense-only/V4L2-only bring-up.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input pyproject.toml path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output pyproject.toml path.",
    )
    parser.add_argument(
        "--skip-depthai",
        action="store_true",
        help="Remove the depthai dependency from the generated pyproject.",
    )
    return parser


def _remove_depthai_dependency(source: str) -> str:
    if DEPTHAI_DEPENDENCY_LINE not in source:
        raise RuntimeError("Failed to locate depthai dependency in upstream pyproject.toml.")
    return source.replace(DEPTHAI_DEPENDENCY_LINE, "", 1)


def main() -> int:
    args = _build_parser().parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    source = input_path.read_text(encoding="utf-8")
    patched = _remove_depthai_dependency(source) if args.skip_depthai else source

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(patched, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
