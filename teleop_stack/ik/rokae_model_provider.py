from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from teleop_stack.ik.differential_ik import PositionJacobian, SpatialJacobian
from teleop_stack.ik.rokae_kinematics import RokaeModelApiLike, RokaeModelProviderConfig
from teleop_stack.models import Pose7
from teleop_stack.paths import repo_root


DEFAULT_MODEL_HELPER_BIN = repo_root() / "build" / "rokae_rci" / "rokae_rci_model_helper"
DEFAULT_MODEL_HELPER_BUILD_SCRIPT = repo_root() / "scripts" / "build_rokae_rci_model_helper.sh"


@dataclass
class BuiltinRokaeModelProvider(RokaeModelApiLike):
    config: RokaeModelProviderConfig
    _process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _cached_q: tuple[float, ...] | None = field(default=None, init=False, repr=False)
    _cached_pose: Pose7 | None = field(default=None, init=False, repr=False)
    _cached_jacobian: PositionJacobian | None = field(default=None, init=False, repr=False)
    _cached_spatial_jacobian: SpatialJacobian | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        helper_bin = self._resolve_helper_binary()
        robot_ip = self._resolve_robot_ip()
        helper_cmd = [
            str(helper_bin),
            "--robot-ip",
            robot_ip,
            "--udp-port",
            str(int(self.config.robot_udp_port)),
            "--xmate-type",
            str(self.config.xmate_type_name),
        ]
        if self.config.helper_use_sudo:
            helper_cmd = ["sudo", "-n", *helper_cmd]
        self._process = subprocess.Popen(
            helper_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def get_cart_pose(self, joint_positions_rad: tuple[float, ...]) -> Pose7:
        pose, _, _ = self._query(joint_positions_rad)
        return pose

    def get_position_jacobian(self, joint_positions_rad: tuple[float, ...]) -> PositionJacobian:
        _, jacobian, _ = self._query(joint_positions_rad)
        return jacobian

    def get_spatial_jacobian(self, joint_positions_rad: tuple[float, ...]) -> SpatialJacobian:
        _, _, spatial_jacobian = self._query(joint_positions_rad)
        if spatial_jacobian is None:
            raise RuntimeError(
                "Rokae model helper response did not include a 6x7 spatial Jacobian. "
                "Rebuild the helper binary so it emits 42 Jacobian values."
            )
        return spatial_jacobian

    def close(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        if process.stdin is not None:
            process.stdin.close()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return None

    def _resolve_helper_binary(self) -> Path:
        if self.config.helper_binary_path is not None:
            helper_bin = Path(self.config.helper_binary_path)
            if not helper_bin.is_file():
                raise FileNotFoundError(f"Configured Rokae model helper binary does not exist: {helper_bin}")
            return helper_bin.resolve()

        if DEFAULT_MODEL_HELPER_BIN.is_file():
            return DEFAULT_MODEL_HELPER_BIN.resolve()

        build_script = Path(self.config.helper_build_script_path) if self.config.helper_build_script_path is not None else DEFAULT_MODEL_HELPER_BUILD_SCRIPT
        if not build_script.is_file():
            raise FileNotFoundError(
                f"Rokae model helper build script does not exist: {build_script}"
            )

        env = os.environ.copy()
        if self.config.sdk_root is not None:
            env["ROKAE_RCI_ROOT"] = str(Path(self.config.sdk_root).resolve())

        completed = subprocess.run(
            [str(build_script)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=repo_root(),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to build the built-in Rokae model helper. "
                f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
            )

        helper_bin = Path(completed.stdout.strip())
        if not helper_bin.is_file():
            raise RuntimeError(
                "Rokae model helper build script completed, but did not produce a valid binary path. "
                f"stdout:\n{completed.stdout}"
            )
        return helper_bin.resolve()

    def _resolve_robot_ip(self) -> str:
        if self.config.robot_ip:
            return str(self.config.robot_ip)
        env_value = os.environ.get("ROKAE_ROBOT_IP")
        if env_value:
            return env_value
        raise RuntimeError(
            "Rokae model-backed host IK requires a robot IP. "
            "Pass --rokae-host-ik-robot-ip or export ROKAE_ROBOT_IP."
        )

    def _query(self, joint_positions_rad: tuple[float, ...]) -> tuple[Pose7, PositionJacobian, SpatialJacobian | None]:
        if self._cached_q == joint_positions_rad and self._cached_pose is not None and self._cached_jacobian is not None:
            return self._cached_pose, self._cached_jacobian, self._cached_spatial_jacobian

        process = self._require_process()
        if process.stdin is None or process.stdout is None:
            raise RuntimeError("Rokae model helper subprocess is missing stdin/stdout pipes.")

        query_line = " ".join(f"{float(value):.17g}" for value in joint_positions_rad)
        try:
            process.stdin.write(query_line + "\n")
            process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError(self._build_helper_failure_message("helper subprocess closed stdin unexpectedly")) from exc

        response_line = self._read_next_protocol_line(process)

        pose, jacobian, spatial_jacobian = _parse_helper_response(response_line)
        self._cached_q = tuple(float(value) for value in joint_positions_rad)
        self._cached_pose = pose
        self._cached_jacobian = jacobian
        self._cached_spatial_jacobian = spatial_jacobian
        return pose, jacobian, spatial_jacobian

    def _read_next_protocol_line(self, process: subprocess.Popen[str]) -> str:
        if process.stdout is None:
            raise RuntimeError("Rokae model helper subprocess is missing stdout pipe.")
        skipped_lines: list[str] = []
        while True:
            response_line = process.stdout.readline()
            if not response_line:
                self._finalize_process_on_stdout_eof(process)
                suffix = f" Skipped stdout lines before EOF: {skipped_lines!r}" if skipped_lines else ""
                raise RuntimeError(self._build_helper_failure_message(f"helper subprocess produced no response.{suffix}"))
            if response_line.startswith("ok "):
                return response_line
            stripped = response_line.strip()
            if stripped.startswith("port:") or stripped.startswith("ip:"):
                skipped_lines.append(stripped)
                continue
            raise RuntimeError(_build_invalid_helper_response_message(response_line))

    def _require_process(self) -> subprocess.Popen[str]:
        if self._process is None:
            raise RuntimeError("Rokae model helper subprocess is not running.")
        if self._process.poll() is not None:
            if self.config.helper_use_sudo and self._process.returncode == 1:
                raise RuntimeError(
                    self._build_helper_failure_message(
                        "helper subprocess exited unexpectedly. If sudo was requested, refresh credentials first with 'sudo -v'"
                    )
                )
            raise RuntimeError(self._build_helper_failure_message("helper subprocess exited unexpectedly"))
        return self._process

    def _finalize_process_on_stdout_eof(self, process: subprocess.Popen[str]) -> None:
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            return

    def _build_helper_failure_message(self, reason: str) -> str:
        stderr_tail = ""
        return_code = None
        if (
            self._process is not None
            and self._process.stderr is not None
        ):
            try:
                return_code = self._process.poll()
                if return_code is None:
                    return_code = self._process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                return_code = None
            except Exception:
                return_code = None
            try:
                stderr_tail = self._process.stderr.read()
            except Exception:
                stderr_tail = ""
        code_suffix = f" returncode={return_code}." if return_code is not None else ""
        suffix = f"\nstderr:\n{stderr_tail}" if stderr_tail else ""
        return f"Rokae model helper failure: {reason}.{code_suffix}{suffix}"


def create_rokae_model_provider(provider_config: RokaeModelProviderConfig) -> RokaeModelApiLike:
    return BuiltinRokaeModelProvider(provider_config)


def _build_invalid_helper_response_message(response_line: str) -> str:
    return (
        "Rokae model helper returned an invalid response line. "
        f"Expected 'ok <7 pose values> <21 or 42 Jacobian values>', got: {response_line!r}"
    )


def _parse_helper_response(response_line: str) -> tuple[Pose7, PositionJacobian, SpatialJacobian | None]:
    tokens = response_line.strip().split()
    if len(tokens) not in {1 + 7 + 21, 1 + 7 + 42} or tokens[0] != "ok":
        raise RuntimeError(_build_invalid_helper_response_message(response_line))

    values = [float(token) for token in tokens[1:]]
    pose = Pose7(
        position_xyz=(values[0], values[1], values[2]),
        quaternion_xyzw=(values[3], values[4], values[5], values[6]),
    )
    jacobian_values = values[7:]
    jacobian: PositionJacobian = (
        tuple(jacobian_values[0:7]),
        tuple(jacobian_values[7:14]),
        tuple(jacobian_values[14:21]),
    )
    spatial_jacobian: SpatialJacobian | None = None
    if len(jacobian_values) == 42:
        spatial_jacobian = (
            tuple(jacobian_values[0:7]),
            tuple(jacobian_values[7:14]),
            tuple(jacobian_values[14:21]),
            tuple(jacobian_values[21:28]),
            tuple(jacobian_values[28:35]),
            tuple(jacobian_values[35:42]),
        )
    return pose, jacobian, spatial_jacobian
