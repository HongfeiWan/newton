from __future__ import annotations

from abc import ABC, abstractmethod

from teleop_stack.models import SingleArmTeleopCommand


class RobotInterface(ABC):
    @abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send_command(self, command: SingleArmTeleopCommand) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError
