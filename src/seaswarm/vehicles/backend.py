"""Vehicle backend interface: states, commands and sensor reports.

The backend is the only layer that touches vehicles. Commands reach it only through the safety
layer (`seaswarm.safety`). The 2D simulator implements it now; a pybullet or hardware backend
(MAVLink, cflib) would implement the same Protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class VehicleType(StrEnum):
    """Kind of vehicle."""

    DRONE = "drone"
    USV = "usv"


class VehicleStatus(StrEnum):
    """What a vehicle is doing."""

    IDLE = "idle"  # holding position, available for tasks
    TRANSIT = "transit"  # following a commanded path
    RTB = "rtb"  # returning to base
    TURNAROUND = "turnaround"  # at base: battery swap or refuel
    STRANDED = "stranded"  # out of energy away from base (a safety violation)


@dataclass(frozen=True)
class VehicleState:
    """Snapshot of one vehicle. Positions ENU metres, yaw rad (CCW from east)."""

    vehicle_id: str
    vehicle_type: VehicleType
    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float
    altitude_m: float
    energy_frac: float
    status: VehicleStatus
    waypoints_left: int


@dataclass(frozen=True)
class GoTo:
    """Fly or sail straight to a point, then hold."""

    vehicle_id: str
    x_m: float
    y_m: float
    speed_mps: float


@dataclass(frozen=True)
class FollowPath:
    """Visit waypoints in order, then hold."""

    vehicle_id: str
    waypoints_m: tuple[tuple[float, float], ...]
    speed_mps: float


@dataclass(frozen=True)
class ReturnToBase:
    """Go back to base at cruise speed, then swap battery or refuel."""

    vehicle_id: str


@dataclass(frozen=True)
class Hold:
    """Stop and hold position (a drone hovers, a USV drifts with the current)."""

    vehicle_id: str


Command = GoTo | FollowPath | ReturnToBase | Hold


@dataclass(frozen=True)
class SensorReport:
    """One sensor look: where it was taken from and whether it detected the person.

    `measured_xy_m` is the noisy detected position, set only when `detected` is True.
    """

    vehicle_id: str
    sensor: str
    t_s: float
    x_m: float
    y_m: float
    altitude_m: float
    detected: bool
    measured_xy_m: tuple[float, float] | None = None


class VehicleBackend(Protocol):
    """What the mission coordinator needs from a fleet of vehicles."""

    def vehicle_ids(self) -> list[str]:
        """Ids of every vehicle, sorted."""
        ...

    def get_state(self, vehicle_id: str) -> VehicleState:
        """Current state of one vehicle."""
        ...

    def send(self, command: Command) -> None:
        """Execute a command already accepted by the safety layer."""
        ...

    def step(self, dt_s: float) -> list[SensorReport]:
        """Advance all vehicles by dt_s seconds; return the sensor looks taken meanwhile."""
        ...
