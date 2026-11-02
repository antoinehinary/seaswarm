"""Planner interface and the data it exchanges with the mission coordinator."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from seaswarm.config import PlannerConfig, SensorConfig, VehicleTypeConfig
from seaswarm.safety.rules import XY, Geofence
from seaswarm.search.pod import SensorModel, sweep_width_m
from seaswarm.search.probability_map import ProbabilityMap
from seaswarm.vehicles.backend import VehicleState, VehicleType


@dataclass(frozen=True)
class VehicleCapability:
    """What a planner needs to know about a vehicle's search performance."""

    vehicle_id: str
    vehicle_type: VehicleType
    cruise_speed_mps: float
    sweep_width_m: float
    track_spacing_m: float
    config: VehicleTypeConfig

    @classmethod
    def build(
        cls,
        vehicle_id: str,
        vehicle_type: VehicleType,
        config: VehicleTypeConfig,
        sensor: SensorConfig,
        daylight: bool,
        altitude_m: float,
        planner: PlannerConfig,
    ) -> VehicleCapability:
        """Capability of a vehicle flying its sensor at `altitude_m` at cruise speed."""
        model = SensorModel.from_config(sensor, daylight, altitude_m)
        width = sweep_width_m(model, config.cruise_speed_mps)
        spacing = width * (1.0 - planner.track_overlap)
        return cls(vehicle_id, vehicle_type, config.cruise_speed_mps, width, spacing, config)

    @property
    def sweep_rate_m2_s(self) -> float:
        """Area swept per second at cruise, m^2/s."""
        return self.sweep_width_m * self.cruise_speed_mps


@dataclass(frozen=True)
class Detection:
    """A reported detection of the person (belief side: measured position, never truth)."""

    detection_id: str
    t_s: float
    vehicle_id: str
    x_m: float
    y_m: float


@dataclass(frozen=True)
class PriorityArea:
    """Operator preference: multiply the utility of tiles centred in this rectangle by `weight`."""

    area_id: str
    x_min_m: float
    y_min_m: float
    x_max_m: float
    y_max_m: float
    weight: float
    expires_s: float


@dataclass(frozen=True)
class OperatorPrefs:
    """Active operator preferences. They change planner utility, never the probabilities."""

    areas: tuple[PriorityArea, ...] = ()

    def weights(self, x_m: NDArray[np.float64], y_m: NDArray[np.float64], t_s: float) -> NDArray:
        """Utility weight at each point: the latest active area covering it, else 1."""
        w = np.ones(np.shape(x_m))
        for a in self.areas:
            if t_s < a.expires_s:
                inside = (
                    (x_m >= a.x_min_m)
                    & (x_m <= a.x_max_m)
                    & (y_m >= a.y_min_m)
                    & (y_m <= a.y_max_m)
                )
                w = np.where(inside, a.weight, w)
        return w


@dataclass(frozen=True)
class MissionView:
    """Everything a planner may know besides the belief. No truth fields."""

    t_s: float
    search_start_s: float
    base_xy_m: XY
    fence: Geofence
    capabilities: dict[str, VehicleCapability]
    current_tiles: dict[str, str | None]
    drift_mps: XY  # mean drift of the belief (forecast current + leeway)
    detections: tuple[Detection, ...] = ()


@dataclass(frozen=True)
class SearchTask:
    """Search one tile (or strip) along a lawnmower path."""

    tile_id: str | None
    waypoints: tuple[XY, ...]
    expected_pos_gain: float
    eta_s: float


@dataclass(frozen=True)
class Assignment:
    """Ordered tasks for one vehicle."""

    vehicle_id: str
    tasks: tuple[SearchTask, ...] = field(default_factory=tuple)


class ReplanTrigger(StrEnum):
    """Why the coordinator asks the planner for a new plan."""

    START = "start"
    PERIODIC = "periodic"
    VEHICLE_IDLE = "vehicle_idle"
    LOW_ENERGY = "low_energy"
    DETECTION = "detection"
    PRIORITY = "priority"
    REFUSAL = "refusal"
    RETURNED = "returned"


class Planner(Protocol):
    """Search allocation strategy."""

    name: str

    def plan(
        self,
        view: MissionView,
        pmap: ProbabilityMap,
        vehicles: Sequence[VehicleState],
        prefs: OperatorPrefs,
    ) -> dict[str, Assignment]:
        """Assignments for (some of) the available vehicles; others keep their current task."""
        ...
