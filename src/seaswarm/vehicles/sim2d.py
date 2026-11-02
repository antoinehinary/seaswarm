"""2D simulator backend: vehicles, sensor looks and the world truth they observe.

This is the only place where vehicles meet the true person: each look is rolled against the
true position, and the planner only ever receives the resulting `SensorReport`s.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from seaswarm.config import DefaultConfig, FleetConfig, Scenario, VehicleTypeConfig
from seaswarm.rng import ScenarioRng
from seaswarm.search.pod import SensorModel
from seaswarm.vehicles.backend import (
    Command,
    FollowPath,
    GoTo,
    Hold,
    ReturnToBase,
    SensorReport,
    VehicleState,
    VehicleStatus,
    VehicleType,
)
from seaswarm.vehicles.energy import drain_rate_per_s
from seaswarm.vehicles.kinematics import KinematicLimits, Pose, heading_to, step_unicycle
from seaswarm.vehicles.sensors import roll_detection
from seaswarm.world.ocean import CurrentField, make_current
from seaswarm.world.target import PersonInWater


@dataclass(frozen=True)
class VehicleSpec:
    """Identity, type, sensor and altitude layer of one vehicle of the fleet."""

    vehicle_id: str
    vehicle_type: VehicleType
    sensor: str
    altitude_m: float


def fleet_specs(fleet: FleetConfig, defaults: DefaultConfig) -> list[VehicleSpec]:
    """Vehicles of a scenario fleet: drone_1.. (EO first, then thermal), usv_1..

    Drone i flies at altitude layer i mod n_layers, so drones launched together from the base
    are vertically separated.
    """
    drone, usv = defaults.vehicles.drone, defaults.vehicles.usv
    sensors = [drone.sensor] * fleet.drones_eo + ["drone_thermal"] * fleet.drones_thermal
    layers = drone.altitude_layers_m
    specs = [
        VehicleSpec(f"drone_{i + 1}", VehicleType.DRONE, s, layers[i % len(layers)])
        for i, s in enumerate(sensors)
    ]
    specs += [
        VehicleSpec(f"usv_{i + 1}", VehicleType.USV, usv.sensor, usv.altitude_layers_m[0])
        for i in range(fleet.usvs)
    ]
    return specs


@dataclass
class _Vehicle:
    spec: VehicleSpec
    config: VehicleTypeConfig
    limits: KinematicLimits
    sensor: SensorModel
    pose: Pose
    energy_frac: float = 1.0
    status: VehicleStatus = VehicleStatus.IDLE
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    speed_cmd_mps: float = 0.0
    turnaround_left_s: float = 0.0
    next_look_s: float = 0.0

    @property
    def altitude_m(self) -> float:
        return self.spec.altitude_m

    def parked(self, base_xy_m: tuple[float, float]) -> bool:
        """Idle at the base: on deck or moored, using no energy and not searching."""
        distance_m = np.hypot(self.pose.x_m - base_xy_m[0], self.pose.y_m - base_xy_m[1])
        return self.status is VehicleStatus.IDLE and distance_m < self.config.waypoint_radius_m

    def state(self) -> VehicleState:
        return VehicleState(
            self.spec.vehicle_id,
            self.spec.vehicle_type,
            self.pose.x_m,
            self.pose.y_m,
            self.pose.yaw_rad,
            self.pose.speed_mps,
            self.altitude_m,
            self.energy_frac,
            self.status,
            len(self.waypoints),
        )


class Sim2DBackend:
    """Kinematic fleet over a drifting person; implements `VehicleBackend`."""

    def __init__(
        self,
        specs: list[VehicleSpec],
        defaults: DefaultConfig,
        person: PersonInWater,
        current: CurrentField,
        base_xy_m: tuple[float, float],
        daylight: bool,
        sensor_rng: np.random.Generator,
    ) -> None:
        """Place every vehicle at the base, idle and fully charged, at the person's time."""
        self.t_s = person.t_s
        self.base_xy_m = base_xy_m
        self._person = person
        self._current = current
        self._rng = sensor_rng
        self._recovered = False
        self._vehicles: dict[str, _Vehicle] = {}
        for spec in specs:
            cfg = getattr(defaults.vehicles, spec.vehicle_type.value)
            sensor = SensorModel.from_config(
                defaults.sensors[spec.sensor], daylight, spec.altitude_m
            )
            pose = Pose(base_xy_m[0], base_xy_m[1], np.pi / 2, 0.0)
            self._vehicles[spec.vehicle_id] = _Vehicle(
                spec,
                cfg,
                KinematicLimits.from_config(cfg),
                sensor,
                pose,
                next_look_s=self.t_s + sensor.period_s,
            )

    @classmethod
    def from_scenario(
        cls, scenario: Scenario, defaults: DefaultConfig, rngs: ScenarioRng
    ) -> Sim2DBackend:
        """Fleet of a scenario at its base; the person has drifted until the search starts."""
        person = PersonInWater.from_scenario(scenario, defaults, rngs)
        dt_s = defaults.sim.dt_s
        for _ in range(round(scenario.lkp.time_since_fall_s / dt_s)):
            person.step(dt_s)
        return cls(
            fleet_specs(scenario.fleet, defaults),
            defaults,
            person,
            make_current(scenario.current),
            (scenario.base.x_m, scenario.base.y_m),
            scenario.daylight,
            rngs.sensors,
        )

    # ------------------------------------------------------------------ VehicleBackend

    def vehicle_ids(self) -> list[str]:
        """Ids of every vehicle, sorted."""
        return sorted(self._vehicles)

    def get_state(self, vehicle_id: str) -> VehicleState:
        """Current state of one vehicle."""
        return self._vehicles[vehicle_id].state()

    def send(self, command: Command) -> None:
        """Execute a command (already accepted by the safety layer)."""
        v = self._vehicles[command.vehicle_id]
        if v.status in (VehicleStatus.TURNAROUND, VehicleStatus.STRANDED):
            return
        match command:
            case GoTo():
                v.waypoints = [(command.x_m, command.y_m)]
                v.speed_cmd_mps, v.status = command.speed_mps, VehicleStatus.TRANSIT
            case FollowPath():
                v.waypoints = list(command.waypoints_m)
                v.speed_cmd_mps, v.status = command.speed_mps, VehicleStatus.TRANSIT
            case ReturnToBase():
                v.waypoints = [self.base_xy_m]
                v.speed_cmd_mps, v.status = v.config.cruise_speed_mps, VehicleStatus.RTB
            case Hold():
                v.waypoints, v.speed_cmd_mps, v.status = [], 0.0, VehicleStatus.IDLE

    def step(self, dt_s: float) -> list[SensorReport]:
        """Advance the person and every vehicle by dt_s; return the looks taken."""
        self._person.step(dt_s)
        self.t_s += dt_s
        reports = []
        for vid in self.vehicle_ids():
            v = self._vehicles[vid]
            self._move(v, dt_s)
            if v.status not in (VehicleStatus.TURNAROUND, VehicleStatus.STRANDED) and not v.parked(
                self.base_xy_m
            ):
                reports.extend(self._look(v))
                if v.spec.vehicle_type is VehicleType.USV:
                    self._check_recovery(v)
        return reports

    def person_recovered(self) -> bool:
        """True once a USV has come within its recovery radius of the person."""
        return self._recovered

    # ------------------------------------------------------------------ truth access (logs)

    def truth_position_m(self) -> np.ndarray:
        """True position of the person. For logs and replays only, never for planning."""
        return self._person.position_m

    # ------------------------------------------------------------------ internals

    def _move(self, v: _Vehicle, dt_s: float) -> None:
        if v.status is VehicleStatus.STRANDED:
            return
        if v.status is VehicleStatus.TURNAROUND:
            v.turnaround_left_s -= dt_s
            if v.turnaround_left_s <= 0.0:
                v.energy_frac, v.status = 1.0, VehicleStatus.IDLE
            return

        while (
            v.waypoints
            and np.hypot(v.waypoints[0][0] - v.pose.x_m, v.waypoints[0][1] - v.pose.y_m)
            < v.config.waypoint_radius_m
        ):
            v.waypoints.pop(0)
        if not v.waypoints and v.status is VehicleStatus.RTB:
            self._start_turnaround(v)
            return
        if not v.waypoints:
            v.status, v.speed_cmd_mps = VehicleStatus.IDLE, 0.0

        yaw_cmd = heading_to(v.pose, *v.waypoints[0]) if v.waypoints else v.pose.yaw_rad
        drift = (0.0, 0.0)
        if v.spec.vehicle_type is VehicleType.USV:
            u, w = self._current.velocity(np.array([[v.pose.x_m, v.pose.y_m]]), self.t_s)[0]
            drift = (float(u), float(w))
        if v.parked(self.base_xy_m):
            return
        v.pose = step_unicycle(v.pose, yaw_cmd, v.speed_cmd_mps, v.limits, dt_s, drift)
        v.energy_frac -= drain_rate_per_s(v.pose.speed_mps, v.config) * dt_s
        if v.energy_frac <= 0.0:
            v.energy_frac, v.status, v.waypoints = 0.0, VehicleStatus.STRANDED, []
            v.pose = Pose(v.pose.x_m, v.pose.y_m, v.pose.yaw_rad, 0.0)

    def _start_turnaround(self, v: _Vehicle) -> None:
        v.pose = Pose(self.base_xy_m[0], self.base_xy_m[1], v.pose.yaw_rad, 0.0)
        v.status, v.speed_cmd_mps = VehicleStatus.TURNAROUND, 0.0
        v.turnaround_left_s = v.config.turnaround_s

    def _look(self, v: _Vehicle) -> list[SensorReport]:
        if self.t_s < v.next_look_s:
            return []
        while v.next_look_s <= self.t_s:
            v.next_look_s += v.sensor.period_s
        xy = np.array([v.pose.x_m, v.pose.y_m])
        detected = roll_detection(v.sensor, xy, self._person.position_m, self._rng)
        measured = None
        if detected:
            noise = v.config.detection_sigma_m * self._rng.standard_normal(2)
            mx, my = self._person.position_m + noise
            measured = (float(mx), float(my))
        return [
            SensorReport(
                v.spec.vehicle_id,
                v.spec.sensor,
                self.t_s,
                v.pose.x_m,
                v.pose.y_m,
                v.altitude_m,
                detected,
                measured,
            )
        ]

    def _check_recovery(self, v: _Vehicle) -> None:
        distance_m = np.hypot(*(self._person.position_m - [v.pose.x_m, v.pose.y_m]))
        if distance_m <= v.config.recovery_radius_m:
            self._recovered = True
