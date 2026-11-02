"""SearchMission: the coordinator that owns the world, the fleet, the belief and the planner.

Each step of `dt_s` simulated seconds:
1. the backend moves the truth person and the vehicles and returns the sensor looks;
2. every look updates the belief (Bayes "searched here, found nothing", or a detection);
3. every `predict_period_s` the belief drifts and spreads (predict step);
4. the safety guard watches the fleet (forced RTB, violation counts);
5. replan triggers are collected and, if any fired, the planner proposes new tasks, which go
   through the safety guard.

Simulated time only advances when `step` or `run` is called; there is no wall-clock coupling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from seaswarm.config import DefaultConfig, Scenario
from seaswarm.planning.base import (
    Assignment,
    Detection,
    MissionView,
    OperatorPrefs,
    Planner,
    ReplanTrigger,
    VehicleCapability,
)
from seaswarm.planning.greedy import GreedyPlanner
from seaswarm.planning.lawnmower import LawnmowerPlanner
from seaswarm.planning.tiles import TileGrid
from seaswarm.rng import make_rngs
from seaswarm.safety.guard import Proposal, SafetyGuard
from seaswarm.safety.refusal import Refusal
from seaswarm.safety.rules import Geofence
from seaswarm.search.bayes import SuccessTracker, update_detection, update_no_detection
from seaswarm.search.motion import BeliefMotion
from seaswarm.search.pod import SensorModel
from seaswarm.search.predict import belief_at, predict
from seaswarm.vehicles.backend import (
    FollowPath,
    GoTo,
    ReturnToBase,
    SensorReport,
    VehicleState,
    VehicleStatus,
    VehicleType,
)
from seaswarm.vehicles.sensors import footprint
from seaswarm.vehicles.sim2d import Sim2DBackend, fleet_specs

logger = logging.getLogger(__name__)


class MissionState(StrEnum):
    """Overall state of the search."""

    SEARCHING = "SEARCHING"
    FOUND = "FOUND"
    RECOVERED = "RECOVERED"
    TIMEOUT = "TIMEOUT"


def make_planner(name: str, tiles: TileGrid, defaults: DefaultConfig) -> Planner:
    """Planner by strategy name."""
    match name:
        case "greedy":
            return GreedyPlanner(tiles)
        case "lawnmower":
            return LawnmowerPlanner(defaults.planner.lawnmower_prior_mass)
    raise ValueError(f"unknown strategy {name!r}; choose lawnmower or greedy")


@dataclass
class MissionEvent:
    """Something worth logging: replan, refusal, detection, forced RTB, state change."""

    t_s: float
    kind: str
    data: dict = field(default_factory=dict)


class SearchMission:
    """One search scenario run with one strategy."""

    def __init__(self, scenario: Scenario, defaults: DefaultConfig, strategy: str) -> None:
        """Set up world, fleet, safety, belief and planner at the start of the search."""
        self.scenario = scenario
        self.defaults = defaults
        self.dt_s = defaults.sim.dt_s
        rngs = make_rngs(scenario.seed)
        self.backend = Sim2DBackend.from_scenario(scenario, defaults, rngs)
        self.base_xy_m = (scenario.base.x_m, scenario.base.y_m)
        fence = Geofence.around(scenario.area, scenario.base, defaults.safety.geofence_margin_m)
        self.guard = SafetyGuard(self.backend, fence, self.base_xy_m, defaults)

        self.t_s = self.backend.t_s
        self.start_s = self.t_s
        self.belief = belief_at(scenario, defaults, self.t_s)
        self.motion = BeliefMotion.from_scenario(scenario, defaults)
        self.success = SuccessTracker()
        self.tiles = TileGrid.from_grid(self.belief.grid, defaults.planner.tile_size_m)
        self.planner = make_planner(strategy, self.tiles, defaults)
        self.prefs = OperatorPrefs()

        self._sensors: dict[str, SensorModel] = {}
        self.capabilities: dict[str, VehicleCapability] = {}
        for spec in fleet_specs(scenario.fleet, defaults):
            cfg = getattr(defaults.vehicles, spec.vehicle_type.value)
            sensor_cfg = defaults.sensors[spec.sensor]
            self._sensors[spec.vehicle_id] = SensorModel.from_config(
                sensor_cfg, scenario.daylight, spec.altitude_m
            )
            self.capabilities[spec.vehicle_id] = VehicleCapability.build(
                spec.vehicle_id,
                spec.vehicle_type,
                cfg,
                sensor_cfg,
                scenario.daylight,
                spec.altitude_m,
                defaults.planner,
            )

        self.state = MissionState.SEARCHING
        self.steps = 0
        self.detections: list[Detection] = []
        self.refusals: list[Refusal] = []
        self.events: list[MissionEvent] = []
        self.time_to_detection_s: float | None = None
        self.time_to_recovery_s: float | None = None
        self._last_predict_s = self.t_s
        self._last_replan_s = -np.inf
        self._last_event_replan_s = -np.inf
        self._last_recovery_aim_s = -np.inf
        self._pending: set[ReplanTrigger] = {ReplanTrigger.START}
        self._last_status = {vid: VehicleStatus.IDLE for vid in self.backend.vehicle_ids()}

    # ------------------------------------------------------------------ main loop

    @property
    def done(self) -> bool:
        """True once the mission has ended (recovered, timed out, or found with no USV)."""
        if self.state in (MissionState.RECOVERED, MissionState.TIMEOUT):
            return True
        return self.state is MissionState.FOUND and self.scenario.fleet.usvs == 0

    def run(self) -> MissionState:
        """Step until the mission ends; bounded by the scenario's time and step limits."""
        while not self.done:
            self.step()
        return self.state

    def step(self) -> None:
        """Advance the whole mission by one time step."""
        reports = self.backend.step(self.dt_s)
        self.t_s += self.dt_s
        self.steps += 1
        for report in reports:
            self._apply_look(report)
        if self.t_s - self._last_predict_s >= self.defaults.belief.predict_period_s - 1e-9:
            elapsed = self.t_s - self._last_predict_s
            predict(self.belief, self.motion, self._last_predict_s, elapsed)
            self._last_predict_s = self.t_s

        for vid in self.guard.check_state(self.t_s):
            self._event("forced_rtb", vehicle_id=vid)
            self._pending.add(ReplanTrigger.LOW_ENERGY)
        self._watch_status()

        if self.state is MissionState.SEARCHING:
            self._maybe_replan()
        elif self.state is MissionState.FOUND:
            self._aim_recovery()
        self._check_end()

    # ------------------------------------------------------------------ belief

    def _apply_look(self, report: SensorReport) -> None:
        if report.detected:
            self._on_detection(report)
            return
        look = footprint(
            self.belief.grid,
            self._sensors[report.vehicle_id],
            report.x_m,
            report.y_m,
            self.defaults.belief.pod_points_per_r_max,
        )
        if look.pod.size:
            self.success.add(update_no_detection(self.belief, look))

    def _on_detection(self, report: SensorReport) -> None:
        x, y = report.measured_xy_m
        detection = Detection(f"d_{len(self.detections) + 1}", report.t_s, report.vehicle_id, x, y)
        self.detections.append(detection)
        sigma = getattr(
            self.defaults.vehicles, self.capabilities[report.vehicle_id].vehicle_type.value
        )
        update_detection(self.belief, (x, y), sigma.detection_sigma_m)
        self._event(
            "detection",
            detection_id=detection.detection_id,
            vehicle_id=report.vehicle_id,
            x_m=x,
            y_m=y,
        )
        if self.state is MissionState.SEARCHING:
            self.state = MissionState.FOUND
            self.time_to_detection_s = self.t_s - self.start_s
            self._event("state", state=self.state.value)
            self._last_recovery_aim_s = -np.inf

    # ------------------------------------------------------------------ planning

    def view(self) -> MissionView:
        """What planners may see: no truth."""
        drift = self.motion.mean_velocity_mps(self.t_s, self.belief.mean_m())
        return MissionView(
            self.t_s,
            self.start_s,
            self.base_xy_m,
            self.guard.fence,
            self.capabilities,
            {vid: self.guard.tile_of(vid) for vid in self.backend.vehicle_ids()},
            (float(drift[0]), float(drift[1])),
            tuple(self.detections),
        )

    def states(self) -> list[VehicleState]:
        """Current state of every vehicle, sorted by id."""
        return [self.guard.get_state(vid) for vid in self.guard.vehicle_ids()]

    def _watch_status(self) -> None:
        for state in self.states():
            before = self._last_status[state.vehicle_id]
            if state.status is VehicleStatus.IDLE and before is VehicleStatus.TURNAROUND:
                self._pending.add(ReplanTrigger.RETURNED)
            elif state.status is VehicleStatus.IDLE and before is VehicleStatus.TRANSIT:
                self._pending.add(ReplanTrigger.VEHICLE_IDLE)
            self._last_status[state.vehicle_id] = state.status

    def _maybe_replan(self) -> None:
        if self.t_s - self._last_replan_s >= self.defaults.planner.replan_period_s:
            self._pending.add(ReplanTrigger.PERIODIC)
        if not self._pending:
            return
        event_only = (
            ReplanTrigger.PERIODIC not in self._pending and ReplanTrigger.START not in self._pending
        )
        cooldown = self.defaults.planner.event_replan_cooldown_s
        if event_only and self.t_s - self._last_event_replan_s < cooldown:
            return
        self.replan(sorted(self._pending))

    def replan(self, triggers: list[ReplanTrigger]) -> None:
        """Ask the planner for tasks and submit the changed ones through the safety guard."""
        self._pending.clear()
        self._last_event_replan_s = self.t_s
        if ReplanTrigger.PERIODIC in triggers or ReplanTrigger.START in triggers:
            self._last_replan_s = self.t_s
        states = self.states()
        plan = self.planner.plan(self.view(), self.belief, states, self.prefs)
        self._event("replan", triggers=[t.value for t in triggers], assigned=sorted(plan))
        by_id = {s.vehicle_id: s for s in states}
        for vid, assignment in sorted(plan.items()):
            self._submit_assignment(by_id[vid], assignment)

    def _submit_assignment(self, state: VehicleState, assignment: Assignment) -> None:
        if not assignment.tasks:
            return
        task = assignment.tasks[0]
        same_tile = task.tile_id is not None and task.tile_id == self.guard.tile_of(
            state.vehicle_id
        )
        if same_tile and state.status is VehicleStatus.TRANSIT:
            return  # already searching that tile: let it finish its pattern
        speed = self.capabilities[state.vehicle_id].cruise_speed_mps
        command = FollowPath(state.vehicle_id, task.waypoints, speed)
        refusal = self.guard.submit(Proposal(command, tile_id=task.tile_id), self.t_s)
        if refusal is not None:
            self.refusals.append(refusal)
            self._event(
                "refusal",
                vehicle_id=state.vehicle_id,
                code=refusal.code.value,
                reason=refusal.reason,
            )
            self._pending.add(ReplanTrigger.REFUSAL)

    # ------------------------------------------------------------------ after a detection

    def _aim_recovery(self) -> None:
        """Keep the detecting drone over the belief mean, send the nearest USV, others home."""
        if self.t_s - self._last_recovery_aim_s < self.defaults.planner.recovery_update_s:
            return
        self._last_recovery_aim_s = self.t_s
        target = self.belief.mean_m()
        watcher = self.detections[-1].vehicle_id
        usvs = [
            s
            for s in self.states()
            if s.vehicle_type is VehicleType.USV
            and s.status in (VehicleStatus.IDLE, VehicleStatus.TRANSIT)
        ]
        rescuer = min(
            usvs, key=lambda s: np.hypot(s.x_m - target[0], s.y_m - target[1]), default=None
        )
        for state in self.states():
            vid = state.vehicle_id
            cfg = self.capabilities[vid].config
            if state.status not in (VehicleStatus.IDLE, VehicleStatus.TRANSIT):
                continue  # returning, in turnaround or stranded: leave it alone
            if rescuer is not None and vid == rescuer.vehicle_id:
                command = GoTo(vid, float(target[0]), float(target[1]), cfg.max_speed_mps)
            elif vid == watcher and state.vehicle_type is VehicleType.DRONE:
                command = GoTo(vid, float(target[0]), float(target[1]), cfg.cruise_speed_mps)
            elif not self._parked(state):
                command = ReturnToBase(vid)
            else:
                continue
            refusal = self.guard.submit(Proposal(command, dispatch=True), self.t_s)
            if refusal is not None:
                self.refusals.append(refusal)
                self._event(
                    "refusal", vehicle_id=vid, code=refusal.code.value, reason=refusal.reason
                )

    def _parked(self, state: VehicleState) -> bool:
        radius = self.capabilities[state.vehicle_id].config.waypoint_radius_m
        return bool(np.hypot(state.x_m - self.base_xy_m[0], state.y_m - self.base_xy_m[1]) < radius)

    # ------------------------------------------------------------------ end conditions

    def _check_end(self) -> None:
        if self.backend.person_recovered() and self.state is not MissionState.RECOVERED:
            self.state = MissionState.RECOVERED
            self.time_to_recovery_s = self.t_s - self.start_s
            self._event("state", state=self.state.value)
            return
        limits = self.scenario.limits
        if not self.done and (
            self.t_s - self.start_s >= limits.max_sim_time_s or self.steps >= limits.max_steps
        ):
            self.state = MissionState.TIMEOUT
            self._event("state", state=self.state.value)

    def _event(self, kind: str, **data) -> None:
        self.events.append(MissionEvent(self.t_s, kind, data))
