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
from seaswarm.planning.auction import AuctionPlanner
from seaswarm.planning.base import (
    Assignment,
    Detection,
    MissionView,
    OperatorPrefs,
    Planner,
    PriorityArea,
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


# Triggers that may fire often (a vehicle finishing a pattern) are rate-limited; all the others
# replan at the next step.
RATE_LIMITED = frozenset({ReplanTrigger.VEHICLE_IDLE})


class MissionState(StrEnum):
    """Overall state of the search."""

    SEARCHING = "SEARCHING"
    FOUND = "FOUND"
    RECOVERED = "RECOVERED"
    TIMEOUT = "TIMEOUT"


STRATEGIES = ("lawnmower", "greedy", "auction")


def make_planner(
    name: str, tiles: TileGrid, defaults: DefaultConfig, tiebreak: np.random.Generator
) -> Planner:
    """Planner by strategy name."""
    match name:
        case "greedy":
            return GreedyPlanner(tiles)
        case "lawnmower":
            return LawnmowerPlanner(defaults.planner.lawnmower_prior_mass)
        case "auction":
            return AuctionPlanner(tiles, defaults.planner, tiebreak)
    raise ValueError(f"unknown strategy {name!r}; choose one of {', '.join(STRATEGIES)}")


@dataclass
class MissionEvent:
    """Something worth logging: replan, refusal, detection, forced RTB, state change."""

    t_s: float
    kind: str
    data: dict = field(default_factory=dict)


class SearchMission:
    """One search scenario run with one strategy."""

    def __init__(
        self, scenario: Scenario, defaults: DefaultConfig, strategy: str, search_only: bool = False
    ) -> None:
        """Set up world, fleet, safety, belief and planner at the start of the search.

        Args:
            scenario: Scenario to run.
            defaults: Project defaults.
            strategy: Planner name ("lawnmower" or "greedy").
            search_only: Evaluation mode: detections are recorded but do not end the search or
                change the belief, so search effectiveness (POS_cum) can be compared over a fixed
                time on every run.
        """
        self.search_only = search_only
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
        self.strategy = strategy
        self._tiebreak = rngs.planner_tiebreak
        self.planner = make_planner(strategy, self.tiles, defaults, self._tiebreak)
        # vehicles under a direct operator command, left alone by the planner and the automatic
        # recovery until that command ends (vehicle -> what it was told to do)
        self.operator_tasks: dict[str, str] = {}
        self.rescuer_id: str | None = None  # USV chosen by the operator for the recovery
        self.prefs = OperatorPrefs()

        self.sensors: dict[str, SensorModel] = {}
        self.capabilities: dict[str, VehicleCapability] = {}
        for spec in fleet_specs(scenario.fleet, defaults):
            cfg = getattr(defaults.vehicles, spec.vehicle_type.value)
            sensor_cfg = defaults.sensors[spec.sensor]
            self.sensors[spec.vehicle_id] = SensorModel.from_config(
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
        self._refused: dict[tuple[str, str | None], float] = {}  # (vehicle, tile) -> refusal time
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
        for vid in self.guard.held:
            self._event("conflict_hold", vehicle_id=vid)
            self._pending.add(ReplanTrigger.CONFLICT)
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
            if not self.search_only:
                return
            # search_only: every look counts as unsuccessful, so the belief and POS_cum describe
            # the planned search alone, independent of where the person really is
        look = footprint(
            self.belief.grid,
            self.sensors[report.vehicle_id],
            report.x_m,
            report.y_m,
            self.defaults.belief.pod_points_per_r_max,
        )
        if look.pod.size:
            self.success.add(update_no_detection(self.belief, look))

    def _on_detection(self, report: SensorReport) -> None:
        if self.search_only:
            if self.time_to_detection_s is None:
                self.time_to_detection_s = self.t_s - self.start_s
                self._event("detection", vehicle_id=report.vehicle_id, search_only=True)
            return
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
            self._excluded(),
        )

    def states(self) -> list[VehicleState]:
        """Current state of every vehicle, sorted by id."""
        return [self.guard.get_state(vid) for vid in self.guard.vehicle_ids()]

    def _watch_status(self) -> None:
        for state in self.states():
            vid = state.vehicle_id
            before = self._last_status[vid]
            if state.status is VehicleStatus.IDLE and before is VehicleStatus.TURNAROUND:
                self._pending.add(ReplanTrigger.RETURNED)
            elif state.status is VehicleStatus.IDLE and before is VehicleStatus.TRANSIT:
                self._pending.add(ReplanTrigger.VEHICLE_IDLE)
            ended = state.status in (VehicleStatus.TURNAROUND, VehicleStatus.STRANDED) or (
                state.status is VehicleStatus.IDLE and before is VehicleStatus.TRANSIT
            )
            if vid in self.operator_tasks and ended:
                self._event("operator_task_done", vehicle_id=vid, task=self.operator_tasks.pop(vid))
            self._last_status[vid] = state.status

    def set_strategy(self, name: str) -> None:
        """Switch the planning strategy; replans at the next step."""
        self.planner = make_planner(name, self.tiles, self.defaults, self._tiebreak)
        self.strategy = name
        self._event("strategy", name=name)
        self._pending.add(ReplanTrigger.OPERATOR)

    def operator_command(self, vehicle_id: str, label: str) -> None:
        """Mark a vehicle as under a direct operator command (planner leaves it alone)."""
        self.operator_tasks[vehicle_id] = label
        self._event("operator_command", vehicle_id=vehicle_id, task=label)
        self._pending.add(ReplanTrigger.OPERATOR)

    def set_priority_area(self, area: PriorityArea) -> None:
        """Add an operator priority area (changes planner utility, never the belief); replans."""
        active = tuple(a for a in self.prefs.areas if a.expires_s > self.t_s)
        self.prefs = OperatorPrefs((*active, area))
        self._event("priority", area_id=area.area_id, weight=area.weight)
        self._pending.add(ReplanTrigger.PRIORITY)

    def clear_priority_areas(self) -> int:
        """Remove every operator priority area; replans. Returns how many were removed."""
        count = len(self.prefs.areas)
        self.prefs = OperatorPrefs()
        self._pending.add(ReplanTrigger.PRIORITY)
        return count

    def _maybe_replan(self) -> None:
        if self.t_s - self._last_replan_s >= self.defaults.planner.replan_period_s:
            self._pending.add(ReplanTrigger.PERIODIC)
        if not self._pending:
            return
        cooldown = self.defaults.planner.event_replan_cooldown_s
        if self._pending <= RATE_LIMITED and self.t_s - self._last_event_replan_s < cooldown:
            return
        self.replan(sorted(self._pending))

    def _excluded(self) -> dict[str, frozenset[str | None]]:
        """Refused (vehicle, tile) choices of the last replan period, not to be proposed again."""
        horizon = self.defaults.planner.replan_period_s
        self._refused = {k: t for k, t in self._refused.items() if self.t_s - t < horizon}
        excluded: dict[str, set[str | None]] = {}
        for vid, tile in self._refused:
            excluded.setdefault(vid, set()).add(tile)
        return {vid: frozenset(tiles) for vid, tiles in excluded.items()}

    def replan(self, triggers: list[ReplanTrigger]) -> None:
        """Ask the planner for tasks and submit the changed ones through the safety guard."""
        self._pending.clear()
        self._last_event_replan_s = self.t_s
        if ReplanTrigger.PERIODIC in triggers or ReplanTrigger.START in triggers:
            self._last_replan_s = self.t_s
        states = [s for s in self.states() if s.vehicle_id not in self.operator_tasks]
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
            self._refused[(state.vehicle_id, task.tile_id)] = self.t_s
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
        if self._last_recovery_aim_s == -np.inf:
            self._event("replan", triggers=[ReplanTrigger.DETECTION.value], assigned=[])
        self._last_recovery_aim_s = self.t_s
        target = self.belief.mean_m()
        watcher = self.detections[-1].vehicle_id
        usvs = [
            s
            for s in self.states()
            if s.vehicle_type is VehicleType.USV
            and s.status in (VehicleStatus.IDLE, VehicleStatus.TRANSIT)
        ]
        chosen = [s for s in usvs if s.vehicle_id == self.rescuer_id]
        rescuer = (
            chosen[0]
            if chosen
            else min(
                usvs, key=lambda s: np.hypot(s.x_m - target[0], s.y_m - target[1]), default=None
            )
        )
        for state in self.states():
            vid = state.vehicle_id
            cfg = self.capabilities[vid].config
            if state.status not in (VehicleStatus.IDLE, VehicleStatus.TRANSIT):
                continue  # returning, in turnaround or stranded: leave it alone
            if vid in self.operator_tasks and vid != self.rescuer_id:
                continue  # the operator's command stands
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
        recovered = self.backend.person_recovered() and not self.search_only
        if recovered and self.state is not MissionState.RECOVERED:
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
