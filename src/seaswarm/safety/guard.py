"""SafetyGuard: the only path from planners, scripts or the agent to a vehicle backend.

Adapted from skypilot-agent's SafeDrone. Every proposed command is checked against the rules and
logged; refused commands never reach the backend. The guard also watches the fleet every step:
it forces a return to base before an energy reserve is broken, and counts actual violations
(separation, geofence, stranded vehicles), whose target is zero.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

import numpy as np

from seaswarm.config import DefaultConfig, VehicleTypeConfig
from seaswarm.safety import rules
from seaswarm.safety.refusal import Refusal, RefusalCode
from seaswarm.safety.rules import XY, Geofence, Plan
from seaswarm.vehicles.backend import (
    Command,
    FollowPath,
    GoTo,
    Hold,
    ReturnToBase,
    VehicleBackend,
    VehicleState,
    VehicleStatus,
)
from seaswarm.vehicles.energy import drain_rate_per_s, rtb_energy_frac

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Proposal:
    """A command proposed for execution, with the search tile it serves (if any)."""

    command: Command
    tile_id: str | None = None
    dispatch: bool = False  # dispatch to a detection: exempt from the one-searcher-per-tile rule


@dataclass(frozen=True)
class DecisionRecord:
    """One logged safety decision."""

    t_s: float
    vehicle_id: str
    command: str
    accepted: bool
    code: str | None
    reason: str


@dataclass
class Violations:
    """Safety violations observed in the fleet's actual states (episodes, not steps)."""

    separation: int = 0
    geofence: int = 0
    stranded: int = 0

    @property
    def total(self) -> int:
        """All violations."""
        return self.separation + self.geofence + self.stranded


class SafetyGuard:
    """Checks proposals against the safety rules and forwards accepted ones to the backend."""

    def __init__(
        self, backend: VehicleBackend, fence: Geofence, base_xy_m: XY, defaults: DefaultConfig
    ) -> None:
        """Guard `backend` with the limits of `defaults.safety`."""
        self._backend = backend
        self.fence = fence
        self.base_xy_m = base_xy_m
        self._defaults = defaults
        self.limits = defaults.safety
        self._planned: dict[str, tuple[tuple[XY, ...], float]] = {}
        self._tile_owner: dict[str, str] = {}
        self.decisions: list[DecisionRecord] = []
        self.violations = Violations()
        self._in_violation: set[tuple[str, ...]] = set()
        self._commanded_at: dict[str, float] = {}
        self.held: list[str] = []  # vehicles stopped by the last check_state to avoid a conflict

    # ------------------------------------------------------------------ queries

    def vehicle_ids(self) -> list[str]:
        """Ids of every vehicle, sorted."""
        return self._backend.vehicle_ids()

    def get_state(self, vehicle_id: str) -> VehicleState:
        """Current state of one vehicle."""
        return self._backend.get_state(vehicle_id)

    def tile_of(self, vehicle_id: str) -> str | None:
        """Tile currently searched by a vehicle, if any."""
        return next((t for t, v in self._tile_owner.items() if v == vehicle_id), None)

    def _config(self, state: VehicleState) -> VehicleTypeConfig:
        return getattr(self._defaults.vehicles, state.vehicle_type.value)

    # ------------------------------------------------------------------ proposals

    def submit(self, proposal: Proposal, t_s: float) -> Refusal | None:
        """Check a proposal; execute it if accepted. Returns the refusal, or None if accepted."""
        command = proposal.command
        vid = command.vehicle_id
        if vid not in self.vehicle_ids():
            refusal = Refusal(
                RefusalCode.VEHICLE_UNAVAILABLE,
                f"there is no vehicle {vid!r}; known vehicles: {', '.join(self.vehicle_ids())}",
                vid,
            )
            return self._record(t_s, command, refusal)
        state = self.get_state(vid)
        match command:
            case ReturnToBase() | Hold():
                return self._accept(t_s, proposal, state)  # recalls are never refused
            case GoTo():
                waypoints, speed = ((command.x_m, command.y_m),), command.speed_mps
            case FollowPath():
                waypoints, speed = command.waypoints_m, command.speed_mps
        cfg = self._config(state)
        refusal = rules.first_refusal(
            [
                lambda: rules.check_available(state),
                lambda: rules.check_geofence(vid, waypoints, self.fence),
                lambda: rules.check_energy(state, waypoints, speed, self.base_xy_m, cfg),
                lambda: rules.check_tile(
                    vid, proposal.tile_id, self._tile_owner, proposal.dispatch
                ),
                lambda: rules.check_separation(
                    Plan(state, waypoints, speed), self._others(vid), self.base_xy_m, self.limits
                ),
            ]
        )
        if refusal is not None:
            return self._record(t_s, command, refusal)
        return self._accept(t_s, proposal, state)

    def _plan_of(self, state: VehicleState) -> Plan:
        """Remaining planned waypoints of a vehicle (stationary if it has none)."""
        waypoints, speed = self._planned.get(state.vehicle_id, ((), 0.0))
        left = state.waypoints_left
        remaining = waypoints[len(waypoints) - left :] if left else ()
        return Plan(state, remaining, speed if remaining else 0.0)

    def _others(self, vehicle_id: str) -> list[Plan]:
        """Plans of every other vehicle."""
        return [self._plan_of(self.get_state(v)) for v in self.vehicle_ids() if v != vehicle_id]

    def _accept(self, t_s: float, proposal: Proposal, state: VehicleState) -> None:
        command, vid = proposal.command, state.vehicle_id
        self._commanded_at[vid] = t_s
        match command:
            case GoTo():
                self._planned[vid] = (((command.x_m, command.y_m),), command.speed_mps)
            case FollowPath():
                self._planned[vid] = (tuple(command.waypoints_m), command.speed_mps)
            case ReturnToBase():
                self._planned[vid] = ((self.base_xy_m,), self._config(state).cruise_speed_mps)
            case Hold():
                self._planned[vid] = ((), 0.0)
        self._release_tile(vid)
        if proposal.tile_id is not None and isinstance(command, GoTo | FollowPath):
            self._tile_owner[proposal.tile_id] = vid
        self._backend.send(command)
        self._record(t_s, command, None)
        return None

    def _release_tile(self, vehicle_id: str) -> None:
        for tile in [t for t, v in self._tile_owner.items() if v == vehicle_id]:
            del self._tile_owner[tile]

    def _record(self, t_s: float, command: Command, refusal: Refusal | None) -> Refusal | None:
        name = type(command).__name__
        self.decisions.append(
            DecisionRecord(
                t_s,
                command.vehicle_id,
                name,
                refusal is None,
                None if refusal is None else refusal.code.value,
                "" if refusal is None else refusal.reason,
            )
        )
        if refusal is not None:
            logger.info("safety: refused %s for %s: %s", name, command.vehicle_id, refusal.reason)
        return refusal

    # ------------------------------------------------------------------ per-step watch

    def check_state(self, t_s: float) -> list[str]:
        """Watch the fleet after a step: force RTB where needed, count violations.

        Returns:
            Ids of vehicles sent back to base by this check (a replan trigger).
        """
        forced = []
        states = [self.get_state(vid) for vid in self.vehicle_ids()]
        for state in states:
            if state.status is not VehicleStatus.TRANSIT:
                self._release_tile(state.vehicle_id)  # a tile is held only while searching it
            if state.status is VehicleStatus.STRANDED:
                self._count(("stranded", state.vehicle_id), True, "stranded")
                continue
            self._count(
                ("geofence", state.vehicle_id),
                not self.fence.contains(state.x_m, state.y_m),
                "geofence",
            )
            if self._reserve_at_risk(state):
                self.submit(Proposal(ReturnToBase(state.vehicle_id)), t_s)
                forced.append(state.vehicle_id)
        self.held = self._resolve_conflicts(t_s)
        for a, b in itertools.combinations(states, 2):
            self._count(
                ("separation", a.vehicle_id, b.vehicle_id), self._too_close(a, b), "separation"
            )
        return forced

    def _resolve_conflicts(self, t_s: float) -> list[str]:
        """Stop a vehicle whose path would break separation within the horizon.

        The command-time check only covers the horizon after each command; paths accepted at
        different times can still converge later. Every step, each constrained pair is projected
        again; on a predicted conflict one moving vehicle holds (always allowed) and the planner
        replans it. The one to give way is a vehicle in transit rather than one returning to base,
        and among equals the one commanded last.
        """
        held: list[str] = []
        plans = [self._plan_of(self.get_state(v)) for v in self.vehicle_ids()]
        for a, b in itertools.combinations(plans, 2):
            if a.state.vehicle_id in held or b.state.vehicle_id in held:
                continue
            required = rules.min_separation_m(a.state, b.state, self.limits)
            moving = [p for p in (a, b) if p.speed_mps > 0.0]
            if required is None or not moving:
                continue
            if rules.first_conflict(a, b, required, self.base_xy_m, self.limits) is None:
                continue
            yielder = min(
                moving,
                key=lambda p: (
                    p.state.status is VehicleStatus.RTB,
                    -self._commanded_at.get(p.state.vehicle_id, 0.0),
                ),
            )
            vid = yielder.state.vehicle_id
            self.submit(Proposal(Hold(vid)), t_s)
            held.append(vid)
            logger.info("safety: %s holds to keep separation", vid)
        return held

    def _reserve_at_risk(self, state: VehicleState) -> bool:
        """True if the reserve would be broken within the energy horizon (status allowing RTB)."""
        if state.status not in (VehicleStatus.TRANSIT, VehicleStatus.IDLE):
            return False
        cfg = self._config(state)
        distance_m = float(np.hypot(state.x_m - self.base_xy_m[0], state.y_m - self.base_xy_m[1]))
        if distance_m < cfg.waypoint_radius_m:
            return False  # parked at base
        soon = (
            state.energy_frac
            - drain_rate_per_s(state.speed_mps, cfg) * self.limits.energy_horizon_s
        )
        return soon < rtb_energy_frac(distance_m, cfg)

    def _too_close(self, a: VehicleState, b: VehicleState) -> bool:
        required = rules.min_separation_m(a, b, self.limits)
        if required is None:
            return False
        for s in (a, b):
            if (
                np.hypot(s.x_m - self.base_xy_m[0], s.y_m - self.base_xy_m[1])
                < self.limits.base_zone_radius_m
            ):
                return False
        return bool(np.hypot(a.x_m - b.x_m, a.y_m - b.y_m) < required)

    def _count(self, key: tuple[str, ...], violating: bool, kind: str) -> None:
        """Count a violation once per episode (when it starts)."""
        if violating and key not in self._in_violation:
            self._in_violation.add(key)
            setattr(self.violations, kind, getattr(self.violations, kind) + 1)
            logger.warning("safety violation: %s", key)
        elif not violating:
            self._in_violation.discard(key)
