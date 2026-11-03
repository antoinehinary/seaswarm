"""Safety rules as pure functions: (state, proposed path, limits) -> Refusal or None.

Rules never touch a vehicle, so every limit can be unit-tested. Each reason is one sentence that
says what is wrong with numbers, so the planner (or the operator agent) can choose differently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from seaswarm.config import AreaConfig, Point, SafetyConfig, VehicleTypeConfig
from seaswarm.safety.refusal import Refusal, RefusalCode
from seaswarm.vehicles.backend import VehicleState, VehicleStatus, VehicleType
from seaswarm.vehicles.energy import drain_rate_per_s, rtb_energy_frac

XY = tuple[float, float]
Check = Callable[[], Refusal | None]


@dataclass(frozen=True)
class Geofence:
    """Axis-aligned rectangle, metres. Convex: a straight leg between inside points stays inside."""

    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    @classmethod
    def around(cls, area: AreaConfig, base: Point, margin_m: float) -> Geofence:
        """Rectangle around the search area (centred on the origin) and the base, plus margin."""
        hx, hy = area.width_m / 2, area.height_m / 2
        return cls(
            min(-hx, base.x_m) - margin_m,
            max(hx, base.x_m) + margin_m,
            min(-hy, base.y_m) - margin_m,
            max(hy, base.y_m) + margin_m,
        )

    def contains(self, x_m: float, y_m: float) -> bool:
        """True if the point is inside (boundary included)."""
        return self.x_min_m <= x_m <= self.x_max_m and self.y_min_m <= y_m <= self.y_max_m


def first_refusal(checks: Iterable[Check]) -> Refusal | None:
    """Run checks in order and return the first refusal, or None if all pass."""
    for check in checks:
        refusal = check()
        if refusal is not None:
            return refusal
    return None


def path_length_m(start: XY, waypoints: Sequence[XY]) -> float:
    """Length of the polyline start -> waypoints, metres."""
    points = np.array([start, *waypoints])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def position_along(start: XY, waypoints: Sequence[XY], distance_m: float) -> XY:
    """Point reached after travelling `distance_m` along the polyline (its end if shorter)."""
    point = np.asarray(start, dtype=float)
    for wp in waypoints:
        leg = np.asarray(wp, dtype=float) - point
        length = float(np.hypot(*leg))
        if distance_m <= length:
            return tuple(point + leg * (distance_m / length)) if length > 0 else tuple(point)
        distance_m -= length
        point = np.asarray(wp, dtype=float)
    return float(point[0]), float(point[1])


# --------------------------------------------------------------------------- checks


def check_available(state: VehicleState) -> Refusal | None:
    """Refuse tasks for a vehicle that is returning, in turnaround or stranded."""
    busy = {
        VehicleStatus.RTB: "is returning to base",
        VehicleStatus.TURNAROUND: "is at base swapping battery or refuelling",
        VehicleStatus.STRANDED: "is out of energy",
    }
    if state.status in busy:
        return Refusal(
            RefusalCode.VEHICLE_UNAVAILABLE,
            f"{state.vehicle_id} {busy[state.status]} and cannot take a task",
            state.vehicle_id,
        )
    return None


def check_geofence(vehicle_id: str, waypoints: Sequence[XY], fence: Geofence) -> Refusal | None:
    """Every waypoint inside the geofence (so, by convexity, every straight leg too)."""
    for x, y in waypoints:
        if not fence.contains(x, y):
            return Refusal(
                RefusalCode.GEOFENCE,
                f"{vehicle_id} waypoint ({x:.0f}, {y:.0f}) m is outside the geofence "
                f"(x {fence.x_min_m:.0f} to {fence.x_max_m:.0f} m, "
                f"y {fence.y_min_m:.0f} to {fence.y_max_m:.0f} m)",
                vehicle_id,
            )
    return None


def check_energy(
    state: VehicleState,
    waypoints: Sequence[XY],
    speed_mps: float,
    base_xy_m: XY,
    config: VehicleTypeConfig,
) -> Refusal | None:
    """After the task, the vehicle must still satisfy the return-to-base reserve."""
    start = (state.x_m, state.y_m)
    task_s = path_length_m(start, waypoints) / speed_mps
    energy_after = state.energy_frac - drain_rate_per_s(speed_mps, config) * task_s
    end = waypoints[-1] if waypoints else start
    needed = rtb_energy_frac(float(np.hypot(end[0] - base_xy_m[0], end[1] - base_xy_m[1])), config)
    if energy_after < needed:
        return Refusal(
            RefusalCode.ENERGY_RESERVE,
            f"{state.vehicle_id} would finish this task with {energy_after:.0%} energy, "
            f"but needs {needed:.0%} to return to base with the reserve",
            state.vehicle_id,
        )
    return None


def min_separation_m(a: VehicleState, b: VehicleState, limits: SafetyConfig) -> float | None:
    """Minimum horizontal distance required between two vehicles (None: no constraint)."""
    types = {a.vehicle_type, b.vehicle_type}
    if types == {VehicleType.DRONE}:
        return limits.separation_drone_m if a.altitude_m == b.altitude_m else None
    if types == {VehicleType.USV}:
        return limits.separation_usv_m
    drone = a if a.vehicle_type is VehicleType.DRONE else b
    if drone.altitude_m >= limits.drone_over_usv_min_altitude_m:
        return None
    return limits.separation_usv_m


@dataclass(frozen=True)
class Plan:
    """Where a vehicle is and where it is going: remaining waypoints at constant speed.

    `stop_m` is the vehicle's braking distance at its current speed, v^2 / (2 a_max): a hold or
    the end of a path is not an instant stop, so separation checks keep that much extra room.
    """

    state: VehicleState
    waypoints: Sequence[XY]
    speed_mps: float
    stop_m: float = 0.0


def first_conflict(
    a: Plan, b: Plan, required_m: float, base_xy_m: XY, limits: SafetyConfig
) -> tuple[float, float] | None:
    """First (time s, distance m) within the horizon at which two plans come closer than required.

    Both plans are projected along their waypoints at constant speed, in 1 s steps (straight legs,
    turns not modelled). The required distance is increased by the larger braking distance of the
    two. Moments when either vehicle is inside the base zone are exempt (deck operations).

    Escape exception: if the pair is already closer than required right now (outside the base
    zone), only closing counts, so the vehicles may always move apart and a hold can never trap
    them. Everywhere else the rule is strict, so two vehicles launched together that diverge too
    slowly are still refused.
    """
    required_m += max(a.stop_m, b.stop_m)
    base = np.asarray(base_xy_m)
    escaping = (
        np.hypot(a.state.x_m - b.state.x_m, a.state.y_m - b.state.y_m) < required_m
        and min(
            np.hypot(a.state.x_m - base[0], a.state.y_m - base[1]),
            np.hypot(b.state.x_m - base[0], b.state.y_m - base[1]),
        )
        >= limits.base_zone_radius_m
    )
    previous_m = np.inf
    for t in np.arange(0.0, limits.separation_horizon_s + 1e-9, 1.0):
        p = position_along((a.state.x_m, a.state.y_m), a.waypoints, a.speed_mps * t)
        q = position_along((b.state.x_m, b.state.y_m), b.waypoints, b.speed_mps * t)
        if min(np.hypot(*(p - base)), np.hypot(*(q - base))) < limits.base_zone_radius_m:
            continue
        distance = float(np.hypot(*np.subtract(p, q)))
        closing = distance < previous_m - 1e-9  # previous_m is inf at the first sample
        previous_m = distance
        counts = (t > 0.0 and closing) if escaping else True
        if distance < required_m and counts:
            return float(t), distance
    return None


def check_separation(
    plan: Plan, others: Sequence[Plan], base_xy_m: XY, limits: SafetyConfig
) -> Refusal | None:
    """Refuse a plan that would bring the vehicle too close to another one within the horizon."""
    for other in others:
        required = min_separation_m(plan.state, other.state, limits)
        if required is None:
            continue
        conflict = first_conflict(plan, other, required, base_xy_m, limits)
        if conflict is not None:
            t, distance = conflict
            braking = max(plan.stop_m, other.stop_m)
            return Refusal(
                RefusalCode.SEPARATION,
                f"{plan.state.vehicle_id} would come within {distance:.0f} m of "
                f"{other.state.vehicle_id} in {t:.0f} s; the minimum is {required:.0f} m "
                f"separation plus {braking:.0f} m braking distance",
                plan.state.vehicle_id,
            )
    return None


def check_tile(
    vehicle_id: str, tile_id: str | None, tile_owners: dict[str, str], dispatch: bool
) -> Refusal | None:
    """At most one searcher per tile per planning cycle, except dispatch to a detection."""
    if tile_id is None or dispatch:
        return None
    owner = tile_owners.get(tile_id)
    if owner is not None and owner != vehicle_id:
        return Refusal(
            RefusalCode.TILE_OCCUPIED,
            f"tile {tile_id} is already being searched by {owner}; choose another tile",
            vehicle_id,
        )
    return None
