"""Utility of searching each tile with a given vehicle, shared by the greedy and auction planners.

    U(vehicle, tile) = w_prio(tile) * expected_pos_gain / (t_travel + t_search)
    expected_pos_gain = POD_sweep * p(tile),  POD_sweep = 1 - exp(-W L / A)   (Koopman)

W is the vehicle's effective sweep width, A the tile area and L the track length flown over the
tile. Effort-limited search: L is the full lawnmower length (spacing W (1 - overlap)) if the
vehicle can afford it, otherwise the length it can still fly before it must head home with the
reserve; that effort is then spread evenly over the tile (spacing A / L). A tile is infeasible
(U = -inf) if not even one track across it is affordable. Without this, a sensor with a tiny
sweep width (an EO camera at night) makes every full pattern longer than a battery and the
vehicle never leaves the base.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from seaswarm.planning.base import MissionView, VehicleCapability
from seaswarm.planning.tiles import Rect, lawnmower_length_m
from seaswarm.safety.rules import XY
from seaswarm.vehicles.backend import VehicleState
from seaswarm.vehicles.energy import drain_rate_per_s, rtb_energy_frac

Array = NDArray[np.float64]


def affordable_prefix(
    state: VehicleState, cap: VehicleCapability, path: tuple[XY, ...], base_xy_m: XY
) -> tuple[XY, ...]:
    """Longest prefix of `path` the vehicle can fly at cruise and still return with the reserve.

    Same rule as the safety layer's energy check, so an affordable prefix is never refused for
    energy. Returns an empty tuple if not even the first waypoint is affordable.
    """
    rate = drain_rate_per_s(cap.cruise_speed_mps, cap.config)
    energy = state.energy_frac
    position = (state.x_m, state.y_m)
    for i, waypoint in enumerate(path):
        leg_m = float(np.hypot(waypoint[0] - position[0], waypoint[1] - position[1]))
        energy -= rate * leg_m / cap.cruise_speed_mps
        home_m = float(np.hypot(waypoint[0] - base_xy_m[0], waypoint[1] - base_xy_m[1]))
        if energy < rtb_energy_frac(home_m, cap.config):
            return path[:i]
        position = waypoint
    return path


@dataclass(frozen=True)
class Effort:
    """Effort-limited sweep of tiles by one vehicle; arrays have the shape of the tile centres."""

    travel_s: Array
    search_s: Array
    track_m: Array
    pod: Array
    feasible: Array

    def spacing_m(self, cap: VehicleCapability, tile_size_m: float, index: tuple) -> float:
        """Track spacing that spreads the affordable track length evenly over one tile."""
        return max(cap.track_spacing_m, tile_size_m**2 / float(self.track_m[index]))


def tile_effort(
    state: VehicleState,
    cap: VehicleCapability,
    x_m: Array,
    y_m: Array,
    tile_size_m: float,
    base_xy_m: XY,
) -> Effort:
    """Effort the vehicle can put into each tile centred at (x_m, y_m) on its current energy."""
    speed = cap.cruise_speed_mps
    rate = drain_rate_per_s(speed, cap.config)
    travel_s = np.hypot(x_m - state.x_m, y_m - state.y_m) / speed
    home_m = np.hypot(x_m - base_xy_m[0], y_m - base_xy_m[1])
    spare = state.energy_frac - rate * travel_s - rtb_energy_frac(home_m, cap.config)
    affordable_m = np.maximum(spare, 0.0) / rate * speed
    full_m = lawnmower_length_m(Rect(0.0, 0.0, tile_size_m, tile_size_m), cap.track_spacing_m)
    track_m = np.minimum(full_m, affordable_m)
    pod = -np.expm1(-cap.sweep_width_m * track_m / tile_size_m**2)
    return Effort(travel_s, track_m / speed, track_m, pod, track_m >= tile_size_m)


@dataclass(frozen=True)
class TileScores:
    """Per-tile utility and its ingredients for one vehicle, each shape (rows, cols)."""

    utility: Array
    gain: Array
    duration_s: Array


def tile_scores(
    state: VehicleState,
    cap: VehicleCapability,
    values: Array,
    weights: Array,
    centres_m: tuple[Array, Array],
    tile_size_m: float,
    view: MissionView,
) -> tuple[TileScores, Effort]:
    """Utility of every tile for one vehicle (travel measured to the tile centre)."""
    effort = tile_effort(state, cap, *centres_m, tile_size_m, view.base_xy_m)
    duration = effort.travel_s + effort.search_s
    gain = weights * effort.pod * values
    with np.errstate(divide="ignore", invalid="ignore"):
        utility = np.where(effort.feasible, gain / duration, -np.inf)
    return TileScores(utility, gain, duration), effort
