"""Utility of searching each tile with a given vehicle, shared by the greedy and auction planners.

    U(vehicle, tile) = w_prio(tile) * expected_pos_gain / (t_travel + t_search)
    expected_pos_gain = POD_sweep * p(tile),  POD_sweep = 1 - exp(-W L / A)   (Koopman)

W is the vehicle's effective sweep width, L the lawnmower track length over the tile, A the tile
area. A tile is infeasible (U = -inf) if, after travelling there and searching it, the vehicle
could not return to base with the energy reserve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from seaswarm.planning.base import MissionView, VehicleCapability
from seaswarm.planning.tiles import Rect, lawnmower_length_m
from seaswarm.safety.rules import XY
from seaswarm.search.pod import random_search_pod
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
) -> TileScores:
    """Utility of every tile for one vehicle (travel measured to the tile centre)."""
    cx, cy = centres_m
    track_m = lawnmower_length_m(Rect(0.0, 0.0, tile_size_m, tile_size_m), cap.track_spacing_m)
    pod = random_search_pod(cap.sweep_width_m, track_m, tile_size_m**2)
    t_search = track_m / cap.cruise_speed_mps
    t_travel = np.hypot(cx - state.x_m, cy - state.y_m) / cap.cruise_speed_mps
    duration = t_travel + t_search
    gain = weights * pod * values

    energy_after = state.energy_frac - drain_rate_per_s(cap.cruise_speed_mps, cap.config) * duration
    home_m = np.hypot(cx - view.base_xy_m[0], cy - view.base_xy_m[1])
    needed = np.vectorize(lambda d: rtb_energy_frac(d, cap.config))(home_m)
    utility = np.where(energy_after >= needed, gain / duration, -np.inf)
    return TileScores(utility, gain, duration)
