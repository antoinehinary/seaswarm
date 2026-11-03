"""Bayesian greedy allocation: each vehicle takes the best tile nobody else has claimed."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from seaswarm.planning.base import Assignment, MissionView, OperatorPrefs, SearchTask
from seaswarm.planning.tiles import TileGrid, lawnmower
from seaswarm.planning.utility import affordable_prefix, tile_scores
from seaswarm.search.probability_map import ProbabilityMap
from seaswarm.vehicles.backend import VehicleState, VehicleStatus

AVAILABLE = (VehicleStatus.IDLE, VehicleStatus.TRANSIT)


class GreedyPlanner:
    """Vehicles in id order pick the feasible unclaimed tile of highest utility.

    Deterministic: vehicles are sorted by id and ties go to the first tile in row-major order.
    """

    name = "greedy"

    def __init__(self, tiles: TileGrid) -> None:
        """Plan over `tiles`."""
        self.tiles = tiles

    def plan(
        self,
        view: MissionView,
        pmap: ProbabilityMap,
        vehicles: Sequence[VehicleState],
        prefs: OperatorPrefs,
    ) -> dict[str, Assignment]:
        """One tile per available vehicle."""
        values = self.tiles.values(pmap)
        centres = self.tiles.centres_m()
        weights = prefs.weights(*centres, view.t_s)
        # tiles being searched stay claimed unless their own vehicle is re-planned onto another
        claimed = np.zeros(self.tiles.shape, dtype=bool)
        for tile_id in view.current_tiles.values():
            if tile_id is not None:
                claimed[self._index(tile_id)] = True
        assignments = {}
        for state in sorted(vehicles, key=lambda s: s.vehicle_id):
            if state.status not in AVAILABLE:
                continue
            own = view.current_tiles.get(state.vehicle_id)
            if own is not None:
                claimed[self._index(own)] = False
            cap = view.capabilities[state.vehicle_id]
            scores, effort = tile_scores(
                state, cap, values, weights, centres, self.tiles.size_m, view
            )
            blocked = claimed.copy()
            for tile_id in view.excluded.get(state.vehicle_id, frozenset()) - {None}:
                blocked[self._index(tile_id)] = True
            utility = np.where(blocked, -np.inf, scores.utility)
            best = int(np.argmax(utility))
            if not np.isfinite(utility.flat[best]) or utility.flat[best] <= 0.0:
                if own is not None:
                    claimed[self._index(own)] = True
                continue
            row, col = np.unravel_index(best, self.tiles.shape)
            claimed[row, col] = True
            spacing = effort.spacing_m(cap, self.tiles.size_m, (row, col))
            path = lawnmower(self.tiles.rect(row, col), spacing, (state.x_m, state.y_m))
            path = affordable_prefix(state, cap, path, view.base_xy_m)
            if not path:
                continue
            task = SearchTask(
                TileGrid.tile_id(row, col),
                path,
                float(scores.gain[row, col]),
                view.t_s + float(scores.duration_s[row, col]),
            )
            assignments[state.vehicle_id] = Assignment(state.vehicle_id, (task,))
        return assignments

    @staticmethod
    def _index(tile_id: str) -> tuple[int, int]:
        _, row, col = tile_id.split("_")
        return int(row), int(col)
