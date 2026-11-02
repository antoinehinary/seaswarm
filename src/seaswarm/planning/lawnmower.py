"""Lawnmower baseline: sweep the high-prior rectangle in fixed strips, ignoring the map after start.

At the first plan, the rectangle holding `prior_mass` of the belief is split into north-south
strips, one per vehicle, with widths proportional to each vehicle's sweep rate (W * v). Each
vehicle mows its strip with north-south tracks. Whenever a vehicle is idle (start, back from a
battery swap) it gets the next sortie of its strip: the longest part it can fly and still return
with the energy reserve, shifted by the forecast mean drift since the start. When a strip is
finished it starts again.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from seaswarm.planning.base import Assignment, MissionView, OperatorPrefs, SearchTask
from seaswarm.planning.tiles import Rect, lawnmower
from seaswarm.planning.utility import affordable_prefix
from seaswarm.safety.rules import XY
from seaswarm.search.probability_map import ProbabilityMap
from seaswarm.vehicles.backend import VehicleState, VehicleStatus


def mass_rectangle(pmap: ProbabilityMap, mass: float) -> Rect:
    """Bounding rectangle of the most probable cells holding `mass` of the in-grid probability."""
    order = np.argsort(pmap.p.ravel())[::-1]
    cumulative = np.cumsum(pmap.p.ravel()[order])
    n_cells = int(np.searchsorted(cumulative, mass * cumulative[-1]) + 1)
    rows, cols = np.unravel_index(order[:n_cells], pmap.grid.shape)
    g, h = pmap.grid, pmap.grid.cell_size_m
    return Rect(
        g.x_min_m + cols.min() * h,
        g.y_min_m + rows.min() * h,
        g.x_min_m + (cols.max() + 1) * h,
        g.y_min_m + (rows.max() + 1) * h,
    )


class LawnmowerPlanner:
    """Fixed parallel-sweep baseline (see module docstring)."""

    name = "lawnmower"

    def __init__(self, prior_mass: float) -> None:
        """Cover the rectangle holding `prior_mass` of the belief at the first plan."""
        self.prior_mass = prior_mass
        self._patterns: dict[str, tuple[XY, ...]] = {}
        self._t0_s = 0.0
        self._done: dict[str, int] = {}  # waypoints of the pattern already flown
        self._issued: dict[str, tuple[int, int]] = {}  # (first index, count) of the last issue

    @property
    def patterns(self) -> dict[str, tuple[XY, ...]]:
        """Full strip pattern of every vehicle, as built at the first plan (before drift)."""
        return dict(self._patterns)

    def plan(
        self,
        view: MissionView,
        pmap: ProbabilityMap,
        vehicles: Sequence[VehicleState],
        prefs: OperatorPrefs,
    ) -> dict[str, Assignment]:
        """The rest of its strip for every idle vehicle; others continue."""
        if not self._patterns:
            self._build(view, pmap)
        dx, dy = (np.array(view.drift_mps) * (view.t_s - self._t0_s)).tolist()
        assignments = {}
        for state in sorted(vehicles, key=lambda s: s.vehicle_id):
            vid = state.vehicle_id
            if vid in self._issued:  # record progress on the last issued part
                first, count = self._issued[vid]
                at_base = (
                    np.hypot(state.x_m - view.base_xy_m[0], state.y_m - view.base_xy_m[1])
                    < view.capabilities[vid].config.waypoint_radius_m
                )
                if state.status is VehicleStatus.TRANSIT:
                    self._done[vid] = first + count - state.waypoints_left
                elif state.status is VehicleStatus.IDLE and not at_base:
                    self._done[vid] = first + count  # finished the issued part
            if state.status is not VehicleStatus.IDLE or None in view.excluded.get(vid, ()):
                continue
            pattern = self._patterns[vid]
            start = self._done.get(vid, 0) % len(pattern)
            rest = tuple((x + dx, y + dy) for x, y in pattern[start:])
            sortie = affordable_prefix(state, view.capabilities[vid], rest, view.base_xy_m)
            if not sortie:
                continue
            self._issued[vid] = (start, len(sortie))
            assignments[vid] = Assignment(vid, (SearchTask(None, sortie, 0.0, view.t_s),))
        return assignments

    def _build(self, view: MissionView, pmap: ProbabilityMap) -> None:
        self._t0_s = view.t_s
        rect = mass_rectangle(pmap, self.prior_mass)
        caps = [view.capabilities[v] for v in sorted(view.capabilities)]
        rates = np.array([c.sweep_rate_m2_s for c in caps])
        edges = rect.x_min_m + rect.width_m * np.concatenate(
            [[0.0], np.cumsum(rates) / rates.sum()]
        )
        for cap, x0, x1 in zip(caps, edges[:-1], edges[1:], strict=True):
            strip = Rect(float(x0), rect.y_min_m, float(x1), rect.y_max_m)
            self._patterns[cap.vehicle_id] = lawnmower(
                strip, cap.track_spacing_m, view.base_xy_m, north_south=True
            )
