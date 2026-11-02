"""Sequential single-item auction with bundles (the centralised core of CBBA).

Tasks are the K = n_tasks_per_vehicle x fleet size most valuable tiles. Each round:
1. every available vehicle sends one `Bid`: its best marginal gain for inserting a remaining task
   at the best position of its current bundle;
   Tiles being searched by another vehicle are not on offer (no swaps, no cycles);
2. the auctioneer awards the task to the highest bid (`Award`; ties broken by the scenario's
   `planner_tiebreak` random stream);
until every bundle holds B tasks or nobody bids a positive gain.

Bundle value (time-discounted reward, as in Choi, Brunet and How 2009):

    S(bundle) = sum_j gain_j * exp(-t_j / tau),  gain_j = w_prio * POD_sweep * p(tile_j)

with t_j the time at which task j would be finished, travelling between tile centres. Marginal
bids shrink as a bundle grows, the "diminishing marginal gain" property CBBA relies on to
converge. A bundle is infeasible if, once done, the vehicle could not return to base with the
energy reserve. Receding horizon: a vehicle executes only the first task of its bundle; the
auction is rerun at the next replan.

Only `_auction` would change in a decentralised version: vehicles would exchange the same bids
and awards with neighbours in radio range instead of with a central auctioneer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from seaswarm.config import PlannerConfig
from seaswarm.planning.base import (
    Assignment,
    MissionView,
    OperatorPrefs,
    SearchTask,
    VehicleCapability,
    without_held_tiles,
)
from seaswarm.planning.greedy import AVAILABLE
from seaswarm.planning.tiles import Rect, TileGrid, lawnmower, lawnmower_length_m
from seaswarm.planning.utility import affordable_prefix
from seaswarm.search.pod import random_search_pod
from seaswarm.search.probability_map import ProbabilityMap
from seaswarm.vehicles.backend import VehicleState
from seaswarm.vehicles.energy import drain_rate_per_s, rtb_energy_frac


@dataclass(frozen=True)
class Bid:
    """A vehicle's offer: inserting `task` at `position` of its bundle adds `value`."""

    vehicle_id: str
    task: int
    position: int
    value: float


@dataclass(frozen=True)
class Award:
    """The auctioneer's decision: `task` goes to `vehicle_id` at `position`."""

    vehicle_id: str
    task: int
    position: int


@dataclass(frozen=True)
class _Task:
    row: int
    col: int
    x_m: float
    y_m: float
    weighted_value: float


class AuctionPlanner:
    """Bundle auction over the most valuable tiles (see module docstring)."""

    name = "auction"

    def __init__(
        self, tiles: TileGrid, config: PlannerConfig, tiebreak: np.random.Generator
    ) -> None:
        """Plan over `tiles`; ties are broken with `tiebreak` (the planner_tiebreak stream)."""
        self.tiles = tiles
        self.config = config
        self._rng = tiebreak
        self.last_awards: list[Award] = []

    def plan(
        self,
        view: MissionView,
        pmap: ProbabilityMap,
        vehicles: Sequence[VehicleState],
        prefs: OperatorPrefs,
    ) -> dict[str, Assignment]:
        """Bundles for every available vehicle; each executes its first task."""
        bidders = sorted((s for s in vehicles if s.status in AVAILABLE), key=lambda s: s.vehicle_id)
        if not bidders:
            return {}
        tasks = self._tasks(
            view, pmap, prefs, n=self.config.auction_tasks_per_vehicle * len(bidders)
        )
        bundles = self._auction(view, bidders, tasks)

        assignments = {}
        for state in bidders:
            bundle = bundles[state.vehicle_id]
            if not bundle:
                continue
            cap = view.capabilities[state.vehicle_id]
            first = tasks[bundle[0]]
            path = lawnmower(
                self.tiles.rect(first.row, first.col), cap.track_spacing_m, (state.x_m, state.y_m)
            )
            path = affordable_prefix(state, cap, path, view.base_xy_m)
            if not path:
                continue
            times = self._finish_times(state, cap, [tasks[i] for i in bundle])
            planned = [
                SearchTask(
                    TileGrid.tile_id(tasks[i].row, tasks[i].col),
                    path if k == 0 else (),
                    self._gain(cap, tasks[i]),
                    view.t_s + times[k],
                )
                for k, i in enumerate(bundle)
            ]
            assignments[state.vehicle_id] = Assignment(state.vehicle_id, tuple(planned))
        return without_held_tiles(assignments, view)

    # ------------------------------------------------------------------ auction rounds

    def _auction(
        self, view: MissionView, bidders: list[VehicleState], tasks: list[_Task]
    ) -> dict[str, list[int]]:
        bundles: dict[str, list[int]] = {s.vehicle_id: [] for s in bidders}
        remaining = set(range(len(tasks)))
        self.last_awards = []
        while remaining:
            bids = [
                bid
                for state in bidders
                if len(bundles[state.vehicle_id]) < self.config.auction_bundle_size
                and (
                    bid := self._best_bid(state, view, bundles[state.vehicle_id], tasks, remaining)
                )
            ]
            if not bids:
                break
            best = max(b.value for b in bids)
            leaders = [b for b in bids if b.value >= best * (1.0 - 1e-12)]
            winner = (
                leaders[int(self._rng.integers(len(leaders)))] if len(leaders) > 1 else leaders[0]
            )
            award = Award(winner.vehicle_id, winner.task, winner.position)
            self.last_awards.append(award)
            bundles[award.vehicle_id].insert(award.position, award.task)
            remaining.discard(award.task)
        return bundles

    def _best_bid(
        self,
        state: VehicleState,
        view: MissionView,
        bundle: list[int],
        tasks: list[_Task],
        remaining: set[int],
    ) -> Bid | None:
        """The vehicle's message: its single best positive, feasible insertion."""
        cap = view.capabilities[state.vehicle_id]
        current = self._bundle_value(state, cap, view, [tasks[i] for i in bundle])
        excluded = set(view.excluded.get(state.vehicle_id, frozenset()))
        excluded |= {t for v, t in view.current_tiles.items() if t and v != state.vehicle_id}
        best: Bid | None = None
        for task in sorted(remaining):
            if TileGrid.tile_id(tasks[task].row, tasks[task].col) in excluded:
                continue  # recently refused for this vehicle, or being searched by another
            for position in range(len(bundle) + 1):
                candidate = bundle[:position] + [task] + bundle[position:]
                value = self._bundle_value(state, cap, view, [tasks[i] for i in candidate])
                gain = value - current
                if gain > 0.0 and (best is None or gain > best.value):
                    best = Bid(state.vehicle_id, task, position, gain)
        return best

    # ------------------------------------------------------------------ scoring

    def _tasks(
        self, view: MissionView, pmap: ProbabilityMap, prefs: OperatorPrefs, n: int
    ) -> list[_Task]:
        """The n tiles of highest weighted probability, in a deterministic order."""
        cx, cy = self.tiles.centres_m()
        weighted = prefs.weights(cx, cy, view.t_s) * self.tiles.values(pmap)
        order = np.argsort(-weighted.ravel(), kind="stable")[:n]
        rows, cols = np.unravel_index(order, self.tiles.shape)
        return [
            _Task(int(r), int(c), float(cx[r, c]), float(cy[r, c]), float(weighted[r, c]))
            for r, c in zip(rows, cols, strict=True)
            if weighted[r, c] > 0.0
        ]

    def _search_s(self, cap: VehicleCapability) -> float:
        size = self.tiles.size_m
        return (
            lawnmower_length_m(Rect(0.0, 0.0, size, size), cap.track_spacing_m)
            / cap.cruise_speed_mps
        )

    def _gain(self, cap: VehicleCapability, task: _Task) -> float:
        size = self.tiles.size_m
        track_m = lawnmower_length_m(Rect(0.0, 0.0, size, size), cap.track_spacing_m)
        return random_search_pod(cap.sweep_width_m, track_m, size**2) * task.weighted_value

    def _finish_times(
        self, state: VehicleState, cap: VehicleCapability, bundle: list[_Task]
    ) -> list[float]:
        """Time (s from now) at which each task of the bundle would be finished."""
        times, t, x, y = [], 0.0, state.x_m, state.y_m
        for task in bundle:
            t += np.hypot(task.x_m - x, task.y_m - y) / cap.cruise_speed_mps + self._search_s(cap)
            times.append(t)
            x, y = task.x_m, task.y_m
        return times

    def _bundle_value(
        self, state: VehicleState, cap: VehicleCapability, view: MissionView, bundle: list[_Task]
    ) -> float:
        """Time-discounted reward of a bundle; -inf if it breaks the energy reserve."""
        if not bundle:
            return 0.0
        times = self._finish_times(state, cap, bundle)
        energy_left = (
            state.energy_frac - drain_rate_per_s(cap.cruise_speed_mps, cap.config) * times[-1]
        )
        last = bundle[-1]
        home_m = float(np.hypot(last.x_m - view.base_xy_m[0], last.y_m - view.base_xy_m[1]))
        if energy_left < rtb_energy_frac(home_m, cap.config):
            return -np.inf
        tau = self.config.reward_time_constant_s
        return float(
            sum(
                self._gain(cap, task) * np.exp(-t / tau)
                for task, t in zip(bundle, times, strict=True)
            )
        )
