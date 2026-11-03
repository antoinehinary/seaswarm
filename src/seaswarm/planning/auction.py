"""Sequential single-item auction with bundles (the centralised core of CBBA).

Tasks are the K = n_tasks_per_vehicle x fleet size most valuable tiles. Each round:
1. every available vehicle sends one `Bid`: its best marginal gain for inserting a remaining task
   at the best position of its current bundle. Tiles being searched by another vehicle, or
   recently refused for this one, are not on offer (no swaps, no cycles);
2. the auctioneer awards the task to the highest bid (`Award`; ties broken by the scenario's
   `planner_tiebreak` random stream);
until every bundle holds B tasks or nobody bids a positive gain.

Bundle value (time-discounted reward, as in Choi, Brunet and How 2009):

    S(bundle) = sum_j gain_j * exp(-t_j / tau),  gain_j = w_prio * POD_sweep * p(tile_j)

with t_j the time at which task j would be finished, travelling between tile centres. Marginal
bids shrink as a bundle grows, the "diminishing marginal gain" property CBBA relies on to
converge. Each task's sweep is effort-limited (see `seaswarm.planning.utility`), evaluated from
the vehicle's current energy. A bundle is infeasible if, once done, the vehicle could not return
to base with the energy reserve. Receding horizon: a vehicle executes only the first task of its
bundle; the auction is rerun at the next replan.

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
from seaswarm.planning.tiles import TileGrid, lawnmower
from seaswarm.planning.utility import Effort, affordable_prefix, tile_effort
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
class _Tasks:
    """Candidate tiles: grid indices, centres and weighted probability."""

    rows: np.ndarray
    cols: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    value: np.ndarray

    def tile_id(self, i: int) -> str:
        return TileGrid.tile_id(int(self.rows[i]), int(self.cols[i]))


@dataclass(frozen=True)
class _Bidder:
    """One vehicle with its effort on every task."""

    state: VehicleState
    cap: VehicleCapability
    effort: Effort
    gain: np.ndarray  # POD of the effort-limited sweep x weighted value, per task


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
        states = sorted((s for s in vehicles if s.status in AVAILABLE), key=lambda s: s.vehicle_id)
        if not states:
            return {}
        tasks = self._tasks(view, pmap, prefs, self.config.auction_tasks_per_vehicle * len(states))
        size = self.tiles.size_m
        bidders = []
        for state in states:
            cap = view.capabilities[state.vehicle_id]
            effort = tile_effort(state, cap, tasks.x_m, tasks.y_m, size, view.base_xy_m)
            bidders.append(_Bidder(state, cap, effort, effort.pod * tasks.value))
        bundles = self._auction(view, bidders, tasks)

        assignments = {}
        for b in bidders:
            bundle = bundles[b.state.vehicle_id]
            if not bundle:
                continue
            first = bundle[0]
            spacing = b.effort.spacing_m(b.cap, size, (first,))
            rect = self.tiles.rect(int(tasks.rows[first]), int(tasks.cols[first]))
            path = lawnmower(rect, spacing, (b.state.x_m, b.state.y_m))
            path = affordable_prefix(b.state, b.cap, path, view.base_xy_m)
            if not path:
                continue
            times = self._finish_times(b, tasks, bundle)
            planned = tuple(
                SearchTask(tasks.tile_id(i), path if k == 0 else (), float(b.gain[i]), view.t_s + t)
                for k, (i, t) in enumerate(zip(bundle, times, strict=True))
            )
            assignments[b.state.vehicle_id] = Assignment(b.state.vehicle_id, planned)
        return without_held_tiles(assignments, view)

    # ------------------------------------------------------------------ auction rounds

    def _auction(
        self, view: MissionView, bidders: list[_Bidder], tasks: _Tasks
    ) -> dict[str, list[int]]:
        bundles: dict[str, list[int]] = {b.state.vehicle_id: [] for b in bidders}
        remaining = set(range(len(tasks.value)))
        self.last_awards = []
        while remaining:
            bids = [
                bid
                for b in bidders
                if len(bundles[b.state.vehicle_id]) < self.config.auction_bundle_size
                and (bid := self._best_bid(b, view, bundles[b.state.vehicle_id], tasks, remaining))
            ]
            if not bids:
                break
            best = max(bid.value for bid in bids)
            leaders = [bid for bid in bids if bid.value >= best * (1.0 - 1e-12)]
            winner = (
                leaders[int(self._rng.integers(len(leaders)))] if len(leaders) > 1 else leaders[0]
            )
            award = Award(winner.vehicle_id, winner.task, winner.position)
            self.last_awards.append(award)
            bundles[award.vehicle_id].insert(award.position, award.task)
            remaining.discard(award.task)
        return bundles

    def _best_bid(
        self, b: _Bidder, view: MissionView, bundle: list[int], tasks: _Tasks, remaining: set[int]
    ) -> Bid | None:
        """The vehicle's message: its single best positive, feasible insertion."""
        vid = b.state.vehicle_id
        excluded = set(view.excluded.get(vid, frozenset()))
        excluded |= {t for v, t in view.current_tiles.items() if t and v != vid}
        current = self._bundle_value(b, view, tasks, bundle)
        best: Bid | None = None
        for task in sorted(remaining):
            if not b.effort.feasible[task] or tasks.tile_id(task) in excluded:
                continue  # unaffordable, recently refused for this vehicle, or searched by another
            for position in range(len(bundle) + 1):
                candidate = bundle[:position] + [task] + bundle[position:]
                gain = self._bundle_value(b, view, tasks, candidate) - current
                if gain > 0.0 and (best is None or gain > best.value):
                    best = Bid(vid, task, position, gain)
        return best

    # ------------------------------------------------------------------ scoring

    def _tasks(
        self, view: MissionView, pmap: ProbabilityMap, prefs: OperatorPrefs, n: int
    ) -> _Tasks:
        """The n tiles of highest weighted probability, in a deterministic order."""
        cx, cy = self.tiles.centres_m()
        weighted = prefs.weights(cx, cy, view.t_s) * self.tiles.values(pmap)
        order = np.argsort(-weighted.ravel(), kind="stable")[:n]
        order = order[weighted.ravel()[order] > 0.0]
        rows, cols = np.unravel_index(order, self.tiles.shape)
        return _Tasks(rows, cols, cx[rows, cols], cy[rows, cols], weighted[rows, cols])

    @staticmethod
    def _finish_times(b: _Bidder, tasks: _Tasks, bundle: list[int]) -> list[float]:
        """Time (s from now) at which each task of the bundle would be finished."""
        times, t, x, y = [], 0.0, b.state.x_m, b.state.y_m
        for i in bundle:
            t += np.hypot(tasks.x_m[i] - x, tasks.y_m[i] - y) / b.cap.cruise_speed_mps
            t += float(b.effort.search_s[i])
            times.append(t)
            x, y = tasks.x_m[i], tasks.y_m[i]
        return times

    def _bundle_value(
        self, b: _Bidder, view: MissionView, tasks: _Tasks, bundle: list[int]
    ) -> float:
        """Time-discounted reward of a bundle; -inf if it breaks the energy reserve."""
        if not bundle:
            return 0.0
        times = self._finish_times(b, tasks, bundle)
        rate = drain_rate_per_s(b.cap.cruise_speed_mps, b.cap.config)
        last = bundle[-1]
        home_m = float(
            np.hypot(tasks.x_m[last] - view.base_xy_m[0], tasks.y_m[last] - view.base_xy_m[1])
        )
        if b.state.energy_frac - rate * times[-1] < rtb_energy_frac(home_m, b.cap.config):
            return -np.inf
        tau = self.config.reward_time_constant_s
        return float(sum(b.gain[i] * np.exp(-t / tau) for i, t in zip(bundle, times, strict=True)))
