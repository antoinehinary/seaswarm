"""Consensus-Based Bundle Algorithm (CBBA), Choi, Brunet and How, IEEE T-RO 25(4), 2009.

The decentralised version of the bundle auction: no auctioneer, each vehicle only talks to the
vehicles within radio range. It reuses the auction's tasks and bundle score (`AuctionPlanner`);
only the allocation protocol differs, so the two can be compared directly.

Each vehicle i keeps, for every task j: y_i[j] the highest bid it knows of, z_i[j] who made it,
and s_i[k] the time of its latest information from each vehicle k. Rounds alternate:

1. Bundle construction. Vehicle i adds tasks one at a time, at the best position of its path,
   choosing the largest marginal gain c_ij that beats the best known bid y_i[j] (ties go to the
   lower id). It records y_i[j] = c_ij, z_i[j] = i.
2. Consensus. Each vehicle receives (y, z, s) from its neighbours and, task by task, updates,
   resets or leaves its own view with the decision table of the paper (`_decide`).
3. Release. If vehicle i lost a task of its bundle, it drops it and every task it added after it,
   because their marginal gains were computed assuming the lost one.

It stops when a round changes nothing. With a connected network and a score with diminishing
marginal gains (ours: time-discounted reward) CBBA converges to a conflict-free assignment within
about (number of tasks) x (network diameter) rounds. If the network is split, each part agrees
internally but two parts may claim the same tile; the safety layer then refuses the second
vehicle (TILE_OCCUPIED) and it replans. The belief is shared by all vehicles (as if broadcast):
only the allocation is decentralised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from seaswarm.config import PlannerConfig
from seaswarm.planning.auction import AuctionPlanner, _Bidder, _Tasks
from seaswarm.planning.base import MissionView
from seaswarm.planning.tiles import TileGrid

UPDATE, RESET, LEAVE = "update", "reset", "leave"


@dataclass
class Agent:
    """One vehicle's CBBA state."""

    vehicle_id: str
    bundle: list[int] = field(default_factory=list)  # tasks in the order they were added
    path: list[int] = field(default_factory=list)  # the same tasks in execution order
    y: dict[int, float] = field(default_factory=dict)  # best known bid per task
    z: dict[int, str | None] = field(default_factory=dict)  # who made it
    s: dict[str, float] = field(default_factory=dict)  # time of latest info from each vehicle


@dataclass(frozen=True)
class Message:
    """What a vehicle broadcasts to its neighbours each round."""

    sender: str
    y: dict[int, float]
    z: dict[int, str | None]
    s: dict[str, float]


def comm_graph(positions: dict[str, tuple[float, float]], range_m: float) -> dict[str, set[str]]:
    """Neighbours of each vehicle: every other vehicle within `range_m` (inf: all)."""
    graph: dict[str, set[str]] = {v: set() for v in positions}
    for a, b in combinations(sorted(positions), 2):
        if (
            np.hypot(positions[a][0] - positions[b][0], positions[a][1] - positions[b][1])
            <= range_m
        ):
            graph[a].add(b)
            graph[b].add(a)
    return graph


def _beats(bid: float, bidder: str, best: float, holder: str | None) -> bool:
    """True if (bid, bidder) beats the current (best, holder); ties go to the lower id."""
    if holder is None:
        return bid > 0.0
    return bid > best + 1e-12 or (abs(bid - best) <= 1e-12 and bidder < holder)


def _decide(i: str, k: str, task: int, me: Agent, msg: Message) -> str:
    """CBBA decision table (Choi et al. 2009, Table 1): receiver i, sender k, one task.

    z_k is who the sender thinks wins, z_i who the receiver thinks wins; s are the timestamps,
    used to tell which of them has the more recent news about a third vehicle m.
    """
    zk, zi = msg.z.get(task), me.z.get(task)
    yk, yi = msg.y.get(task, 0.0), me.y.get(task, 0.0)

    def newer(m: str) -> bool:
        return msg.s.get(m, -1.0) > me.s.get(m, -1.0)

    if zk == k:  # the sender thinks it wins
        if zi == i:
            return UPDATE if _beats(yk, k, yi, i) else LEAVE
        if zi == k or zi is None:
            return UPDATE
        return UPDATE if newer(zi) or _beats(yk, k, yi, zi) else LEAVE
    if zk == i:  # the sender thinks the receiver wins
        if zi == i or zi is None:
            return LEAVE
        if zi == k:
            return RESET
        return RESET if newer(zi) else LEAVE
    if zk is not None:  # the sender thinks a third vehicle m wins
        m = zk
        if zi == i:
            return UPDATE if newer(m) and _beats(yk, m, yi, i) else LEAVE
        if zi == k:
            return UPDATE if newer(m) else RESET
        if zi == m:
            return UPDATE if newer(m) else LEAVE
        if zi is None:
            return UPDATE if newer(m) else LEAVE
        n = zi  # the receiver thinks a fourth vehicle n wins
        if newer(m) and (newer(n) or _beats(yk, m, yi, n)):
            return UPDATE
        if newer(n) and not newer(m):
            return RESET
        return LEAVE
    # the sender thinks nobody wins
    if zi == i or zi is None:
        return LEAVE
    if zi == k:
        return UPDATE
    return UPDATE if newer(zi) else LEAVE


class CBBAPlanner(AuctionPlanner):
    """The auction's tasks and bundle score, allocated by CBBA over a range-limited network."""

    name = "cbba"

    def __init__(
        self,
        tiles: TileGrid,
        config: PlannerConfig,
        tiebreak: np.random.Generator,
        comm_range_m: float,
        max_rounds: int = 100,
    ) -> None:
        """CBBA planner whose vehicles hear each other within `comm_range_m` (inf: everyone)."""
        super().__init__(tiles, config, tiebreak)
        self.comm_range_m = comm_range_m
        self.max_rounds = max_rounds
        self.last_rounds = 0
        self.last_conflicts = 0  # tasks claimed by more than one vehicle when CBBA stopped

    def _auction(
        self, view: MissionView, bidders: list[_Bidder], tasks: _Tasks
    ) -> dict[str, list[int]]:
        """Run CBBA rounds; return each vehicle's path (bundle in execution order)."""
        by_id = {b.state.vehicle_id: b for b in bidders}
        positions = {v: (b.state.x_m, b.state.y_m) for v, b in by_id.items()}
        graph = comm_graph(positions, self.comm_range_m)
        agents = {v: Agent(v) for v in sorted(by_id)}
        for rounds in range(1, self.max_rounds + 1):
            before = {v: (list(a.path), dict(a.y), dict(a.z)) for v, a in agents.items()}
            for v, agent in agents.items():
                self._build_bundle(agent, by_id[v], view, tasks)
            messages = {v: Message(v, dict(a.y), dict(a.z), dict(a.s)) for v, a in agents.items()}
            for v, agent in agents.items():
                for k in sorted(graph[v]):
                    self._receive(agent, messages[k], round_time=float(rounds))
                self._release(agent)
            if all(before[v] == (a.path, a.y, a.z) for v, a in agents.items()):
                break
        self.last_rounds = rounds
        claimed = [t for a in agents.values() for t in a.path]
        self.last_conflicts = len(claimed) - len(set(claimed))
        return {v: list(a.path) for v, a in agents.items()}

    def _build_bundle(self, agent: Agent, b: _Bidder, view: MissionView, tasks: _Tasks) -> None:
        """Add tasks while one beats the best known bid (insertion at the best path position)."""
        vid = b.state.vehicle_id
        excluded = set(view.excluded.get(vid, frozenset()))
        excluded |= {t for v, t in view.current_tiles.items() if t and v != vid}
        while len(agent.bundle) < self.config.auction_bundle_size:
            current = self._bundle_value(b, view, tasks, agent.path)
            best: tuple[float, int, int] | None = None
            for task in range(len(tasks.value)):
                if task in agent.path or not b.effort.feasible[task]:
                    continue
                if tasks.tile_id(task) in excluded:
                    continue
                for position in range(len(agent.path) + 1):
                    candidate = agent.path[:position] + [task] + agent.path[position:]
                    gain = self._bundle_value(b, view, tasks, candidate) - current
                    if not _beats(gain, vid, agent.y.get(task, 0.0), agent.z.get(task)):
                        continue
                    if best is None or gain > best[0] + 1e-12:
                        best = (gain, task, position)
            if best is None:
                return
            gain, task, position = best
            agent.bundle.append(task)
            agent.path.insert(position, task)
            agent.y[task], agent.z[task] = gain, vid

    @staticmethod
    def _receive(agent: Agent, msg: Message, round_time: float) -> None:
        """Merge one neighbour's message into the agent's view of winners and bids."""
        for task in set(msg.z) | set(agent.z):
            action = _decide(agent.vehicle_id, msg.sender, task, agent, msg)
            if action == UPDATE:
                agent.y[task], agent.z[task] = msg.y.get(task, 0.0), msg.z.get(task)
            elif action == RESET:
                agent.y[task], agent.z[task] = 0.0, None
        agent.s[msg.sender] = round_time
        for other, t in msg.s.items():
            if other != agent.vehicle_id:
                agent.s[other] = max(agent.s.get(other, -1.0), t)

    @staticmethod
    def _release(agent: Agent) -> None:
        """Drop the first lost task of the bundle and every task added after it."""
        lost = next(
            (n for n, t in enumerate(agent.bundle) if agent.z.get(t) != agent.vehicle_id), None
        )
        if lost is None:
            return
        for task in agent.bundle[lost + 1 :]:
            if agent.z.get(task) == agent.vehicle_id:
                agent.y[task], agent.z[task] = 0.0, None
        dropped = set(agent.bundle[lost:])
        agent.bundle = agent.bundle[:lost]
        agent.path = [t for t in agent.path if t not in dropped]
