"""Run logs: `log.npz` (sampled states and belief snapshots) and `events.jsonl`.

The truth track is recorded for replays (`--show-truth`) and evaluation only; nothing in the
mission reads it back.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from seaswarm.mission import SearchMission
from seaswarm.vehicles.backend import VehicleStatus

STATUS_CODES = {status: i for i, status in enumerate(VehicleStatus)}


class RunRecorder:
    """Samples a mission while it runs; `write` saves the run directory."""

    def __init__(
        self, mission: SearchMission, state_every_s: float = 5.0, belief_every_s: float = 60.0
    ):
        """Record `mission` every `state_every_s` (vehicles, truth) and `belief_every_s` (map)."""
        self.mission = mission
        self.state_every_s = state_every_s
        self.belief_every_s = belief_every_s
        self.vehicle_ids = mission.backend.vehicle_ids()
        self._t: list[float] = []
        self._xy: list[list[tuple[float, float]]] = []
        self._yaw: list[list[float]] = []
        self._status: list[list[int]] = []
        self._energy: list[list[float]] = []
        self._truth: list[np.ndarray] = []
        self._pos_cum: list[float] = []
        self._belief_t: list[float] = []
        self._belief: list[np.ndarray] = []
        self._next_state_s = mission.t_s
        self._next_belief_s = mission.t_s
        self.sample()

    def sample(self) -> None:
        """Record whatever is due at the mission's current time."""
        m = self.mission
        if m.t_s >= self._next_state_s - 1e-9:
            states = m.states()
            self._t.append(m.t_s)
            self._xy.append([(s.x_m, s.y_m) for s in states])
            self._yaw.append([s.yaw_rad for s in states])
            self._status.append([STATUS_CODES[s.status] for s in states])
            self._energy.append([s.energy_frac for s in states])
            self._truth.append(m.backend.truth_position_m())
            self._pos_cum.append(m.success.pos_cum)
            self._next_state_s += self.state_every_s
        if m.t_s >= self._next_belief_s - 1e-9:
            self._belief_t.append(m.t_s)
            self._belief.append(m.belief.p.astype(np.float32))
            self._next_belief_s += self.belief_every_s

    def write(self, run_dir: Path, meta: dict) -> None:
        """Write `log.npz` and `events.jsonl` into `run_dir`."""
        m = self.mission
        run_dir.mkdir(parents=True, exist_ok=True)
        grid = m.belief.grid
        fence = m.guard.fence
        detections = np.array([(d.t_s, d.x_m, d.y_m) for d in m.detections]).reshape(-1, 3)
        np.savez_compressed(
            run_dir / "log.npz",
            t_s=np.array(self._t),
            vehicle_ids=np.array(self.vehicle_ids),
            vehicle_types=np.array(
                [m.capabilities[v].vehicle_type.value for v in self.vehicle_ids]
            ),
            xy_m=np.array(self._xy),
            yaw_rad=np.array(self._yaw),
            status=np.array(self._status),
            status_names=np.array([s.value for s in VehicleStatus]),
            energy_frac=np.array(self._energy),
            truth_m=np.array(self._truth),
            pos_cum=np.array(self._pos_cum),
            belief_t_s=np.array(self._belief_t),
            belief=np.array(self._belief),
            grid_extent_m=np.array(
                [grid.x_min_m, grid.x_edges_m[-1], grid.y_min_m, grid.y_edges_m[-1]]
            ),
            fence_m=np.array([fence.x_min_m, fence.x_max_m, fence.y_min_m, fence.y_max_m]),
            base_m=np.array(m.base_xy_m),
            r_max_m=np.array([m.sensors[v].r_max_m for v in self.vehicle_ids]),
            detections=detections,
            meta=np.array(json.dumps(meta)),
        )
        with (run_dir / "events.jsonl").open("w") as f:
            for event in m.events:
                f.write(json.dumps({"t_s": event.t_s, "kind": event.kind, **event.data}) + "\n")
            for decision in m.guard.decisions:
                f.write(json.dumps({"kind": "safety_decision", **asdict(decision)}) + "\n")
