"""Operator interface of a search mission: situation reports and direct commands.

This is what the MCP server exposes. Every function returns a JSON-serialisable dict; commands
return {"ok": True, ...} or {"ok": False, "refusal": {"code", "reason", "vehicle_id"}} and never
raise for a refused or impossible request. All quantities are SI; positions are also given as a
compass bearing and distance from the base, the way an operator talks about them. Commands go
through the mission's safety guard like every planner command.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from seaswarm.mission import STRATEGIES, MissionState, SearchMission
from seaswarm.planning.base import PriorityArea
from seaswarm.planning.tiles import Rect, lawnmower
from seaswarm.planning.utility import affordable_prefix
from seaswarm.safety.guard import Proposal
from seaswarm.safety.refusal import Refusal, RefusalCode
from seaswarm.units import yaw_rad_to_bearing_deg
from seaswarm.vehicles.backend import FollowPath, GoTo, ReturnToBase, VehicleType

SECTORS = ("NW", "N", "NE", "W", "C", "E", "SW", "S", "SE")  # 3 x 3 split of the search area
MAX_PRIORITY_DURATION_S = 3600.0
MAX_ADVANCE_S = 600.0

Result = dict[str, Any]


def _refused(code: RefusalCode, reason: str, vehicle_id: str | None = None) -> Result:
    return {
        "ok": False,
        "refusal": {"code": code.value, "reason": reason, "vehicle_id": vehicle_id},
    }


def _refusal(refusal: Refusal) -> Result:
    return _refused(refusal.code, refusal.reason, refusal.vehicle_id)


def _invalid(reason: str) -> Result:
    return {
        "ok": False,
        "refusal": {"code": "INVALID_ARGUMENT", "reason": reason, "vehicle_id": None},
    }


class Operations:
    """Operator-facing view and commands of one `SearchMission`."""

    def __init__(self, mission: SearchMission) -> None:
        """Wrap `mission`."""
        self.m = mission
        self._area_count = 0

    # ------------------------------------------------------------------ helpers

    def _from_base(self, x_m: float, y_m: float) -> dict[str, float]:
        """Bearing (deg) and distance (m) of a point seen from the base."""
        dx, dy = x_m - self.m.base_xy_m[0], y_m - self.m.base_xy_m[1]
        bearing = float(yaw_rad_to_bearing_deg(np.arctan2(dy, dx)))
        return {
            "bearing_from_base_deg": round(bearing, 1),
            "distance_from_base_m": round(float(np.hypot(dx, dy)), 1),
        }

    def _elapsed_s(self) -> float:
        return round(self.m.t_s - self.m.start_s, 1)

    def _task_label(self, vehicle_id: str) -> str | None:
        if vehicle_id in self.m.operator_tasks:
            return self.m.operator_tasks[vehicle_id]
        tile = self.m.guard.tile_of(vehicle_id)
        return f"searching tile {tile}" if tile else None

    def _known_vehicle(self, vehicle_id: str) -> Result | None:
        if vehicle_id in self.m.backend.vehicle_ids():
            return None
        known = ", ".join(self.m.backend.vehicle_ids())
        return _refused(
            RefusalCode.VEHICLE_UNAVAILABLE,
            f"there is no vehicle {vehicle_id!r}; known vehicles: {known}",
            vehicle_id,
        )

    # ------------------------------------------------------------------ reports

    def situation(self) -> Result:
        """Mission state, vehicles, detections, best estimate of the person, recent refusals."""
        mean = self.m.belief.mean_m()
        vehicles = []
        for s in self.m.states():
            vehicles.append(
                {
                    "id": s.vehicle_id,
                    "type": s.vehicle_type.value,
                    "x_m": round(s.x_m, 1),
                    "y_m": round(s.y_m, 1),
                    "heading_deg": round(float(yaw_rad_to_bearing_deg(s.yaw_rad)), 1),
                    "speed_mps": round(s.speed_mps, 2),
                    "energy_pct": round(100.0 * s.energy_frac, 1),
                    "status": s.status.value,
                    "current_task": self._task_label(s.vehicle_id),
                    **self._from_base(s.x_m, s.y_m),
                }
            )
        detections = [
            {
                "id": d.detection_id,
                "vehicle_id": d.vehicle_id,
                "time_s": round(d.t_s - self.m.start_s, 1),
                "x_m": round(d.x_m, 1),
                "y_m": round(d.y_m, 1),
                **self._from_base(d.x_m, d.y_m),
            }
            for d in self.m.detections
        ]
        refusals = [
            {"code": r.code.value, "reason": r.reason, "vehicle_id": r.vehicle_id}
            for r in self.m.refusals[-5:]
        ]
        return {
            "ok": True,
            "sim_time_s": self._elapsed_s(),
            "mission_state": self.m.state.value,
            "strategy": self.m.strategy,
            "base": {"x_m": self.m.base_xy_m[0], "y_m": self.m.base_xy_m[1]},
            "best_estimate": {
                "x_m": round(float(mean[0]), 1),
                "y_m": round(float(mean[1]), 1),
                **self._from_base(*mean),
            },
            "pos_cum": round(self.m.success.pos_cum, 3),
            "vehicles": vehicles,
            "detections": detections,
            "priority_areas": [
                {
                    "id": a.area_id,
                    "rect_m": [a.x_min_m, a.y_min_m, a.x_max_m, a.y_max_m],
                    "weight": a.weight,
                    "expires_in_s": round(a.expires_s - self.m.t_s, 1),
                }
                for a in self.m.prefs.areas
                if a.expires_s > self.m.t_s
            ],
            "last_refusals": refusals,
        }

    def probability_map(self, top_k: int = 10) -> Result:
        """Summary of the belief: most probable cells, spread and mass outside the grid."""
        if not 1 <= top_k <= 50:
            return _invalid(f"top_k must be between 1 and 50, got {top_k}")
        pmap = self.m.belief
        mean = pmap.mean_m()
        return {
            "ok": True,
            "top_cells": [
                {"x_m": x, "y_m": y, "p": round(p, 5), **self._from_base(x, y)}
                for x, y, p in pmap.top_cells(top_k)
            ],
            "p_out": round(pmap.p_out, 4),
            "entropy_bits": round(pmap.entropy_bits(), 2),
            "centroid_x_m": round(float(mean[0]), 1),
            "centroid_y_m": round(float(mean[1]), 1),
            "area_90pct_km2": round(pmap.area_km2(0.9), 2),
            **self._from_base(*mean),
        }

    # ------------------------------------------------------------------ preferences

    def set_priority_area(
        self,
        x_min_m: float,
        y_min_m: float,
        x_max_m: float,
        y_max_m: float,
        weight: float,
        duration_s: float,
    ) -> Result:
        """Weight the planner's utility of tiles inside a rectangle (never the probabilities)."""
        if x_min_m >= x_max_m or y_min_m >= y_max_m:
            return _invalid("the area needs x_min_m < x_max_m and y_min_m < y_max_m")
        if not 0.0 <= weight <= 5.0:
            return _invalid(f"weight must be between 0 and 5, got {weight}")
        if not 0.0 < duration_s <= MAX_PRIORITY_DURATION_S:
            return _invalid(
                f"duration_s must be in (0, {MAX_PRIORITY_DURATION_S:.0f}], got {duration_s}"
            )
        self._area_count += 1
        area_id = f"area_{self._area_count}"
        self.m.set_priority_area(
            PriorityArea(
                area_id, x_min_m, y_min_m, x_max_m, y_max_m, weight, self.m.t_s + duration_s
            )
        )
        return {
            "ok": True,
            "area_id": area_id,
            "rect_m": [x_min_m, y_min_m, x_max_m, y_max_m],
            "weight": weight,
        }

    def sector_rect(self, sector: str) -> Rect:
        """Rectangle of one cell of the 3 x 3 split of the search area (N is top-middle)."""
        area = self.m.scenario.area
        row, col = divmod(SECTORS.index(sector), 3)
        w, h = area.width_m / 3, area.height_m / 3
        x0 = -area.width_m / 2 + col * w
        y1 = area.height_m / 2 - row * h
        return Rect(x0, y1 - h, x0 + w, y1)

    def set_priority_sector(self, sector: str, weight: float, duration_s: float) -> Result:
        """Priority on one ninth of the search area: NW, N, NE, W, C, E, SW, S or SE."""
        if sector not in SECTORS:
            return _invalid(f"sector must be one of {', '.join(SECTORS)}, got {sector!r}")
        r = self.sector_rect(sector)
        result = self.set_priority_area(
            r.x_min_m, r.y_min_m, r.x_max_m, r.y_max_m, weight, duration_s
        )
        return {**result, "sector": sector} if result["ok"] else result

    def clear_priority_areas(self) -> Result:
        """Remove every priority area."""
        return {"ok": True, "cleared": self.m.clear_priority_areas()}

    def set_strategy(self, name: str) -> Result:
        """Switch the planning strategy."""
        if name not in STRATEGIES:
            return _invalid(f"strategy must be one of {', '.join(STRATEGIES)}, got {name!r}")
        self.m.set_strategy(name)
        return {"ok": True, "strategy": name}

    # ------------------------------------------------------------------ direct vehicle commands

    def assign_vehicle(
        self, vehicle_id: str, x_m: float, y_m: float, radius_m: float = 300.0
    ) -> Result:
        """Send a vehicle to search the square of half-side `radius_m` centred on (x_m, y_m)."""
        if (unknown := self._known_vehicle(vehicle_id)) is not None:
            return unknown
        if not 50.0 <= radius_m <= 2000.0:
            return _invalid(f"radius_m must be between 50 and 2000, got {radius_m}")
        state = self.m.guard.get_state(vehicle_id)
        cap = self.m.capabilities[vehicle_id]
        rect = Rect(x_m - radius_m, y_m - radius_m, x_m + radius_m, y_m + radius_m)
        path = affordable_prefix(
            state,
            cap,
            lawnmower(rect, cap.track_spacing_m, (state.x_m, state.y_m)),
            self.m.base_xy_m,
        )
        if not path:
            path = ((x_m, y_m),)  # let the safety layer explain why it cannot go there
        command = FollowPath(vehicle_id, path, cap.cruise_speed_mps)
        refusal = self.m.guard.submit(Proposal(command), self.m.t_s)
        if refusal is not None:
            self.m.refusals.append(refusal)
            return _refusal(refusal)
        label = f"operator: search {2 * radius_m:.0f} m square around ({x_m:.0f}, {y_m:.0f})"
        self.m.operator_command(vehicle_id, label)
        return {"ok": True, "vehicle_id": vehicle_id, "task": label, "waypoints": len(path)}

    def dispatch_to_detection(self, detection_id: str, vehicle_id: str) -> Result:
        """Send a vehicle at full speed to a reported detection (a USV recovers the person)."""
        if (unknown := self._known_vehicle(vehicle_id)) is not None:
            return unknown
        detection = next((d for d in self.m.detections if d.detection_id == detection_id), None)
        if detection is None:
            known = ", ".join(d.detection_id for d in self.m.detections) or "none yet"
            return _invalid(f"there is no detection {detection_id!r}; detections: {known}")
        cap = self.m.capabilities[vehicle_id]
        command = GoTo(vehicle_id, detection.x_m, detection.y_m, cap.config.max_speed_mps)
        refusal = self.m.guard.submit(Proposal(command, dispatch=True), self.m.t_s)
        if refusal is not None:
            self.m.refusals.append(refusal)
            return _refusal(refusal)
        label = f"operator: dispatched to detection {detection_id}"
        self.m.operator_command(vehicle_id, label)
        if cap.vehicle_type is VehicleType.USV:
            self.m.rescuer_id = vehicle_id
        return {
            "ok": True,
            "vehicle_id": vehicle_id,
            "task": label,
            **self._from_base(detection.x_m, detection.y_m),
        }

    def recall_vehicle(self, vehicle_id: str) -> Result:
        """Order a vehicle back to base (always accepted)."""
        if (unknown := self._known_vehicle(vehicle_id)) is not None:
            return unknown
        self.m.guard.submit(Proposal(ReturnToBase(vehicle_id)), self.m.t_s)
        self.m.operator_command(vehicle_id, "operator: recalled to base")
        return {"ok": True, "vehicle_id": vehicle_id, "task": "returning to base"}

    # ------------------------------------------------------------------ time

    def advance(self, duration_s: float) -> Result:
        """Run the simulation for `duration_s` seconds (1-600) and summarise what happened."""
        if not 1.0 <= duration_s <= MAX_ADVANCE_S:
            return _invalid(
                f"duration_s must be between 1 and {MAX_ADVANCE_S:.0f}, got {duration_s}"
            )
        start_s, n_events, n_refusals = self.m.t_s, len(self.m.events), len(self.m.refusals)
        while not self.m.done and self.m.t_s - start_s < duration_s - 1e-9:
            self.m.step()
        new_events = self.m.events[n_events:]
        new_detections = [e.data for e in new_events if e.kind == "detection"]
        return {
            "ok": True,
            "advanced_s": round(self.m.t_s - start_s, 1),
            "sim_time_s": self._elapsed_s(),
            "mission_state": self.m.state.value,
            "ended": self.m.done,
            "new_detections": [
                {
                    "id": d["detection_id"],
                    "vehicle_id": d["vehicle_id"],
                    **self._from_base(d["x_m"], d["y_m"]),
                }
                for d in new_detections
                if "detection_id" in d
            ],
            "events": dict(Counter(e.kind for e in new_events if e.kind != "detection")),
            "new_refusals": [
                {"code": r.code.value, "reason": r.reason, "vehicle_id": r.vehicle_id}
                for r in self.m.refusals[n_refusals:][-5:]
            ],
            "pos_cum": round(self.m.success.pos_cum, 3),
        }

    def mission_over(self) -> bool:
        """True once the mission has ended."""
        return self.m.done or self.m.state is MissionState.TIMEOUT
