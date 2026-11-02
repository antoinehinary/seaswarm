from pathlib import Path

import numpy as np

from seaswarm.config import load_default_config, load_scenario
from seaswarm.planning.auction import AuctionPlanner
from seaswarm.planning.base import MissionView, OperatorPrefs, VehicleCapability
from seaswarm.planning.tiles import TileGrid
from seaswarm.safety.rules import Geofence
from seaswarm.search.grid import Grid
from seaswarm.search.probability_map import ProbabilityMap
from seaswarm.vehicles.backend import VehicleState, VehicleStatus, VehicleType

DEFAULTS = load_default_config()
SCENARIO = load_scenario(Path(__file__).resolve().parents[2] / "scenarios" / "windy_mixed.yaml")
GRID = Grid.from_area(SCENARIO.area)
TILES = TileGrid.from_grid(GRID, DEFAULTS.planner.tile_size_m)
BASE = (SCENARIO.base.x_m, SCENARIO.base.y_m)
FENCE = Geofence.around(SCENARIO.area, SCENARIO.base, 1000.0)


def _cap(vid: str, altitude: float = 60.0) -> VehicleCapability:
    usv = vid.startswith("usv")
    vtype = VehicleType.USV if usv else VehicleType.DRONE
    cfg = DEFAULTS.vehicles.usv if usv else DEFAULTS.vehicles.drone
    sensor = DEFAULTS.sensors["usv_mast" if usv else "drone_eo"]
    return VehicleCapability.build(vid, vtype, cfg, sensor, True, altitude, DEFAULTS.planner)


IDS = ("drone_1", "drone_2", "drone_3", "usv_1")
CAPS = {v: _cap(v, 0.0 if v.startswith("usv") else 60.0) for v in IDS}


def _state(vid: str, x: float, y: float, energy: float = 1.0) -> VehicleState:
    vtype = VehicleType.USV if vid.startswith("usv") else VehicleType.DRONE
    return VehicleState(vid, vtype, x, y, 0.0, 0.0, 60.0, energy, VehicleStatus.IDLE, 0)


def _view(current_tiles: dict | None = None, excluded: dict | None = None) -> MissionView:
    return MissionView(
        900.0, 900.0, BASE, FENCE, CAPS, current_tiles or {}, (0.3, 0.1), (), excluded or {}
    )


def _planner(seed: int = 0) -> AuctionPlanner:
    return AuctionPlanner(TILES, DEFAULTS.planner, np.random.default_rng(seed))


def _two_blobs() -> ProbabilityMap:
    a = ProbabilityMap.from_gaussian(GRID, (-1500.0, 0.0), 500.0)
    b = ProbabilityMap.from_gaussian(GRID, (1500.0, 1000.0), 500.0)
    return ProbabilityMap(GRID, 0.5 * a.p + 0.5 * b.p, 0.5 * a.p_out + 0.5 * b.p_out)


def _fleet() -> list[VehicleState]:
    return [_state(v, 0.0, -1000.0) for v in IDS]


def test_no_tile_is_awarded_twice() -> None:
    planner = _planner()
    plan = planner.plan(_view(), _two_blobs(), _fleet(), OperatorPrefs())
    awarded = [a.task for a in planner.last_awards]
    assert len(awarded) == len(set(awarded)) > len(IDS)
    planned_tiles = [t.tile_id for a in plan.values() for t in a.tasks]
    assert len(planned_tiles) == len(set(planned_tiles))
    assert all(len(a.tasks) <= DEFAULTS.planner.auction_bundle_size for a in plan.values())


def test_bundles_respect_the_energy_reserve() -> None:
    low = [_state("drone_1", BASE[0], BASE[1] + 300.0, energy=0.25)]
    plan = _planner().plan(_view(), _two_blobs(), low, OperatorPrefs())
    # 25 % of 1500 s leaves no room to fly ~6 km out, search and come back
    assert plan == {}


def test_bundle_tasks_are_in_execution_order() -> None:
    plan = _planner().plan(_view(), _two_blobs(), _fleet(), OperatorPrefs())
    for assignment in plan.values():
        etas = [t.eta_s for t in assignment.tasks]
        assert etas == sorted(etas)
        assert assignment.tasks[0].waypoints  # only the first task is executed now


def test_auction_is_deterministic_for_a_seed() -> None:
    a = _planner(seed=4).plan(_view(), _two_blobs(), _fleet(), OperatorPrefs())
    b = _planner(seed=4).plan(_view(), _two_blobs(), _fleet(), OperatorPrefs())
    assert a == b


def test_tiles_held_or_refused_are_not_offered() -> None:
    plan = _planner().plan(_view(), _two_blobs(), _fleet(), OperatorPrefs())
    best_tile = plan["drone_1"].tasks[0].tile_id
    held = _planner().plan(
        _view(current_tiles={"usv_1": best_tile}), _two_blobs(), _fleet(), OperatorPrefs()
    )
    refused = _planner().plan(
        _view(excluded={"drone_1": frozenset({best_tile})}), _two_blobs(), _fleet(), OperatorPrefs()
    )
    for other in (held, refused):
        assert "drone_1" not in other or best_tile not in [
            t.tile_id for t in other["drone_1"].tasks
        ]
