from pathlib import Path

import numpy as np
import pytest

from seaswarm.config import load_default_config, load_scenario
from seaswarm.planning.auction import AuctionPlanner
from seaswarm.planning.base import MissionView, OperatorPrefs, VehicleCapability
from seaswarm.planning.cbba import RESET, UPDATE, Agent, CBBAPlanner, Message, _decide, comm_graph
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
IDS = ("drone_1", "drone_2", "drone_3", "usv_1")


def _cap(vid: str) -> VehicleCapability:
    usv = vid.startswith("usv")
    cfg = DEFAULTS.vehicles.usv if usv else DEFAULTS.vehicles.drone
    sensor = DEFAULTS.sensors["usv_mast" if usv else "drone_eo"]
    vtype = VehicleType.USV if usv else VehicleType.DRONE
    return VehicleCapability.build(
        vid, vtype, cfg, sensor, True, 0.0 if usv else 60.0, DEFAULTS.planner
    )


CAPS = {v: _cap(v) for v in IDS}
VIEW = MissionView(900.0, 900.0, BASE, FENCE, CAPS, {}, (0.3, 0.1))


def _fleet(spread_m: float = 0.0) -> list[VehicleState]:
    return [
        VehicleState(
            v,
            CAPS[v].vehicle_type,
            i * spread_m,
            -1000.0,
            0.0,
            0.0,
            60.0,
            1.0,
            VehicleStatus.IDLE,
            0,
        )
        for i, v in enumerate(IDS)
    ]


def _belief() -> ProbabilityMap:
    a = ProbabilityMap.from_gaussian(GRID, (-1500.0, 0.0), 500.0)
    b = ProbabilityMap.from_gaussian(GRID, (1500.0, 1000.0), 500.0)
    return ProbabilityMap(GRID, 0.5 * a.p + 0.5 * b.p, 0.5 * a.p_out + 0.5 * b.p_out)


def _cbba(range_m: float = np.inf) -> CBBAPlanner:
    return CBBAPlanner(TILES, DEFAULTS.planner, np.random.default_rng(0), range_m)


def test_comm_graph_links_vehicles_within_range() -> None:
    graph = comm_graph({"a": (0.0, 0.0), "b": (900.0, 0.0), "c": (2500.0, 0.0)}, 1000.0)
    assert graph == {"a": {"b"}, "b": {"a"}, "c": set()}


def test_decision_table_spot_cases() -> None:
    me = Agent("i", y={0: 0.5}, z={0: "i"})
    higher = Message("k", {0: 0.7}, {0: "k"}, {})
    lower = Message("k", {0: 0.3}, {0: "k"}, {})
    assert _decide("i", "k", 0, me, higher) == UPDATE  # sender outbids me
    assert _decide("i", "k", 0, me, lower) != UPDATE  # I keep it
    me_thinks_k = Agent("i", y={0: 0.6}, z={0: "k"})
    k_says_i = Message("k", {0: 0.6}, {0: "i"}, {})
    assert _decide("i", "k", 0, me_thinks_k, k_says_i) == RESET  # we disagree about each other


def test_connected_cbba_is_conflict_free_and_as_good_as_the_central_auction() -> None:
    cbba, auction = _cbba(), AuctionPlanner(TILES, DEFAULTS.planner, np.random.default_rng(0))
    plan_c = cbba.plan(VIEW, _belief(), _fleet(), OperatorPrefs())
    plan_a = auction.plan(VIEW, _belief(), _fleet(), OperatorPrefs())
    assert cbba.last_conflicts == 0
    tiles = [t.tile_id for a in plan_c.values() for t in a.tasks]
    assert len(tiles) == len(set(tiles)) and len(plan_c) == len(IDS)
    gain = lambda plan: sum(t.expected_pos_gain for a in plan.values() for t in a.tasks)  # noqa: E731
    assert gain(plan_c) >= 0.9 * gain(plan_a)
    assert 1 <= cbba.last_rounds < cbba.max_rounds


def test_a_split_network_can_claim_the_same_tile_twice() -> None:
    cbba = _cbba(range_m=10.0)  # nobody hears anybody
    cbba.plan(VIEW, _belief(), _fleet(spread_m=500.0), OperatorPrefs())
    assert cbba.last_conflicts > 0


def test_cbba_is_deterministic() -> None:
    a = _cbba().plan(VIEW, _belief(), _fleet(), OperatorPrefs())
    b = _cbba().plan(VIEW, _belief(), _fleet(), OperatorPrefs())
    assert a == b


@pytest.mark.parametrize("strategy", ["cbba", "cbba@1000", "cbba@inf"])
def test_cbba_missions_run_safely(strategy: str) -> None:
    from seaswarm.mission import SearchMission

    raw = SCENARIO.model_dump()
    raw["limits"]["max_sim_time_s"] = 1800.0
    mission = SearchMission(SCENARIO.model_validate(raw), DEFAULTS, strategy)
    mission.run()
    assert mission.guard.violations.total == 0
