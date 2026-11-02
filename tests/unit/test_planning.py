from pathlib import Path

import numpy as np
import pytest

from seaswarm.config import load_default_config, load_scenario
from seaswarm.planning.base import MissionView, OperatorPrefs, PriorityArea, VehicleCapability
from seaswarm.planning.greedy import GreedyPlanner
from seaswarm.planning.lawnmower import LawnmowerPlanner, mass_rectangle
from seaswarm.planning.tiles import Rect, TileGrid, lawnmower, lawnmower_length_m
from seaswarm.safety.rules import Geofence, path_length_m
from seaswarm.search.grid import Grid
from seaswarm.search.pod import SensorModel, random_search_pod, sweep_width_m
from seaswarm.search.probability_map import ProbabilityMap
from seaswarm.vehicles.backend import VehicleState, VehicleStatus, VehicleType

DEFAULTS = load_default_config()
SCENARIO = load_scenario(Path(__file__).resolve().parents[2] / "scenarios" / "windy_mixed.yaml")
GRID = Grid.from_area(SCENARIO.area)
TILES = TileGrid.from_grid(GRID, DEFAULTS.planner.tile_size_m)
BASE = (SCENARIO.base.x_m, SCENARIO.base.y_m)
FENCE = Geofence.around(SCENARIO.area, SCENARIO.base, 1000.0)


def _cap(vid: str, vtype: VehicleType, sensor: str, altitude: float) -> VehicleCapability:
    cfg = getattr(DEFAULTS.vehicles, vtype.value)
    return VehicleCapability.build(
        vid, vtype, cfg, DEFAULTS.sensors[sensor], True, altitude, DEFAULTS.planner
    )


CAPS = {
    "drone_1": _cap("drone_1", VehicleType.DRONE, "drone_eo", 60.0),
    "drone_2": _cap("drone_2", VehicleType.DRONE, "drone_eo", 80.0),
    "usv_1": _cap("usv_1", VehicleType.USV, "usv_mast", 0.0),
}


def _state(vid: str, x: float, y: float, energy: float = 1.0, status=VehicleStatus.IDLE, left=0):
    vtype = VehicleType.USV if vid.startswith("usv") else VehicleType.DRONE
    return VehicleState(vid, vtype, x, y, 0.0, 0.0, 60.0, energy, status, left)


def _view(t_s: float = 900.0) -> MissionView:
    return MissionView(t_s, 900.0, BASE, FENCE, CAPS, {}, (0.3, 0.1))


def _blob(x: float, y: float, sigma: float = 400.0) -> ProbabilityMap:
    return ProbabilityMap.from_gaussian(GRID, (x, y), sigma)


# --------------------------------------------------------------------------- search theory


def test_sweep_width_of_a_perfect_sensor_is_its_diameter() -> None:
    perfect = SensorModel(pod_max=1.0, sigma_r_m=1e9, r_max_m=50.0, period_s=1.0)
    assert sweep_width_m(perfect, speed_mps=5.0) == pytest.approx(100.0, rel=0.01)


def test_more_looks_widen_the_sweep() -> None:
    eo = SensorModel.from_config(DEFAULTS.sensors["drone_eo"], True, 60.0)
    assert sweep_width_m(eo, 5.0) > sweep_width_m(eo, 15.0)


def test_random_search_formula() -> None:
    assert random_search_pod(50.0, 1000.0, 50_000.0) == pytest.approx(1 - np.exp(-1.0))


# --------------------------------------------------------------------------- tiles


def test_tile_values_sum_to_the_in_grid_mass() -> None:
    pmap = _blob(1000.0, 500.0)
    values = TILES.values(pmap)
    assert values.shape == (25, 25)
    assert values.sum() == pytest.approx(pmap.p.sum())


def test_lawnmower_covers_the_rectangle_from_the_nearest_corner() -> None:
    rect = Rect(0.0, 0.0, 400.0, 400.0)
    path = lawnmower(rect, 100.0, (500.0, 500.0))  # start near the north-east corner
    assert path[0] == (400.0, 350.0)
    ys = sorted({y for _, y in path})
    assert ys == [50.0, 150.0, 250.0, 350.0]
    assert path_length_m(path[0], path[1:]) == pytest.approx(lawnmower_length_m(rect, 100.0))


def test_north_south_lawnmower_is_the_transpose() -> None:
    path = lawnmower(Rect(0.0, 0.0, 200.0, 600.0), 100.0, (0.0, 0.0), north_south=True)
    assert sorted({x for x, _ in path}) == [50.0, 150.0]
    assert path[0] == (50.0, 0.0) and path[1] == (50.0, 600.0)


# --------------------------------------------------------------------------- greedy


def test_greedy_sends_vehicles_to_distinct_high_value_tiles() -> None:
    vehicles = [_state("drone_1", 0.0, 0.0), _state("drone_2", 0.0, 0.0), _state("usv_1", 0.0, 0.0)]
    plan = GreedyPlanner(TILES).plan(_view(), _blob(1000.0, 500.0), vehicles, OperatorPrefs())
    tiles = [a.tasks[0].tile_id for a in plan.values()]
    assert len(tiles) == 3 and len(set(tiles)) == 3
    for assignment in plan.values():
        x, y = assignment.tasks[0].waypoints[0]
        assert np.hypot(x - 1000.0, y - 500.0) < 1200.0


def test_greedy_is_deterministic() -> None:
    vehicles = [_state("usv_1", 0.0, 0.0), _state("drone_1", 0.0, 0.0)]
    a = GreedyPlanner(TILES).plan(_view(), _blob(0.0, 0.0), vehicles, OperatorPrefs())
    b = GreedyPlanner(TILES).plan(_view(), _blob(0.0, 0.0), vehicles[::-1], OperatorPrefs())
    assert a == b


def test_greedy_skips_tiles_beyond_the_energy_reserve() -> None:
    # nearly empty drone: only tiles close to base are feasible, so it takes nothing far away
    low = _state("drone_1", BASE[0], BASE[1] + 500.0, energy=0.12)
    plan = GreedyPlanner(TILES).plan(_view(), _blob(3000.0, 3000.0), [low], OperatorPrefs())
    if "drone_1" in plan:
        x, y = plan["drone_1"].tasks[0].waypoints[0]
        assert np.hypot(x - BASE[0], y - BASE[1]) < 2500.0


def test_priority_area_changes_the_choice_not_the_belief() -> None:
    pmap = ProbabilityMap(GRID, np.full(GRID.shape, 1.0 / GRID.nx / GRID.ny))
    before = pmap.p.copy()
    prefs = OperatorPrefs((PriorityArea("a1", 2000.0, 2000.0, 4000.0, 4000.0, 5.0, 1e9),))
    vehicle = [_state("usv_1", 0.0, 0.0)]
    plan = GreedyPlanner(TILES).plan(_view(), pmap, vehicle, prefs)
    _, row, col = plan["usv_1"].tasks[0].tile_id.split("_")
    cx, cy = TILES.centres_m()
    x, y = cx[int(row), int(col)], cy[int(row), int(col)]
    assert 2000.0 <= x <= 4000.0 and 2000.0 <= y <= 4000.0  # weights apply at tile centres
    np.testing.assert_array_equal(pmap.p, before)


def test_unavailable_vehicles_get_nothing() -> None:
    rtb = _state("drone_1", 0.0, 0.0, status=VehicleStatus.RTB)
    assert GreedyPlanner(TILES).plan(_view(), _blob(0.0, 0.0), [rtb], OperatorPrefs()) == {}


# --------------------------------------------------------------------------- lawnmower


def test_mass_rectangle_holds_the_requested_mass() -> None:
    pmap = _blob(0.0, 0.0, sigma=500.0)
    rect = mass_rectangle(pmap, 0.95)
    assert rect.x_min_m < -800.0 and rect.x_max_m > 800.0
    assert rect.width_m < 3000.0


def test_lawnmower_strips_are_proportional_to_sweep_rate_and_resume() -> None:
    planner = LawnmowerPlanner(prior_mass=0.95)
    vehicles = [_state(v, *BASE) for v in ("drone_1", "drone_2", "usv_1")]
    plan = planner.plan(_view(900.0), _blob(0.0, 0.0), vehicles, OperatorPrefs())
    widths = {}
    for vid, a in plan.items():
        xs = [x for x, _ in a.tasks[0].waypoints]
        widths[vid] = max(xs) - min(xs) + CAPS[vid].track_spacing_m
    rates = {v: CAPS[v].sweep_rate_m2_s for v in CAPS}
    assert widths["usv_1"] / widths["drone_1"] == pytest.approx(
        rates["usv_1"] / rates["drone_1"], rel=0.25
    )
    # drone_1 flew 10 waypoints of its pattern, then went home and came back
    n = len(plan["drone_1"].tasks[0].waypoints)
    flying = _state("drone_1", 0.0, 0.0, status=VehicleStatus.TRANSIT, left=n - 10)
    planner.plan(_view(1000.0), _blob(0.0, 0.0), [flying], OperatorPrefs())
    back = planner.plan(_view(2000.0), _blob(0.0, 0.0), [_state("drone_1", *BASE)], OperatorPrefs())
    assert len(back["drone_1"].tasks[0].waypoints) == n - 10
    # resumed waypoints are shifted by the forecast drift since the start
    dx = 0.3 * (2000.0 - 900.0)
    assert back["drone_1"].tasks[0].waypoints[0][0] == pytest.approx(
        plan["drone_1"].tasks[0].waypoints[10][0] + dx
    )
