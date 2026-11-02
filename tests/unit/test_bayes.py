import numpy as np
import pytest

from seaswarm.config import AreaConfig, load_default_config
from seaswarm.search.bayes import SuccessTracker, update_detection, update_no_detection
from seaswarm.search.grid import Grid
from seaswarm.search.motion import BeliefMotion
from seaswarm.search.pod import SensorModel
from seaswarm.search.predict import predict
from seaswarm.search.probability_map import ProbabilityMap
from seaswarm.vehicles.sensors import Footprint, footprint
from seaswarm.world.drift import LeewayModel
from seaswarm.world.ocean import UniformCurrent, WindField

GRID = Grid.from_area(AreaConfig(width_m=1000, height_m=1000, cell_size_m=100))
DEFAULTS = load_default_config()


def _look(cells: list[tuple[int, int]], pod: list[float]) -> Footprint:
    rows, cols = zip(*cells, strict=True)
    return Footprint(np.array(rows), np.array(cols), np.array(pod))


def test_two_cell_hand_example() -> None:
    p = np.zeros(GRID.shape)
    p[0, 0], p[0, 1] = 0.5, 0.3  # p_out = 0.2
    pmap = ProbabilityMap(GRID, p, 0.2)
    pos = update_no_detection(pmap, _look([(0, 0), (0, 1)], [0.5, 0.0]))
    # POS = 0.5 * 0.5 = 0.25; denominator 0.75
    assert pos == pytest.approx(0.25)
    assert pmap.p[0, 0] == pytest.approx(0.25 / 0.75)
    assert pmap.p[0, 1] == pytest.approx(0.3 / 0.75)
    assert pmap.p_out == pytest.approx(0.2 / 0.75)
    assert pmap.total() == pytest.approx(1.0, abs=1e-15)


def test_cell_searched_with_pod_one_is_emptied() -> None:
    pmap = ProbabilityMap(GRID, np.full(GRID.shape, 0.01))
    update_no_detection(pmap, _look([(3, 4)], [1.0]))
    assert pmap.p[3, 4] == 0.0
    assert pmap.p[0, 0] == pytest.approx(0.01 / 0.99)


def test_certain_detection_cannot_fail() -> None:
    p = np.zeros(GRID.shape)
    p[2, 2] = 1.0
    with pytest.raises(ValueError, match="POS = 1"):
        update_no_detection(ProbabilityMap(GRID, p), _look([(2, 2)], [1.0]))


def test_detection_replaces_the_belief() -> None:
    pmap = ProbabilityMap(GRID, np.full(GRID.shape, 0.01))
    update_detection(pmap, (120.0, -80.0), sigma_m=20.0)
    row, col = GRID.cell_of(np.array([[120.0, -80.0]]))
    assert pmap.p.argmax() == np.ravel_multi_index((row[0], col[0]), GRID.shape)
    np.testing.assert_allclose(pmap.mean_m(), [120.0, -80.0], atol=GRID.cell_size_m / 2)
    assert pmap.total() == pytest.approx(1.0, abs=1e-12)


def test_cumulative_pos() -> None:
    tracker = SuccessTracker()
    for pos in (0.1, 0.2, 0.5):
        tracker.add(pos)
    assert tracker.pos_cum == pytest.approx(1.0 - 0.9 * 0.8 * 0.5)


def test_total_stays_one_after_1000_random_predicts_and_updates() -> None:
    rng = np.random.default_rng(0)
    big = Grid.from_area(AreaConfig(width_m=10000, height_m=10000, cell_size_m=100))
    pmap = ProbabilityMap.from_gaussian(big, (0.0, 0.0), 400.0)
    leeway = LeewayModel.from_config(DEFAULTS.leeway_objects["PIW-1"], 1800.0)
    sensors = [
        SensorModel.from_config(DEFAULTS.sensors["drone_eo"], True, 60.0),
        SensorModel.from_config(DEFAULTS.sensors["usv_mast"], True),
    ]
    t_s = 0.0
    for i in range(1000):
        if i % 2 == 0:
            u, v = rng.normal(0.0, 0.4, 2)
            motion = BeliefMotion(leeway, UniformCurrent(u, v), WindField(rng.uniform(0, 15), 0.0))
            predict(pmap, motion, t_s, 60.0)
            t_s += 60.0
        else:
            x, y = pmap.mean_m() + rng.normal(0.0, 300.0, 2)
            update_no_detection(pmap, footprint(big, sensors[i % 4 // 2], x, y, points_per_r_max=8))
        assert abs(pmap.total() - 1.0) < 1e-9
    assert np.all(pmap.p >= 0.0)
