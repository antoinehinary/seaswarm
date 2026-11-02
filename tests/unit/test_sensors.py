import numpy as np
import pytest

from seaswarm.config import AreaConfig, load_default_config
from seaswarm.search.grid import Grid
from seaswarm.search.pod import SensorModel
from seaswarm.vehicles.sensors import footprint, roll_detection

GRID = Grid.from_area(AreaConfig(width_m=10000, height_m=10000, cell_size_m=100))
CELL_AREA = GRID.cell_size_m**2
SENSORS = load_default_config().sensors
EO = SensorModel.from_config(SENSORS["drone_eo"], daylight=True, altitude_m=60.0)
THERMAL = SensorModel.from_config(SENSORS["drone_thermal"], daylight=True, altitude_m=60.0)
USV = SensorModel.from_config(SENSORS["usv_mast"], daylight=True)


@pytest.mark.parametrize(
    "sensor", [EO, THERMAL, USV], ids=["drone_eo", "drone_thermal", "usv_mast"]
)
@pytest.mark.parametrize("position", [(0.0, 0.0), (50.0, 50.0), (23.0, -71.0)])
def test_footprint_integrates_to_the_coverage_area(
    sensor: SensorModel, position: tuple[float, float]
) -> None:
    look = footprint(GRID, sensor, *position, points_per_r_max=8)
    assert look.pod.sum() * CELL_AREA == pytest.approx(sensor.coverage_area_m2(), rel=0.02)


def test_small_footprint_between_cell_centres_still_counts() -> None:
    # 42 m camera exactly between four cell centres: centre sampling would give POD 0 everywhere
    assert EO.pod(np.hypot(50.0, 50.0)) == 0.0
    look = footprint(GRID, EO, 0.0, 0.0, points_per_r_max=8)
    assert len(look.pod) == 4
    np.testing.assert_allclose(look.pod, look.pod[0])


def test_footprint_is_clipped_at_the_grid_edge() -> None:
    look = footprint(GRID, USV, 4990.0, 0.0, points_per_r_max=8)
    assert look.cols.max() == GRID.nx - 1
    assert footprint(GRID, USV, 9000.0, 0.0, points_per_r_max=8).pod.size == 0


def test_detection_roll_frequency_matches_pod() -> None:
    rng = np.random.default_rng(0)
    hits = [roll_detection(USV, np.zeros(2), np.array([70.0, 0.0]), rng) for _ in range(20000)]
    assert np.mean(hits) == pytest.approx(0.5 * np.exp(-0.5), abs=0.01)
