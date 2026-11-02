import time

import numpy as np
import pytest

from seaswarm.config import AreaConfig, load_default_config
from seaswarm.search.bayes import update_no_detection
from seaswarm.search.grid import Grid
from seaswarm.search.motion import BeliefMotion
from seaswarm.search.pod import SensorModel
from seaswarm.search.predict import predict
from seaswarm.search.probability_map import ProbabilityMap
from seaswarm.vehicles.sensors import footprint
from seaswarm.world.drift import LeewayModel
from seaswarm.world.ocean import UniformCurrent, WindField


@pytest.mark.bench
def test_predict_and_update_on_200x200_under_20_ms() -> None:
    defaults = load_default_config()
    grid = Grid.from_area(AreaConfig(width_m=20000, height_m=20000, cell_size_m=100))
    pmap = ProbabilityMap.from_gaussian(grid, (0.0, 0.0), 1500.0)
    leeway = LeewayModel.from_config(defaults.leeway_objects["PIW-1"], 1800.0)
    motion = BeliefMotion(leeway, UniformCurrent(0.3, 0.1), WindField(10.0, 250.0))
    usv = SensorModel.from_config(defaults.sensors["usv_mast"], daylight=True)
    durations = []
    for i in range(50):
        start = time.perf_counter()
        predict(pmap, motion, 600.0 + 60.0 * i, 60.0)
        update_no_detection(pmap, footprint(grid, usv, 100.0 * i, 0.0, points_per_r_max=8))
        durations.append(time.perf_counter() - start)
    median_ms = 1e3 * float(np.median(durations))
    print(f"predict + update on 200 x 200: median {median_ms:.2f} ms")
    assert median_ms < 20.0
