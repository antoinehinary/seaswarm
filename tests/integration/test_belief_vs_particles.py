"""The probability map must match a Monte Carlo cloud of the truth drift model.

Sampling noise sets the bar: at 100 m cells two independent 5000-particle clouds of the same
distribution are ~0.25-0.3 apart in Hellinger distance, so with 5000 particles the 0.15 target is
checked at the planner's 400 m tile resolution, and at 100 m cells the map must be closer to the
cloud than a second cloud is. The 100 000-particle check at 100 m cells is a `bench` test.
"""

from pathlib import Path

import numpy as np
import pytest

from seaswarm.config import load_default_config, load_scenario
from seaswarm.rng import make_rngs
from seaswarm.search.predict import belief_at
from seaswarm.search.probability_map import ProbabilityMap, hellinger
from seaswarm.world.drift import drift_cloud

SCENARIOS = Path(__file__).resolve().parents[2] / "scenarios"
DEFAULTS = load_default_config()
T_S = 2 * 3600.0
TILE_FACTOR = 4  # 4 x 4 cells of 100 m = the planner's 400 m tiles


def _cloud_map(name: str, n: int, seed: int) -> ProbabilityMap:
    scenario = load_scenario(SCENARIOS / f"{name}.yaml")
    belief = belief_at(scenario, DEFAULTS, 0.0)  # only for its grid
    xy = drift_cloud(scenario, DEFAULTS, n, make_rngs(seed).particles, T_S)
    return ProbabilityMap.from_samples(belief.grid, xy)


@pytest.mark.parametrize("name", ["calm_single", "windy_mixed"])
def test_belief_matches_5000_particles_after_2_hours(name: str) -> None:
    belief = belief_at(load_scenario(SCENARIOS / f"{name}.yaml"), DEFAULTS, T_S)
    cloud, other_cloud = _cloud_map(name, 5000, seed=0), _cloud_map(name, 5000, seed=1)

    centroid_error_cells = np.linalg.norm(belief.mean_m() - cloud.mean_m()) / 100.0
    assert centroid_error_cells < 1.0
    np.testing.assert_allclose(
        np.sqrt(np.diag(belief.covariance_m2())), np.sqrt(np.diag(cloud.covariance_m2())), rtol=0.05
    )
    assert hellinger(belief.coarsened(TILE_FACTOR), cloud.coarsened(TILE_FACTOR)) < 0.15
    assert hellinger(belief, cloud) < hellinger(other_cloud, cloud)


@pytest.mark.bench
@pytest.mark.parametrize("name", ["calm_single", "windy_mixed"])
def test_belief_matches_100k_particles_at_cell_resolution(name: str) -> None:
    belief = belief_at(load_scenario(SCENARIOS / f"{name}.yaml"), DEFAULTS, T_S)
    distance = hellinger(belief, _cloud_map(name, 100_000, seed=2))
    print(f"{name}: Hellinger(map, 100k particles) at 100 m cells = {distance:.3f}")
    assert distance < 0.15
