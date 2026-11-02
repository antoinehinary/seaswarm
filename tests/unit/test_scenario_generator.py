import numpy as np

from seaswarm.config import load_default_config
from seaswarm.eval.scenario import CURRENT_MPS, WIND_MPS, generate_scenario
from seaswarm.search.motion import BeliefMotion

DEFAULTS = load_default_config()
SCENARIOS = [generate_scenario(i, 0, DEFAULTS) for i in range(60)]


def test_generation_is_deterministic_and_seeded_by_index() -> None:
    assert generate_scenario(7, 0, DEFAULTS) == SCENARIOS[7]
    assert generate_scenario(7, 1, DEFAULTS) != SCENARIOS[7]
    assert [s.id for s in SCENARIOS[:2]] == ["gen_000", "gen_001"]
    assert [s.seed for s in SCENARIOS[:2]] == [0, 1]


def test_parameters_cover_the_requested_ranges() -> None:
    winds = [s.wind.speed_mps for s in SCENARIOS]
    currents = [s.current.speed_mps for s in SCENARIOS]
    assert WIND_MPS[0] <= min(winds) and max(winds) <= WIND_MPS[1]
    assert CURRENT_MPS[0] <= min(currents) and max(currents) <= CURRENT_MPS[1]
    assert 0.1 < np.mean([not s.daylight for s in SCENARIOS]) < 0.5


def test_expected_drift_stays_in_the_area_at_mid_mission() -> None:
    for s in SCENARIOS:
        motion = BeliefMotion.from_scenario(s, DEFAULTS)
        mid = np.array([s.lkp.x_m, s.lkp.y_m]) + motion.mean_velocity_mps(0.0, np.zeros(2)) * 5400.0
        assert np.all(np.abs(mid) < 5000.0), s.id
