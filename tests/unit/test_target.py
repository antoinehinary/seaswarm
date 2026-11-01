from pathlib import Path

import numpy as np
import pytest

from seaswarm.config import load_default_config, load_scenario
from seaswarm.rng import make_rngs
from seaswarm.world.drift import ParticleDrift, sample_around_lkp, simulate
from seaswarm.world.target import PersonInWater

SCENARIO = load_scenario(Path(__file__).resolve().parents[2] / "scenarios" / "calm_single.yaml")
DEFAULTS = load_default_config()


def _person(seed: int) -> PersonInWater:
    return PersonInWater.from_scenario(SCENARIO, DEFAULTS, make_rngs(seed))


def test_person_starts_near_the_lkp_at_the_fall_time() -> None:
    person = _person(0)
    assert person.t_s == 0.0
    lkp = np.array([SCENARIO.lkp.x_m, SCENARIO.lkp.y_m])
    assert np.linalg.norm(person.position_m - lkp) < 5 * SCENARIO.lkp.sigma_m


def test_person_matches_a_one_particle_drift_with_the_same_streams() -> None:
    person = _person(3)
    rngs = make_rngs(3)
    start = sample_around_lkp(SCENARIO.lkp, 1, rngs.initial_position)
    reference = ParticleDrift.from_scenario(SCENARIO, DEFAULTS, start, rngs.drift)
    for _ in range(600):
        person.step(1.0)
    simulate(reference, 600.0, 1.0, 600.0, max_steps=600)
    np.testing.assert_array_equal(person.position_m, reference.positions_m[0])
    assert person.t_s == 600.0


def test_position_is_a_copy() -> None:
    person = _person(0)
    person.position_m[0] = 1e9
    assert person.position_m[0] != 1e9


def test_person_must_be_a_single_particle() -> None:
    drift = ParticleDrift.from_scenario(SCENARIO, DEFAULTS, np.zeros((2, 2)), make_rngs(0).drift)
    with pytest.raises(ValueError, match="one particle"):
        PersonInWater(drift)


def test_lkp_samples_follow_the_prior() -> None:
    samples = sample_around_lkp(SCENARIO.lkp, 20000, np.random.default_rng(0))
    np.testing.assert_allclose(samples.mean(axis=0), [SCENARIO.lkp.x_m, SCENARIO.lkp.y_m], atol=5)
    np.testing.assert_allclose(samples.std(axis=0), SCENARIO.lkp.sigma_m, rtol=0.02)
