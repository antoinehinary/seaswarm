from pathlib import Path

import numpy as np
import pytest

from seaswarm.config import load_default_config, load_scenario
from seaswarm.search.motion import BeliefMotion, correlated_displacement_variance
from seaswarm.world.drift import LeewayModel
from seaswarm.world.ocean import OUProcess, UniformCurrent, WindField

DEFAULTS = load_default_config()
PIW1 = LeewayModel.from_config(DEFAULTS.leeway_objects["PIW-1"], tau_s=1800.0)


def _motion(leeway: LeewayModel = PIW1, **kwargs) -> BeliefMotion:
    return BeliefMotion(leeway, UniformCurrent(0.2, 0.0), WindField(10.0, 270.0), **kwargs)


def test_displacement_variance_limits() -> None:
    sigma, tau = 0.12, 1800.0
    # ballistic for t << tau, diffusive (2 sigma^2 tau (t - tau)) for t >> tau
    assert correlated_displacement_variance(sigma, tau, 10.0) == pytest.approx(
        sigma**2 * 10.0**2, rel=0.01
    )
    t = 50 * tau
    assert correlated_displacement_variance(sigma, tau, t) == pytest.approx(
        2 * sigma**2 * tau * (t - tau), rel=1e-9
    )
    assert correlated_displacement_variance(sigma, np.inf, 100.0) == pytest.approx(sigma**2 * 1e4)


def test_displacement_variance_matches_simulated_ou() -> None:
    rng = np.random.default_rng(0)
    ou = OUProcess.stationary(sigma=0.1, tau_s=300.0, shape=20000, rng=rng)
    x = np.zeros(20000)
    for _ in range(1200):
        x += ou.value  # dt = 1 s, same order as the drift integrator
        ou.step(1.0, rng)
    assert x.var() == pytest.approx(correlated_displacement_variance(0.1, 300.0, 1200.0), rel=0.03)


def test_jibe_correlation_time() -> None:
    rate_per_s = -np.log(0.96) / 3600.0
    assert _motion().jibe_tau_s == pytest.approx(1.0 / (2.0 * rate_per_s))
    assert np.isinf(_motion(LeewayModel.simple()).jibe_tau_s)


def test_mean_velocity_is_current_plus_downwind_leeway() -> None:
    v = _motion().mean_velocity_mps(0.0, np.zeros(2))
    np.testing.assert_allclose(v, [0.2 + 0.0096 * 10.0, 0.0], atol=1e-12)


def test_covariance_is_aligned_with_the_wind() -> None:
    motion = _motion()
    down, cross = motion.axis_variances_m2(3600.0)
    cov = motion.covariance_m2(3600.0)  # wind from 270: downwind = east = x
    np.testing.assert_allclose(cov, [[down, 0.0], [0.0, cross]], atol=1e-9)


def test_crosswind_lobes_add_variance() -> None:
    no_lobes = LeewayModel(0.0096, 0.0, 0.12, 0.0, 0.0, 0.094, 0.04, 1800.0)
    assert _motion().axis_variances_m2(7200.0)[1] > _motion(no_lobes).axis_variances_m2(7200.0)[1]


def test_increments_add_up_to_the_cumulative_covariance() -> None:
    scenario = load_scenario(Path(__file__).resolve().parents[2] / "scenarios" / "calm_single.yaml")
    motion = BeliefMotion.from_scenario(scenario, DEFAULTS)
    times = np.arange(0.0, 7201.0, 60.0)
    total = sum(
        motion.covariance_increment_m2(a, b) for a, b in zip(times[:-1], times[1:], strict=True)
    )
    np.testing.assert_allclose(total, motion.covariance_m2(7200.0), rtol=1e-9)
