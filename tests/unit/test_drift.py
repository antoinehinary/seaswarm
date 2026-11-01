from pathlib import Path

import numpy as np
import pytest

from seaswarm.config import OUNoiseConfig, load_default_config, load_scenario
from seaswarm.world.drift import LeewayModel, ParticleDrift, simulate
from seaswarm.world.ocean import UniformCurrent, WindField

SCENARIOS = Path(__file__).resolve().parents[2] / "scenarios"
PIW1 = LeewayModel.from_config(load_default_config().leeway_objects["PIW-1"], tau_s=1800.0)
CALM = UniformCurrent(0.0, 0.0)


def _drift(
    n: int,
    leeway: LeewayModel = PIW1,
    current=CALM,
    wind: WindField | None = None,
    seed: int = 0,
    start_m: tuple[float, float] = (0.0, 0.0),
    **kwargs,
) -> ParticleDrift:
    wind = wind or WindField(speed_mps=10.0, from_bearing_deg=270.0)
    positions_m = np.tile(start_m, (n, 1))
    return ParticleDrift(positions_m, leeway, current, wind, np.random.default_rng(seed), **kwargs)


# --------------------------------------------------------------------------- acceptance tests


def test_current_only_drift_is_exact() -> None:
    current = UniformCurrent(0.3, -0.1)
    start = np.array([[1200.0, 800.0]])
    drift = _drift(1, leeway=LeewayModel.simple(k=0.0), current=current, start_m=(1200.0, 800.0))
    simulate(drift, duration_s=3600.0, dt_s=1.0, record_every_s=3600.0, max_steps=3600)
    expected = start + np.array([0.3, -0.1]) * 3600.0
    np.testing.assert_allclose(drift.positions_m, expected, rtol=0.0, atol=1e-6)


def test_mean_downwind_leeway_is_0_96_percent_of_wind() -> None:
    # 2000 particles, wind 10 m/s from the west, no current, 3 h
    drift = _drift(2000)
    simulate(drift, duration_s=3 * 3600.0, dt_s=1.0, record_every_s=3600.0, max_steps=20000)
    downwind = np.array([1.0, 0.0])
    mean_speed_mps = (drift.positions_m @ downwind).mean() / (3 * 3600.0)
    assert mean_speed_mps == pytest.approx(0.0096 * 10.0, rel=0.05)


def test_wind_from_270_moves_the_person_east() -> None:
    drift = _drift(1, leeway=LeewayModel.simple(k=0.0096))
    simulate(drift, duration_s=3600.0, dt_s=1.0, record_every_s=3600.0, max_steps=3600)
    x_m, y_m = drift.positions_m[0]
    assert x_m == pytest.approx(0.0096 * 10.0 * 3600.0)
    assert y_m == pytest.approx(0.0, abs=1e-9)


def test_same_seed_gives_bit_identical_trajectories() -> None:
    runs = [simulate(_drift(50, seed=s), 1800.0, 1.0, 60.0, max_steps=1800)[1] for s in (4, 4, 5)]
    np.testing.assert_array_equal(runs[0], runs[1])
    assert not np.array_equal(runs[0], runs[2])


# --------------------------------------------------------------------------- model details


def test_crosswind_mean_is_zero_but_spread_is_not() -> None:
    drift = _drift(2000)
    simulate(drift, duration_s=3 * 3600.0, dt_s=1.0, record_every_s=3600.0, max_steps=20000)
    y_m = drift.positions_m[:, 1]  # crosswind axis for a westerly wind
    assert abs(y_m.mean()) < 3 * y_m.std() / np.sqrt(len(y_m))
    assert y_m.std() > 0.0054 * 10.0 * 3 * 3600.0 * 0.5  # both lobes are far from the axis


def test_crosswind_is_to_the_right_of_downwind_for_positive_side() -> None:
    leeway = LeewayModel(0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1800.0)  # crosswind only
    drift = _drift(20, leeway=leeway)
    side = drift.crosswind_side
    assert set(side) == {-1.0, 1.0}
    # downwind is east, so the right-hand side (+1) is south and the left (-1) north
    expected = np.column_stack([np.zeros(20), -0.1 * side])
    np.testing.assert_allclose(drift.leeway_velocity(), expected, atol=1e-12)


def test_jibe_rate_is_four_percent_per_hour() -> None:
    drift = _drift(20000, wind=WindField(0.0, 0.0))
    start = drift.crosswind_side
    simulate(drift, duration_s=3600.0, dt_s=1.0, record_every_s=3600.0, max_steps=3600)
    flipped = np.mean(drift.crosswind_side != start)
    # flips form a Poisson process of rate -ln(0.96) per hour; a net flip needs an odd count
    rate = -np.log(0.96)
    assert flipped == pytest.approx((1.0 - np.exp(-2.0 * rate)) / 2.0, abs=0.005)


def test_jibe_probability_per_step_compounds_to_the_hourly_value() -> None:
    p_step = PIW1.jibe_probability(1.0)
    assert 1.0 - (1.0 - p_step) ** 3600 == pytest.approx(0.04)


def test_gusts_never_make_the_wind_speed_negative() -> None:
    drift = _drift(5000, wind=WindField(1.0, 270.0), gusts=OUNoiseConfig(sigma_mps=2.0, tau_s=60))
    assert np.all(drift.wind_speed_mps() >= 0.0)
    assert np.any(drift.wind_speed_mps() == 0.0)


def test_current_noise_spreads_particles_in_still_air() -> None:
    noise = OUNoiseConfig(sigma_mps=0.05, tau_s=600.0)
    drift = _drift(500, leeway=LeewayModel.simple(k=0.0), current_noise=noise)
    simulate(drift, duration_s=3600.0, dt_s=1.0, record_every_s=3600.0, max_steps=3600)
    # long-time OU spread: sigma * sqrt(2 * tau * T) ~ 0.05 * sqrt(2 * 600 * 3600) ~ 104 m
    assert drift.positions_m.std(axis=0) == pytest.approx([100.0, 100.0], rel=0.2)


def test_from_scenario_uses_the_scenario_fields() -> None:
    scenario = load_scenario(SCENARIOS / "windy_mixed.yaml")
    drift = ParticleDrift.from_scenario(
        scenario, load_default_config(), np.zeros((10, 2)), np.random.default_rng(0)
    )
    assert drift.leeway.downwind_slope == 0.0096
    assert drift.wind.rotation_deg_per_h == -5.0
    assert drift.wind_speed_mps().std() > 0.0  # gusts are on


def test_simulate_enforces_the_step_limit() -> None:
    with pytest.raises(ValueError, match="exceed"):
        simulate(_drift(1), duration_s=100.0, dt_s=1.0, record_every_s=10.0, max_steps=99)


def test_simulate_rejects_partial_steps() -> None:
    with pytest.raises(ValueError, match="whole number"):
        simulate(_drift(1), duration_s=10.5, dt_s=1.0, record_every_s=1.0, max_steps=100)


def test_simulate_records_frames_including_the_start() -> None:
    times, frames = simulate(_drift(3), 600.0, 1.0, 60.0, max_steps=600)
    np.testing.assert_allclose(times, np.arange(0.0, 601.0, 60.0))
    assert frames.shape == (11, 3, 2)
