import numpy as np
import pytest

from seaswarm.config import UniformCurrentConfig, WindConfig
from seaswarm.world.ocean import (
    GyreCurrent,
    LinearGradientCurrent,
    OUProcess,
    UniformCurrent,
    WindField,
    make_current,
)


def _grid_points(half_width_m: float = 5000.0, n: int = 41) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-half_width_m, half_width_m, n)
    xx, yy = np.meshgrid(x, x)
    return xx, yy


def _divergence(field, half_width_m: float = 5000.0, h_m: float = 1.0) -> np.ndarray:
    """Central-difference divergence (1/s) of a current field on a grid of points."""
    xx, yy = _grid_points(half_width_m)
    p = np.column_stack([xx.ravel(), yy.ravel()])
    ex, ey = np.array([h_m, 0.0]), np.array([0.0, h_m])
    du_dx = (field.velocity(p + ex, 0.0)[:, 0] - field.velocity(p - ex, 0.0)[:, 0]) / (2 * h_m)
    dv_dy = (field.velocity(p + ey, 0.0)[:, 1] - field.velocity(p - ey, 0.0)[:, 1]) / (2 * h_m)
    return du_dx + dv_dy


# --------------------------------------------------------------------------- OU process


def test_ou_keeps_its_stationary_std() -> None:
    rng = np.random.default_rng(0)
    ou = OUProcess.stationary(sigma=0.12, tau_s=1800.0, shape=20000, rng=rng)
    for _ in range(3600):
        ou.step(1.0, rng)
    assert ou.value.std() == pytest.approx(0.12, rel=0.02)


def test_ou_autocorrelation_at_one_tau_is_one_over_e() -> None:
    rng = np.random.default_rng(1)
    ou = OUProcess.stationary(sigma=1.0, tau_s=100.0, shape=20000, rng=rng)
    start = ou.value.copy()
    for _ in range(100):
        ou.step(1.0, rng)
    correlation = np.corrcoef(start, ou.value)[0, 1]
    assert correlation == pytest.approx(np.exp(-1.0), abs=0.02)


def test_ou_statistics_do_not_depend_on_dt() -> None:
    # exact discretisation: one 600 s step and 600 steps of 1 s give the same distribution
    rng = np.random.default_rng(2)
    coarse = OUProcess(sigma=1.0, tau_s=300.0, value=np.zeros(20000))
    fine = OUProcess(sigma=1.0, tau_s=300.0, value=np.zeros(20000))
    coarse.step(600.0, rng)
    for _ in range(600):
        fine.step(1.0, rng)
    expected_std = np.sqrt(1.0 - np.exp(-2.0 * 600.0 / 300.0))
    assert coarse.value.std() == pytest.approx(expected_std, rel=0.03)
    assert fine.value.std() == pytest.approx(expected_std, rel=0.03)


# --------------------------------------------------------------------------- currents


def test_uniform_current_from_config_points_towards_its_bearing() -> None:
    field = make_current(UniformCurrentConfig(type="uniform", speed_mps=0.3, to_bearing_deg=90))
    v = field.velocity(np.zeros((3, 2)), 0.0)
    np.testing.assert_allclose(v, [[0.3, 0.0]] * 3, atol=1e-12)


def test_uniform_current_returns_a_writable_copy() -> None:
    v = UniformCurrent(0.1, 0.2).velocity(np.zeros((2, 2)), 0.0)
    v[0, 0] = 9.0  # must not raise (broadcast_to alone would be read-only)
    assert v[1, 0] == 0.1


def test_shear_current_is_faster_on_the_left_and_divergence_free() -> None:
    field = LinearGradientCurrent(speed_mps=0.4, to_bearing_deg=90.0, shear_per_s=2e-5)
    # flowing east: "left" is north
    north, south = field.velocity(np.array([[0.0, 1000.0], [0.0, -1000.0]]), 0.0)
    np.testing.assert_allclose(north, [0.42, 0.0], atol=1e-12)
    np.testing.assert_allclose(south, [0.38, 0.0], atol=1e-12)
    assert np.max(np.abs(_divergence(field))) < 1e-12


def test_gyre_is_divergence_free() -> None:
    field = GyreCurrent(500.0, -300.0, radius_m=2000.0, max_speed_mps=0.5, clockwise=True)
    # speeds ~0.5 m/s over ~2 km: a real divergence would be ~1e-4 1/s
    assert np.max(np.abs(_divergence(field))) < 1e-9


def test_gyre_speed_peaks_at_its_radius() -> None:
    field = GyreCurrent(0.0, 0.0, radius_m=2000.0, max_speed_mps=0.5, clockwise=False)
    r_m = np.linspace(0.0, 6000.0, 601)
    speed = np.linalg.norm(field.velocity(np.column_stack([r_m, np.zeros_like(r_m)]), 0.0), axis=1)
    assert r_m[np.argmax(speed)] == pytest.approx(2000.0)
    assert speed.max() == pytest.approx(0.5)


@pytest.mark.parametrize(("clockwise", "v_sign"), [(True, -1.0), (False, 1.0)])
def test_gyre_turns_in_the_requested_sense(clockwise: bool, v_sign: float) -> None:
    # east of the centre, a clockwise eddy flows south and a counter-clockwise one north
    field = GyreCurrent(0.0, 0.0, radius_m=1000.0, max_speed_mps=0.5, clockwise=clockwise)
    u_mps, v_mps = field.velocity(np.array([[1000.0, 0.0]]), 0.0)[0]
    assert u_mps == pytest.approx(0.0, abs=1e-12)
    assert np.sign(v_mps) == v_sign


# --------------------------------------------------------------------------- wind


def test_wind_from_270_blows_east() -> None:
    wind = WindField.from_config(WindConfig(type="uniform", speed_mps=8.0, from_bearing_deg=270))
    np.testing.assert_allclose(wind.velocity(0.0), [8.0, 0.0], atol=1e-12)


def test_wind_veers_at_the_configured_rate() -> None:
    wind = WindField(speed_mps=8.0, from_bearing_deg=350.0, rotation_deg_per_h=20.0)
    assert wind.from_bearing_at(1800.0) == pytest.approx(0.0, abs=1e-9)
    assert wind.from_bearing_at(3600.0) == pytest.approx(10.0)


def test_downwind_unit_is_defined_in_calm_air() -> None:
    wind = WindField(speed_mps=0.0, from_bearing_deg=0.0)
    np.testing.assert_allclose(wind.downwind_unit(0.0), [0.0, -1.0], atol=1e-12)
