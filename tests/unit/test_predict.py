import numpy as np
import pytest
from scipy import ndimage

from seaswarm.config import AreaConfig, load_default_config
from seaswarm.search.grid import Grid
from seaswarm.search.motion import BeliefMotion
from seaswarm.search.predict import advance, convolve_gaussian, predict
from seaswarm.search.probability_map import ProbabilityMap
from seaswarm.world.drift import LeewayModel
from seaswarm.world.ocean import UniformCurrent, WindField

GRID = Grid.from_area(AreaConfig(width_m=10000, height_m=10000, cell_size_m=100))
H = GRID.cell_size_m
PIW1 = LeewayModel.from_config(load_default_config().leeway_objects["PIW-1"], tau_s=1800.0)


def _prior(mean_m=(0.0, 0.0), sigma_m: float = 300.0) -> ProbabilityMap:
    return ProbabilityMap.from_gaussian(GRID, mean_m, sigma_m)


def test_fractional_shift_moves_the_mean_without_spreading() -> None:
    pmap = _prior()
    cov0 = pmap.covariance_m2()
    for _ in range(40):  # 1.2 km east, 2.8 km south: stays well inside the grid
        convolve_gaussian(pmap, np.array([0.3 * H, -0.7 * H]), np.zeros((2, 2)))
    assert pmap.p_out < 1e-12
    np.testing.assert_allclose(pmap.mean_m(), [12.0 * H, -28.0 * H], atol=1e-6)
    np.testing.assert_allclose(pmap.covariance_m2(), cov0, rtol=1e-6, atol=1e-3)


def test_linear_interpolation_shift_adds_numerical_diffusion() -> None:
    # The brief's ndimage.shift(order=1): each fractional shift f adds f (1 - f) cells^2 of
    # variance. 100 steps of 0.3 cell add 21 cells^2 = 210 000 m^2, i.e. the 300 m prior
    # widens to ~490 m with no physics at all. The spectral step adds none (test above).
    p = _prior().p
    var0 = _prior().covariance_m2()[0, 0]
    for _ in range(100):
        p = ndimage.shift(p, (0.0, 0.3), order=1, mode="constant", cval=0.0)
    shifted = ProbabilityMap(GRID, p / p.sum())
    added_cells2 = (shifted.covariance_m2()[0, 0] - var0) / H**2
    assert added_cells2 == pytest.approx(100 * 0.3 * 0.7, rel=0.01)


def test_rotated_anisotropic_spread_adds_exactly_its_covariance() -> None:
    pmap = _prior()
    cov0 = pmap.covariance_m2()
    angle = np.radians(30.0)
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    added = rot @ np.diag([400.0**2, 150.0**2]) @ rot.T
    convolve_gaussian(pmap, np.zeros(2), added)
    np.testing.assert_allclose(pmap.covariance_m2(), cov0 + added, rtol=1e-6)


def test_tiny_spread_is_exact_too() -> None:
    # one 60 s step of early leeway noise: ~0.05 cells^2, far below what a sampled kernel resolves
    pmap = _prior()
    cov0 = pmap.covariance_m2()
    added = np.diag([500.0, 200.0])  # m^2
    convolve_gaussian(pmap, np.zeros(2), added)
    np.testing.assert_allclose(pmap.covariance_m2() - cov0, added, atol=1e-3)


def test_mass_leaving_the_grid_goes_to_p_out() -> None:
    pmap = _prior(mean_m=(4500.0, 0.0))
    for _ in range(20):
        convolve_gaussian(pmap, np.array([50.0, 0.0]), np.eye(2) * 2500.0)
        assert pmap.total() == pytest.approx(1.0, abs=1e-12)
    assert pmap.p_out > 0.9


def test_sharp_features_stay_non_negative_and_conserved() -> None:
    p = np.zeros(GRID.shape)
    p[50, 50] = 1.0
    pmap = ProbabilityMap(GRID, p)
    convolve_gaussian(pmap, np.array([0.5 * H, 0.0]), np.eye(2) * 10.0)
    assert np.all(pmap.p >= 0.0)
    assert pmap.total() == pytest.approx(1.0, abs=1e-12)


def test_predict_drifts_with_current_and_leeway() -> None:
    motion = BeliefMotion(PIW1, UniformCurrent(0.3, 0.0), WindField(10.0, 180.0))
    pmap = _prior()
    t_s = advance(pmap, motion, 0.0, 3600.0, 60.0)
    assert t_s == 3600.0
    # current east 0.3 m/s; wind from south pushes north at 0.96 % of 10 m/s
    np.testing.assert_allclose(pmap.mean_m(), [0.3 * 3600, 0.096 * 3600], atol=1.0)
    added = pmap.covariance_m2() - _prior().covariance_m2()
    np.testing.assert_allclose(added, motion.covariance_m2(3600.0), rtol=1e-3, atol=50.0)


def test_advance_takes_a_shorter_last_step() -> None:
    motion = BeliefMotion(PIW1, UniformCurrent(0.0, 0.0), WindField(0.0, 0.0))
    assert advance(_prior(), motion, 100.0, 150.0, 60.0) == pytest.approx(250.0)


def test_predict_single_step_is_in_place() -> None:
    motion = BeliefMotion(PIW1, UniformCurrent(0.1, 0.0), WindField(5.0, 0.0))
    pmap = _prior()
    predict(pmap, motion, 0.0, 60.0)
    assert pmap.mean_m()[0] > 0.0
