import numpy as np
import pytest

from seaswarm.config import AreaConfig
from seaswarm.search.grid import Grid
from seaswarm.search.probability_map import ProbabilityMap

GRID = Grid.from_area(AreaConfig(width_m=10000, height_m=10000, cell_size_m=100))


def test_gaussian_prior_is_normalised_and_centred() -> None:
    pmap = ProbabilityMap.from_gaussian(GRID, (1230.0, -870.0), sigma_m=200.0)
    assert pmap.total() == pytest.approx(1.0, abs=1e-12)
    assert pmap.p_out == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(pmap.mean_m(), [1230.0, -870.0], atol=0.5)


def test_gaussian_prior_variance_includes_the_cell_width() -> None:
    # cell-integrated Gaussian: variance of the cell centres = sigma^2 + h^2 / 12 (Sheppard)
    pmap = ProbabilityMap.from_gaussian(GRID, (0.0, 0.0), sigma_m=300.0)
    expected = 300.0**2 + 100.0**2 / 12
    np.testing.assert_allclose(np.diag(pmap.covariance_m2()), [expected] * 2, rtol=1e-3)


def test_mass_off_the_grid_goes_to_p_out() -> None:
    pmap = ProbabilityMap.from_gaussian(GRID, (5000.0, 0.0), sigma_m=500.0)  # on the east edge
    assert pmap.p_out == pytest.approx(0.5, abs=1e-6)
    assert pmap.total() == pytest.approx(1.0, abs=1e-12)


def test_invalid_maps_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        ProbabilityMap(GRID, np.zeros((3, 3)), 1.0)
    with pytest.raises(ValueError, match="non-negative"):
        ProbabilityMap(GRID, -np.ones(GRID.shape) / GRID.nx / GRID.ny, 2.0)
    with pytest.raises(ValueError, match="total"):
        ProbabilityMap(GRID, np.zeros(GRID.shape), 0.5)


def test_from_samples_counts_particles() -> None:
    xy = np.array([[50.0, 50.0], [60.0, 70.0], [-50.0, 50.0], [9e9, 0.0]])
    pmap = ProbabilityMap.from_samples(GRID, xy)
    assert pmap.p.max() == 0.5 and pmap.p_out == 0.25


def test_entropy_of_uniform_and_point_maps() -> None:
    uniform = ProbabilityMap(GRID, np.full(GRID.shape, 1e-4))
    assert uniform.entropy_bits() == pytest.approx(np.log2(10000))
    point = np.zeros(GRID.shape)
    point[10, 10] = 1.0
    assert ProbabilityMap(GRID, point).entropy_bits() == 0.0


def test_area_and_top_cells() -> None:
    p = np.zeros(GRID.shape)
    p[50, 50], p[50, 51], p[10, 10] = 0.6, 0.35, 0.05
    pmap = ProbabilityMap(GRID, p)
    assert pmap.area_km2(0.9) == pytest.approx(0.02)  # two 100 m cells
    assert pmap.top_cells(2) == [(50.0, 50.0, 0.6), (150.0, 50.0, 0.35)]
