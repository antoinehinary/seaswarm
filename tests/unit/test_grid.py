import numpy as np
import pytest

from seaswarm.config import AreaConfig
from seaswarm.search.grid import Grid

GRID = Grid.from_area(AreaConfig(width_m=10000, height_m=6000, cell_size_m=100))


def test_grid_from_area_is_centred_on_the_origin() -> None:
    assert GRID.shape == (60, 100)
    assert (GRID.x_min_m, GRID.y_min_m) == (-5000.0, -3000.0)
    assert GRID.x_edges_m[-1] == 5000.0
    assert GRID.x_centres_m[0] == -4950.0
    x, y = GRID.centres_m()
    assert x.shape == y.shape == GRID.shape
    assert y[0, 0] == -2950.0  # row 0 is the southern edge


def test_cell_of_finds_the_cell_and_its_centre() -> None:
    row, col = GRID.cell_of(np.array([[-4999.0, -2999.0], [10.0, -10.0], [4999.9, 2999.9]]))
    np.testing.assert_array_equal(row, [0, 29, 59])
    np.testing.assert_array_equal(col, [0, 50, 99])
    assert GRID.x_centres_m[50] == 50.0 and GRID.y_centres_m[29] == -50.0


@pytest.mark.parametrize("point", [(-5000.1, 0.0), (5000.0, 0.0), (0.0, 3000.0), (0.0, -3001.0)])
def test_points_off_the_grid_get_minus_one(point: tuple[float, float]) -> None:
    row, col = GRID.cell_of(np.array([point]))
    assert (row[0], col[0]) == (-1, -1)


def test_histogram_counts_inside_and_outside() -> None:
    xy = np.array([[10.0, 10.0], [20.0, 30.0], [-4990.0, 0.0], [9e9, 0.0]])
    counts, n_outside = GRID.histogram(xy)
    assert counts.sum() == 3 and n_outside == 1
    row, col = GRID.cell_of(xy[:1])
    assert counts[row[0], col[0]] == 2
