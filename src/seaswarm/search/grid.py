"""Square-celled grid over the search area.

Arrays are indexed [row, col] = [iy, ix], with row 0 at the southern edge (use
`imshow(..., origin="lower")`). Cells are half-open: [x_lo, x_hi) x [y_lo, y_hi).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from seaswarm.config import AreaConfig

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Grid:
    """Geometry of the grid: size, cell size and position of the south-west corner."""

    nx: int
    ny: int
    cell_size_m: float
    x_min_m: float
    y_min_m: float

    @classmethod
    def from_area(cls, area: AreaConfig) -> Grid:
        """Grid covering a scenario search area, centred on the ENU origin."""
        return cls(
            nx=round(area.width_m / area.cell_size_m),
            ny=round(area.height_m / area.cell_size_m),
            cell_size_m=area.cell_size_m,
            x_min_m=-area.width_m / 2,
            y_min_m=-area.height_m / 2,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Array shape (ny, nx)."""
        return self.ny, self.nx

    @property
    def x_edges_m(self) -> Array:
        """Cell edges along x, shape (nx + 1,), metres."""
        return self.x_min_m + self.cell_size_m * np.arange(self.nx + 1)

    @property
    def y_edges_m(self) -> Array:
        """Cell edges along y, shape (ny + 1,), metres."""
        return self.y_min_m + self.cell_size_m * np.arange(self.ny + 1)

    @property
    def x_centres_m(self) -> Array:
        """Cell centres along x, shape (nx,), metres."""
        return self.x_min_m + self.cell_size_m * (np.arange(self.nx) + 0.5)

    @property
    def y_centres_m(self) -> Array:
        """Cell centres along y, shape (ny,), metres."""
        return self.y_min_m + self.cell_size_m * (np.arange(self.ny) + 0.5)

    def centres_m(self) -> tuple[Array, Array]:
        """x and y of every cell centre, each shape (ny, nx), metres."""
        return np.meshgrid(self.x_centres_m, self.y_centres_m)

    def cell_of(self, xy_m: Array) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
        """(row, col) of the cell containing each point of xy_m (N, 2); -1 for points outside."""
        xy_m = np.asarray(xy_m, dtype=float).reshape(-1, 2)
        col = np.floor((xy_m[:, 0] - self.x_min_m) / self.cell_size_m).astype(np.intp)
        row = np.floor((xy_m[:, 1] - self.y_min_m) / self.cell_size_m).astype(np.intp)
        outside = (col < 0) | (col >= self.nx) | (row < 0) | (row >= self.ny)
        row[outside] = -1
        col[outside] = -1
        return row, col

    def histogram(self, xy_m: Array) -> tuple[Array, int]:
        """Count points per cell.

        Returns:
            (counts, n_outside): counts shape (ny, nx), and the number of points off the grid.
        """
        row, col = self.cell_of(xy_m)
        inside = row >= 0
        counts = np.zeros(self.shape)
        np.add.at(counts, (row[inside], col[inside]), 1.0)
        return counts, int(np.count_nonzero(~inside))
