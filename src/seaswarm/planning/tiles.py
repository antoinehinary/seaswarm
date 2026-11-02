"""Search tiles (blocks of grid cells) and lawnmower (boustrophedon) patterns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from seaswarm.safety.rules import XY
from seaswarm.search.grid import Grid
from seaswarm.search.probability_map import ProbabilityMap


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle, metres."""

    x_min_m: float
    y_min_m: float
    x_max_m: float
    y_max_m: float

    @property
    def width_m(self) -> float:
        """East-west extent."""
        return self.x_max_m - self.x_min_m

    @property
    def height_m(self) -> float:
        """North-south extent."""
        return self.y_max_m - self.y_min_m

    def shifted(self, dx_m: float, dy_m: float) -> Rect:
        """Same rectangle moved by (dx, dy)."""
        return Rect(
            self.x_min_m + dx_m, self.y_min_m + dy_m, self.x_max_m + dx_m, self.y_max_m + dy_m
        )


@dataclass(frozen=True)
class TileGrid:
    """Square tiles of `factor` x `factor` grid cells, ids "t_<row>_<col>" (row 0 south)."""

    grid: Grid
    factor: int

    @classmethod
    def from_grid(cls, grid: Grid, tile_size_m: float) -> TileGrid:
        """Tiles of `tile_size_m` over a grid; the size must be a whole number of cells."""
        factor = round(tile_size_m / grid.cell_size_m)
        if (
            abs(factor * grid.cell_size_m - tile_size_m) > 1e-9
            or grid.nx % factor
            or grid.ny % factor
        ):
            raise ValueError(f"{tile_size_m} m tiles do not divide the {grid.shape} grid")
        return cls(grid, factor)

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols) of tiles."""
        return self.grid.ny // self.factor, self.grid.nx // self.factor

    @property
    def size_m(self) -> float:
        """Tile side, metres."""
        return self.factor * self.grid.cell_size_m

    def values(self, pmap: ProbabilityMap) -> NDArray[np.float64]:
        """Probability mass in each tile, shape (rows, cols)."""
        return pmap.coarsened(self.factor).p

    def centres_m(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """x and y of every tile centre, each shape (rows, cols)."""
        rows, cols = self.shape
        xs = self.grid.x_min_m + self.size_m * (np.arange(cols) + 0.5)
        ys = self.grid.y_min_m + self.size_m * (np.arange(rows) + 0.5)
        return np.meshgrid(xs, ys)

    def rect(self, row: int, col: int) -> Rect:
        """Bounds of one tile."""
        x0 = self.grid.x_min_m + col * self.size_m
        y0 = self.grid.y_min_m + row * self.size_m
        return Rect(x0, y0, x0 + self.size_m, y0 + self.size_m)

    @staticmethod
    def tile_id(row: int, col: int) -> str:
        """Id of the tile at (row, col)."""
        return f"t_{row}_{col}"


def lawnmower(
    rect: Rect, spacing_m: float, start_xy_m: XY, north_south: bool = False
) -> tuple[XY, ...]:
    """Boustrophedon over `rect`, starting at the corner nearest `start_xy_m`.

    Tracks run east-west (or north-south). n = ceil(extent / spacing) tracks are evenly spaced, so
    the actual spacing is <= `spacing_m`, each at the centre of its band.
    """
    if north_south:  # same pattern in the transposed frame
        flipped = Rect(rect.y_min_m, rect.x_min_m, rect.y_max_m, rect.x_max_m)
        path = lawnmower(flipped, spacing_m, (start_xy_m[1], start_xy_m[0]))
        return tuple((y, x) for x, y in path)
    n = max(1, int(np.ceil(rect.height_m / spacing_m)))
    ys = rect.y_min_m + (np.arange(n) + 0.5) * rect.height_m / n
    if abs(start_xy_m[1] - rect.y_max_m) < abs(start_xy_m[1] - rect.y_min_m):
        ys = ys[::-1]
    west_first = abs(start_xy_m[0] - rect.x_min_m) <= abs(start_xy_m[0] - rect.x_max_m)
    waypoints: list[XY] = []
    for i, y in enumerate(ys):
        eastward = (i % 2 == 0) == west_first
        ends = (rect.x_min_m, rect.x_max_m) if eastward else (rect.x_max_m, rect.x_min_m)
        waypoints += [(float(ends[0]), float(y)), (float(ends[1]), float(y))]
    return tuple(waypoints)


def lawnmower_length_m(rect: Rect, spacing_m: float) -> float:
    """Length of `lawnmower(rect, spacing_m, ...)` from its first waypoint, metres."""
    n = max(1, int(np.ceil(rect.height_m / spacing_m)))
    return n * rect.width_m + (n - 1) * rect.height_m / n
