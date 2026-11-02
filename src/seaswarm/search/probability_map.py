"""Probability map: the belief about where the person in water is.

State: `p[ny, nx]`, the probability that the person is in each cell, plus `p_out`, the probability
that they are off the grid. Invariant: p.sum() + p_out == 1 (to 1e-9). Every operation that moves
mass off the grid adds it to p_out, so the invariant holds by bookkeeping, not by renormalising.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from seaswarm.search.grid import Grid

Array = NDArray[np.float64]

TOTAL_TOLERANCE = 1e-9


class ProbabilityMap:
    """Cell probabilities over a grid plus the probability of being off the grid."""

    def __init__(self, grid: Grid, p: Array, p_out: float = 0.0) -> None:
        """Wrap a probability array; checks shape, sign and total."""
        p = np.array(p, dtype=float)
        if p.shape != grid.shape:
            raise ValueError(f"p has shape {p.shape}, the grid is {grid.shape}")
        if np.any(p < 0.0) or p_out < 0.0:
            raise ValueError("probabilities must be non-negative")
        self.grid = grid
        self.p = p
        self.p_out = float(p_out)
        self.check_total()

    @classmethod
    def from_gaussian(
        cls, grid: Grid, mean_m: tuple[float, float], sigma_m: float
    ) -> ProbabilityMap:
        """Isotropic Gaussian prior, integrated exactly over each cell.

        The 2D Gaussian is separable, so the mass of a cell is the product of two 1D masses, each
        a difference of normal CDFs at the cell edges. Whatever falls outside goes to p_out.
        """
        mass_x = np.diff(ndtr((grid.x_edges_m - mean_m[0]) / sigma_m))
        mass_y = np.diff(ndtr((grid.y_edges_m - mean_m[1]) / sigma_m))
        p = np.outer(mass_y, mass_x)
        return cls(grid, p, p_out=max(0.0, 1.0 - p.sum()))

    @classmethod
    def from_samples(cls, grid: Grid, xy_m: Array) -> ProbabilityMap:
        """Empirical map from equally weighted samples (Monte Carlo particles)."""
        counts, n_outside = grid.histogram(xy_m)
        n = counts.sum() + n_outside
        return cls(grid, counts / n, p_out=n_outside / n)

    def copy(self) -> ProbabilityMap:
        """Independent copy."""
        return ProbabilityMap(self.grid, self.p.copy(), self.p_out)

    def total(self) -> float:
        """p.sum() + p_out; equals 1 up to rounding."""
        return float(self.p.sum() + self.p_out)

    def check_total(self) -> None:
        """Raise if the invariant p.sum() + p_out == 1 is broken."""
        if abs(self.total() - 1.0) > TOTAL_TOLERANCE:
            raise ValueError(f"total probability is {self.total():.12f}, expected 1")

    def mean_m(self) -> Array:
        """Mean (x, y) of the in-grid probability, metres."""
        x, y = self.grid.centres_m()
        mass = self.p.sum()
        return np.array([(self.p * x).sum(), (self.p * y).sum()]) / mass

    def covariance_m2(self) -> Array:
        """2x2 covariance of the in-grid probability (cell centres), m^2."""
        x, y = self.grid.centres_m()
        dx, dy = x - self.mean_m()[0], y - self.mean_m()[1]
        w = self.p / self.p.sum()
        return np.array(
            [[(w * dx * dx).sum(), (w * dx * dy).sum()], [(w * dx * dy).sum(), (w * dy * dy).sum()]]
        )

    def entropy_bits(self) -> float:
        """Shannon entropy over the cells plus the off-grid bin, bits."""
        q = np.append(self.p.ravel(), self.p_out)
        q = q[q > 0.0]
        return float(-(q * np.log2(q)).sum())

    def area_km2(self, fraction: float = 0.9) -> float:
        """Smallest area (km^2) of cells holding `fraction` of the in-grid probability."""
        ranked = np.sort(self.p.ravel())[::-1]
        n_cells = int(np.searchsorted(np.cumsum(ranked), fraction * ranked.sum()) + 1)
        return min(n_cells, ranked.size) * self.grid.cell_size_m**2 / 1e6

    def top_cells(self, k: int) -> list[tuple[float, float, float]]:
        """The k most probable cells as (x_m, y_m, p), most probable first."""
        flat = np.argsort(self.p.ravel())[::-1][:k]
        rows, cols = np.unravel_index(flat, self.grid.shape)
        xs, ys = self.grid.x_centres_m, self.grid.y_centres_m
        return [
            (float(xs[c]), float(ys[r]), float(self.p[r, c]))
            for r, c in zip(rows, cols, strict=True)
        ]
