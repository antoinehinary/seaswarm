"""Predict step of the probability map: shift by the mean drift, spread by the drift covariance.

The step is one convolution of the map with the Gaussian N(d, S): displacement d = v_bar * dt and
covariance increment S. It is computed in Fourier space, where a convolution is a product and the
Gaussian has a closed form (its characteristic function):

    P'(k) = P(k) * exp(-i k.d) * exp(-k^T S k / 2)

This is exact for any sub-cell shift and any (even tiny or strongly anisotropic) covariance, so it
adds no numerical diffusion, unlike linear-interpolation shifting (see tests/unit/test_predict.py).

The map is zero-padded so nothing wraps around the periodic FFT domain; probability that ends up
in the padding has left the grid and is added to p_out. Small negative values from ringing next to
sharp features are clipped, and the remaining mass rescaled to what it was before the step.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import fft

from seaswarm.config import DefaultConfig, Scenario
from seaswarm.search.grid import Grid
from seaswarm.search.motion import BeliefMotion
from seaswarm.search.probability_map import ProbabilityMap

Array = NDArray[np.float64]

PAD_SIGMAS = 6.0  # padding covers the shift plus 6 standard deviations of the kernel


def convolve_gaussian(pmap: ProbabilityMap, shift_m: Array, cov_m2: Array) -> None:
    """Convolve the map in place with N(shift_m, cov_m2); mass leaving the grid goes to p_out.

    Args:
        pmap: Map to update.
        shift_m: Mean displacement (dx, dy), metres.
        cov_m2: 2x2 displacement covariance, m^2 (positive semi-definite).
    """
    h = pmap.grid.cell_size_m
    mass_before = pmap.p.sum()
    if mass_before == 0.0:
        return
    spread_m = np.sqrt(max(np.linalg.eigvalsh(cov_m2).max(), 0.0))
    pad = int(np.ceil((np.abs(shift_m).max() + PAD_SIGMAS * spread_m) / h)) + 2
    ny, nx = pmap.grid.shape
    shape = (fft.next_fast_len(ny + 2 * pad), fft.next_fast_len(nx + 2 * pad, real=True))
    padded = np.zeros(shape)
    padded[pad : pad + ny, pad : pad + nx] = pmap.p

    ky = 2.0 * np.pi * fft.fftfreq(shape[0], d=h)[:, None]  # rad/m
    kx = 2.0 * np.pi * fft.rfftfreq(shape[1], d=h)[None, :]
    quad = cov_m2[0, 0] * kx**2 + 2.0 * cov_m2[0, 1] * kx * ky + cov_m2[1, 1] * ky**2
    transfer = np.exp(-1j * (kx * shift_m[0] + ky * shift_m[1]) - 0.5 * quad)
    moved = fft.irfft2(fft.rfft2(padded) * transfer, s=shape)

    np.maximum(moved, 0.0, out=moved)
    moved *= mass_before / moved.sum()
    inside = moved[pad : pad + ny, pad : pad + nx]
    pmap.p = inside.copy()
    pmap.p_out += mass_before - inside.sum()


def predict(pmap: ProbabilityMap, motion: BeliefMotion, t_s: float, dt_s: float) -> None:
    """Advance the map in place from t_s to t_s + dt_s (times since the fall, s)."""
    shift_m = motion.mean_velocity_mps(t_s + 0.5 * dt_s, pmap.mean_m()) * dt_s
    convolve_gaussian(pmap, shift_m, motion.covariance_increment_m2(t_s, t_s + dt_s))


def advance(
    pmap: ProbabilityMap, motion: BeliefMotion, t_s: float, duration_s: float, period_s: float
) -> float:
    """Run predict steps of `period_s` (the last one shorter if needed); return the new time, s."""
    end_s = t_s + duration_s
    while t_s < end_s - 1e-9:
        dt_s = min(period_s, end_s - t_s)
        predict(pmap, motion, t_s, dt_s)
        t_s += dt_s
    return t_s


def belief_at(scenario: Scenario, defaults: DefaultConfig, t_s: float) -> ProbabilityMap:
    """Prior belief `t_s` after the fall: Gaussian around the LKP, then predict steps."""
    grid = Grid.from_area(scenario.area)
    lkp = scenario.lkp
    pmap = ProbabilityMap.from_gaussian(grid, (lkp.x_m, lkp.y_m), lkp.sigma_m)
    motion = BeliefMotion.from_scenario(scenario, defaults)
    advance(pmap, motion, 0.0, t_s, defaults.belief.predict_period_s)
    return pmap
