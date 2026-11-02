"""Aggregate statistics for the evaluation: survival curves, bootstrap intervals, paired tests.

Detection times are right-censored: a run that ends without a detection only tells us the time is
longer than the run. The Kaplan-Meier estimator handles that; with every run censored at the same
time limit (our case) it reduces to the empirical fraction found by each time, but it stays
correct if runs of different lengths are mixed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def kaplan_meier(times_s: Sequence[float], observed: Sequence[bool]) -> tuple[Array, Array]:
    """Product-limit estimate of P(found by t).

    Args:
        times_s: Detection time, or censoring time for runs without detection.
        observed: True where the time is a detection, False where it is censored.

    Returns:
        (t, found): step function knots, found[i] = estimated fraction found by t[i].
    """
    t = np.asarray(times_s, dtype=float)
    d = np.asarray(observed, dtype=bool)
    knots, found = [0.0], [0.0]
    survival = 1.0
    for time in np.unique(t[d]):
        at_risk = np.count_nonzero(t >= time)
        events = np.count_nonzero((t == time) & d)
        survival *= 1.0 - events / at_risk
        knots.append(float(time))
        found.append(1.0 - survival)
    return np.array(knots), np.array(found)


def km_median_s(times_s: Sequence[float], observed: Sequence[bool]) -> float:
    """Median time to detection from Kaplan-Meier; inf if fewer than half are ever found."""
    knots, found = kaplan_meier(times_s, observed)
    reached = np.nonzero(found >= 0.5)[0]
    return float(knots[reached[0]]) if reached.size else float("inf")


def found_by(times_s: Sequence[float], observed: Sequence[bool], t_s: float) -> Array:
    """Per run: 1.0 if detected by t_s, else 0.0."""
    return ((np.asarray(times_s) <= t_s) & np.asarray(observed, dtype=bool)).astype(float)


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Array], float] = np.mean,
    n_resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(estimate, low, high): percentile bootstrap interval over runs, fixed seed (reproducible)."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resampled = [statistic(x[rng.integers(0, len(x), len(x))]) for _ in range(n_resamples)]
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(resampled, [tail, 100.0 - tail])
    return float(statistic(x)), float(low), float(high)


def paired_difference(
    a: Sequence[float], b: Sequence[float], seed: int = 0
) -> tuple[float, float, float]:
    """Mean of a - b over paired runs (same scenario and seed) with a bootstrap interval.

    Pairing removes the scenario-to-scenario variance, which dominates: the interval of the mean
    difference is much tighter than comparing two independent means.
    """
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), seed=seed)
