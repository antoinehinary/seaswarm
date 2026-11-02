"""Bayes updates of the probability map after a look.

No detection (the "searched here, found nothing" update):

    p_i   <- p_i (1 - POD_i) / (1 - sum_j p_j POD_j)
    p_out <- p_out / (1 - sum_j p_j POD_j)          (POD = 0 off the grid)

The denominator is the probability of not detecting; sum_j p_j POD_j is the probability of success
of that look, POS_k. Over many looks, POS_cum = 1 - prod_k (1 - POS_k).

Detection: in v1 sensors have no false alarms, so a detection at measured position z replaces the
belief with a Gaussian around z (measurement noise sigma).
"""

from __future__ import annotations

from dataclasses import dataclass

from seaswarm.search.probability_map import ProbabilityMap
from seaswarm.vehicles.sensors import Footprint


def update_no_detection(pmap: ProbabilityMap, look: Footprint) -> float:
    """Apply an unsuccessful look in place and return its probability of success POS_k."""
    cells = (look.rows, look.cols)
    pos = float((pmap.p[cells] * look.pod).sum())
    if pos >= 1.0:
        raise ValueError("a look with POS = 1 cannot fail to detect")
    pmap.p[cells] *= 1.0 - look.pod
    pmap.p /= 1.0 - pos
    pmap.p_out /= 1.0 - pos
    return pos


def update_detection(pmap: ProbabilityMap, z_m: tuple[float, float], sigma_m: float) -> None:
    """Replace the belief in place by a Gaussian around the detection z_m (no false alarms)."""
    fresh = ProbabilityMap.from_gaussian(pmap.grid, z_m, sigma_m)
    pmap.p, pmap.p_out = fresh.p, fresh.p_out


@dataclass
class SuccessTracker:
    """Cumulative probability of success POS_cum = 1 - prod_k (1 - POS_k)."""

    p_all_failed: float = 1.0

    def add(self, pos_k: float) -> None:
        """Record the POS of one more unsuccessful look."""
        self.p_all_failed *= 1.0 - pos_k

    @property
    def pos_cum(self) -> float:
        """Probability that at least one of the looks so far would have detected the person."""
        return 1.0 - self.p_all_failed
