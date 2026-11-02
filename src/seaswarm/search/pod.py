"""Probability of detection (POD) of one sensing event, as a function of horizontal range.

Lateral-range model: POD(r) = pod_max * exp(-r^2 / (2 sigma_r^2)) for r <= r_max, else 0.
Successive looks are treated as independent, which is optimistic: real misses are correlated
(same sea state, same posture of the person).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from seaswarm.config import SensorConfig

Array = NDArray[np.float64]


@dataclass(frozen=True)
class SensorModel:
    """POD model of one sensor at a given altitude and light condition."""

    pod_max: float
    sigma_r_m: float
    r_max_m: float
    period_s: float

    @classmethod
    def from_config(
        cls, config: SensorConfig, daylight: bool, altitude_m: float | None = None
    ) -> SensorModel:
        """Build from a `sensors` entry; drone cameras need their altitude.

        Downward cameras see out to h * tan(FOV / 2), and their peak POD falls with altitude:
        pod_max(h) = pod_max(h_ref) * exp(-(h - h_ref) / h_scale), clipped to [0, 1].
        """
        pod_ref = config.pod_max_day if daylight else config.pod_max_night
        if config.r_max_m is not None:
            return cls(pod_ref, config.sigma_r_m, config.r_max_m, config.period_s)
        if altitude_m is None or altitude_m <= 0.0:
            raise ValueError("a downward-looking camera needs a positive altitude_m")
        r_max_m = altitude_m * np.tan(np.radians(config.fov_deg) / 2.0)
        decay = np.exp(-(altitude_m - config.altitude_ref_m) / config.altitude_scale_m)
        pod_max = float(np.clip(pod_ref * decay, 0.0, 1.0))
        return cls(pod_max, config.sigma_r_m, float(r_max_m), config.period_s)

    def pod(self, r_m: ArrayLike) -> Array:
        """POD of one look at horizontal range r_m (scalar or array), metres."""
        r = np.asarray(r_m, dtype=float)
        pod = self.pod_max * np.exp(-(r**2) / (2.0 * self.sigma_r_m**2))
        return np.where(r <= self.r_max_m, pod, 0.0)

    def coverage_area_m2(self) -> float:
        """Integral of POD over the plane, m^2: the expected area 'cleared' by one look."""
        s2 = self.sigma_r_m**2
        return float(self.pod_max * 2.0 * np.pi * s2 * -np.expm1(-(self.r_max_m**2) / (2.0 * s2)))


def sweep_width_m(sensor: SensorModel, speed_mps: float, n_offsets: int = 200) -> float:
    """Effective sweep width W of a vehicle passing in a straight line, metres.

    A person at lateral offset x from the track is seen by the looks taken every `period_s` along
    the track; their cumulative detection probability is p(x) = 1 - prod_k (1 - POD(r_k)). The
    effective sweep width is W = integral of p(x) over all offsets: the width of a perfect sensor
    that would detect as many people as this one (Koopman's definition).
    """
    step_m = speed_mps * sensor.period_s
    n_looks = int(np.ceil(sensor.r_max_m / step_m))
    along = step_m * np.arange(-n_looks, n_looks + 1)[None, :]
    offsets = np.linspace(0.0, sensor.r_max_m, n_offsets)
    # average over where the person sits between two looks (looks are phase-shifted by u)
    phases = np.linspace(0.0, step_m, 8, endpoint=False)
    p_miss = np.mean(
        [np.prod(1.0 - sensor.pod(np.hypot(offsets[:, None], along + u)), axis=1) for u in phases],
        axis=0,
    )
    return float(2.0 * np.trapezoid(1.0 - p_miss, offsets))


def random_search_pod(sweep_width_m: float, track_length_m: float, area_m2: float) -> float:
    """Koopman's random-search formula: POD = 1 - exp(-W L / A) for one sweep of an area."""
    return float(-np.expm1(-sweep_width_m * track_length_m / area_m2))
