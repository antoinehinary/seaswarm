"""Motion model of the belief: how the probability map drifts and spreads over time.

The map knows the environment forecast (wind, current, leeway coefficients of the object class)
but never the true person. It moves with the mean drift and spreads with the exact variance of
the drift noise, derived from the same processes as the truth model (seaswarm.world.drift):

- leeway noise eps_d, eps_c: Ornstein-Uhlenbeck, std sigma, correlation time tau;
- crosswind side s(t) = +-1 flipping at rate lam: a telegraph process, autocorrelation
  exp(-2 lam |lag|), so the crosswind leeway s * L_c behaves like an OU velocity with std L_c and
  correlation time 1 / (2 lam); s * eps_c has correlation time 1 / (1/tau + 2 lam);
- current noise and gusts: OU.

For a stationary process with std sigma and correlation time tau, the displacement variance after
t seconds is exactly 2 sigma^2 tau (t - tau (1 - exp(-t / tau))): ballistic (sigma^2 t^2) for
t << tau, diffusive (2 sigma^2 tau t) for t >> tau. Using it instead of a constant diffusion
coefficient keeps the map's spread right during the first hour.

Approximations: the noise is treated as independent of the observations (Bayes updates do not
change it), the two crosswind lobes are replaced by one Gaussian of the same variance, and the
current used for the mean drift is taken at the map's mean position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from seaswarm.config import DefaultConfig, OUNoiseConfig, Scenario
from seaswarm.units import SECONDS_PER_HOUR
from seaswarm.world.drift import LeewayModel
from seaswarm.world.ocean import CurrentField, WindField, make_current

Array = NDArray[np.float64]


def correlated_displacement_variance(sigma: float, tau_s: float, t_s: float) -> float:
    """Variance (m^2) of the integral over [0, t] of a stationary velocity noise.

    Args:
        sigma: Standard deviation of the velocity noise, m/s.
        tau_s: Correlation time, s; `inf` for a constant random velocity (variance sigma^2 t^2).
        t_s: Integration time, s.
    """
    if np.isinf(tau_s):
        return sigma**2 * t_s**2
    return 2.0 * sigma**2 * tau_s * (t_s + tau_s * np.expm1(-t_s / tau_s))


def _combined_tau(*taus_s: float) -> float:
    """Correlation time of a product of independent processes (rates add)."""
    rate = sum(0.0 if np.isinf(t) else 1.0 / t for t in taus_s)
    return np.inf if rate == 0.0 else 1.0 / rate


@dataclass(frozen=True)
class BeliefMotion:
    """Mean drift and spread of the belief, from the scenario's forecast fields."""

    leeway: LeewayModel
    current: CurrentField
    wind: WindField
    current_noise: OUNoiseConfig | None = None
    gusts: OUNoiseConfig | None = None

    @classmethod
    def from_scenario(cls, scenario: Scenario, defaults: DefaultConfig) -> BeliefMotion:
        """Motion model for a scenario's search object, wind and current."""
        obj = defaults.leeway_objects[scenario.drift.object]
        return cls(
            LeewayModel.from_config(obj, scenario.drift.tau_s),
            make_current(scenario.current),
            WindField.from_config(scenario.wind),
            scenario.current.noise,
            scenario.wind.gusts,
        )

    @property
    def jibe_tau_s(self) -> float:
        """Correlation time 1 / (2 lam) of the crosswind side, s (inf without jibing)."""
        rate_per_s = -np.log1p(-self.leeway.jibe_probability_per_h) / SECONDS_PER_HOUR
        return np.inf if rate_per_s == 0.0 else 1.0 / (2.0 * rate_per_s)

    def mean_velocity_mps(self, t_s: float, at_m: Array) -> Array:
        """Mean drift (u, v) m/s at time t_s: current at `at_m` plus mean downwind leeway.

        The crosswind leeway has zero mean because the side is equally likely to be +1 or -1.
        """
        lw = self.leeway
        current = self.current.velocity(np.asarray(at_m, dtype=float).reshape(1, 2), t_s)[0]
        downwind = lw.downwind_slope * self.wind.speed_mps + lw.downwind_offset_mps
        return current + downwind * self.wind.downwind_unit(t_s)

    def axis_variances_m2(self, t_s: float) -> tuple[float, float]:
        """Downwind and crosswind displacement variances accumulated since the fall, m^2."""
        lw, var = self.leeway, correlated_displacement_variance
        tau_j = self.jibe_tau_s
        down = var(lw.downwind_sigma_mps, lw.tau_s, t_s)
        crosswind_mean_mps = lw.crosswind_slope * self.wind.speed_mps + lw.crosswind_offset_mps
        cross = var(crosswind_mean_mps, tau_j, t_s) + var(
            lw.crosswind_sigma_mps, _combined_tau(lw.tau_s, tau_j), t_s
        )
        if self.gusts is not None:
            g = self.gusts
            down += var(lw.downwind_slope * g.sigma_mps, g.tau_s, t_s)
            cross += var(lw.crosswind_slope * g.sigma_mps, _combined_tau(g.tau_s, tau_j), t_s)
        return down, cross

    def covariance_m2(self, t_s: float) -> Array:
        """2x2 ENU displacement covariance accumulated since the fall, m^2.

        Uses the wind direction at t_s; with a rotating wind this is an approximation.
        """
        down, cross = self.axis_variances_m2(t_s)
        e_dw = self.wind.downwind_unit(t_s)
        e_cw = np.array([e_dw[1], -e_dw[0]])
        cov = down * np.outer(e_dw, e_dw) + cross * np.outer(e_cw, e_cw)
        if self.current_noise is not None:
            n = self.current_noise
            cov += correlated_displacement_variance(n.sigma_mps, n.tau_s, t_s) * np.eye(2)
        return cov

    def covariance_increment_m2(self, t0_s: float, t1_s: float) -> Array:
        """Covariance to add when predicting from t0_s to t1_s (since the fall), m^2.

        Differences of the cumulative variance along the current wind axes, so the sum of the
        increments reproduces the exact variance growth.
        """
        down0, cross0 = self.axis_variances_m2(t0_s)
        down1, cross1 = self.axis_variances_m2(t1_s)
        e_dw = self.wind.downwind_unit(0.5 * (t0_s + t1_s))
        e_cw = np.array([e_dw[1], -e_dw[0]])
        cov = (down1 - down0) * np.outer(e_dw, e_dw) + (cross1 - cross0) * np.outer(e_cw, e_cw)
        if self.current_noise is not None:
            n = self.current_noise
            v0 = correlated_displacement_variance(n.sigma_mps, n.tau_s, t0_s)
            v1 = correlated_displacement_variance(n.sigma_mps, n.tau_s, t1_s)
            cov += (v1 - v0) * np.eye(2)
        return cov
