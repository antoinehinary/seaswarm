"""Drift of a person in water: surface current plus wind-driven leeway (US Coast Guard method).

Velocity of each particle:

    v = U_current(x) + n_current + L_d * e_dw + s * L_c * e_cw
    L_d = a_d * |W10| + b_d + eps_d
    L_c = a_c * |W10| + b_c + eps_c

e_dw is the downwind unit vector, e_cw the unit vector 90 deg to the right of downwind, s = +-1 the
crosswind side, which flips ("jibes") at random. eps_d, eps_c and the optional current noise
n_current and gust on |W10| are Ornstein-Uhlenbeck processes, independent per particle.

Why OU and not white noise: white noise drawn every 1 s step averages out within minutes (the
spread after time T grows like sigma * sqrt(dt * T)), giving almost no dispersion. Real leeway
errors persist for tens of minutes (sea state, posture), so the noise must be correlated in time:
OU with correlation time tau gives a spread that grows like sigma * sqrt(2 * tau * T) at long times.

Integration is explicit Euler at dt_s; with the default dt of 1 s, positions are accurate to well
below a grid cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from seaswarm.config import DefaultConfig, LeewayObjectConfig, OUNoiseConfig, Scenario
from seaswarm.units import SECONDS_PER_HOUR
from seaswarm.world.ocean import CurrentField, OUProcess, WindField, make_current

Array = NDArray[np.float64]


@dataclass(frozen=True)
class LeewayModel:
    """Leeway coefficients of one search object (SI) and the correlation time of its noise."""

    downwind_slope: float
    downwind_offset_mps: float
    downwind_sigma_mps: float
    crosswind_slope: float
    crosswind_offset_mps: float
    crosswind_sigma_mps: float
    jibe_probability_per_h: float
    tau_s: float

    @classmethod
    def from_config(cls, obj: LeewayObjectConfig, tau_s: float) -> LeewayModel:
        """Build from a `leeway_objects` entry of the default config."""
        return cls(
            obj.downwind_slope,
            obj.downwind_offset_mps,
            obj.downwind_sigma_mps,
            obj.crosswind_slope,
            obj.crosswind_offset_mps,
            obj.crosswind_sigma_mps,
            obj.jibe_probability_per_h,
            tau_s,
        )

    @classmethod
    def simple(
        cls, k: float = 0.0096, sigma_mps: float = 0.0, tau_s: float = 1800.0
    ) -> LeewayModel:
        """Textbook form: leeway = k * W10 downwind (+ optional noise), no crosswind, no jibing."""
        return cls(k, 0.0, sigma_mps, 0.0, 0.0, 0.0, 0.0, tau_s)

    def jibe_probability(self, dt_s: float) -> float:
        """Probability that the crosswind side flips during one step of dt_s seconds."""
        return 1.0 - (1.0 - self.jibe_probability_per_h) ** (dt_s / SECONDS_PER_HOUR)


class ParticleDrift:
    """N drifting particles, each an independent realisation of the noise processes.

    The truth person is the case N = 1; a particle cloud (N in the hundreds or thousands)
    samples the distribution of the person's position. All randomness comes from the one
    generator passed in, so a given seed always gives the same trajectories.
    """

    def __init__(
        self,
        positions_m: Array,
        leeway: LeewayModel,
        current: CurrentField,
        wind: WindField,
        rng: np.random.Generator,
        *,
        current_noise: OUNoiseConfig | None = None,
        gusts: OUNoiseConfig | None = None,
        t0_s: float = 0.0,
    ) -> None:
        """Set up the particles and draw every noise process from its stationary distribution.

        Args:
            positions_m: Initial ENU positions, shape (N, 2), metres.
            leeway: Leeway coefficients of the search object.
            current: Mean surface current field.
            wind: Mean 10 m wind field.
            rng: Generator for all drift randomness (the scenario's `drift` stream).
            current_noise: Optional OU perturbation added to each current component.
            gusts: Optional OU perturbation of the wind speed.
            t0_s: Simulation time of the initial positions, s.
        """
        self._xy = np.array(positions_m, dtype=float).reshape(-1, 2)
        n = len(self._xy)
        self.leeway = leeway
        self.current = current
        self.wind = wind
        self.t_s = t0_s
        self._rng = rng
        self._side = rng.choice([-1.0, 1.0], size=n)
        self._eps_d = OUProcess.stationary(leeway.downwind_sigma_mps, leeway.tau_s, n, rng)
        self._eps_c = OUProcess.stationary(leeway.crosswind_sigma_mps, leeway.tau_s, n, rng)
        self._current_noise = (
            OUProcess.stationary(current_noise.sigma_mps, current_noise.tau_s, (n, 2), rng)
            if current_noise
            else None
        )
        self._gust = OUProcess.stationary(gusts.sigma_mps, gusts.tau_s, n, rng) if gusts else None

    @classmethod
    def from_scenario(
        cls,
        scenario: Scenario,
        defaults: DefaultConfig,
        positions_m: Array,
        rng: np.random.Generator,
        t0_s: float = 0.0,
    ) -> ParticleDrift:
        """Build the drift of a scenario's search object in that scenario's wind and current."""
        obj = defaults.leeway_objects[scenario.drift.object]
        return cls(
            positions_m,
            LeewayModel.from_config(obj, scenario.drift.tau_s),
            make_current(scenario.current),
            WindField.from_config(scenario.wind),
            rng,
            current_noise=scenario.current.noise,
            gusts=scenario.wind.gusts,
            t0_s=t0_s,
        )

    @property
    def positions_m(self) -> Array:
        """Copy of the current ENU positions, shape (N, 2), metres."""
        return self._xy.copy()

    @property
    def crosswind_side(self) -> Array:
        """Copy of the crosswind side of each particle: +1 right of downwind, -1 left."""
        return self._side.copy()

    def wind_speed_mps(self) -> Array:
        """Per-particle 10 m wind speed |W10| including gusts (never negative), m/s."""
        speed = np.full(len(self._xy), self.wind.speed_mps)
        if self._gust is not None:
            speed = np.maximum(speed + self._gust.value, 0.0)
        return speed

    def leeway_velocity(self) -> Array:
        """Leeway velocity (N, 2) m/s relative to the water, at the current state."""
        lw = self.leeway
        e_dw = self.wind.downwind_unit(self.t_s)
        e_cw = np.array([e_dw[1], -e_dw[0]])  # e_dw rotated -90 deg: right of downwind
        w10 = self.wind_speed_mps()
        l_d = lw.downwind_slope * w10 + lw.downwind_offset_mps + self._eps_d.value
        l_c = lw.crosswind_slope * w10 + lw.crosswind_offset_mps + self._eps_c.value
        return l_d[:, None] * e_dw + (self._side * l_c)[:, None] * e_cw

    def velocity(self) -> Array:
        """Total drift velocity (N, 2) m/s over ground, at the current state."""
        v = self.current.velocity(self._xy, self.t_s) + self.leeway_velocity()
        if self._current_noise is not None:
            v += self._current_noise.value
        return v

    def step(self, dt_s: float) -> None:
        """Advance every particle by dt_s seconds (explicit Euler), then update the noise."""
        self._xy += self.velocity() * dt_s
        self.t_s += dt_s
        for process in (self._eps_d, self._eps_c, self._current_noise, self._gust):
            if process is not None:
                process.step(dt_s, self._rng)
        jibes = self._rng.random(len(self._xy)) < self.leeway.jibe_probability(dt_s)
        self._side[jibes] *= -1.0


def simulate(
    drift: ParticleDrift, duration_s: float, dt_s: float, record_every_s: float, max_steps: int
) -> tuple[Array, Array]:
    """Run a drift forward and record the positions at a fixed interval.

    Args:
        drift: Particles to advance (modified in place).
        duration_s: Simulated time to run, s. Must be a whole number of steps.
        dt_s: Integration step, s.
        record_every_s: Recording interval, s. Must be a whole number of steps.
        max_steps: Hard limit on the number of steps; exceeding it raises before running.

    Returns:
        (times_s, positions_m): times shape (F,), positions shape (F, N, 2), the first frame
        being the initial state.

    Raises:
        ValueError: If the duration or interval is not a whole number of steps, or the run
            would exceed max_steps.
    """
    n_steps = _whole_steps(duration_s, dt_s, "duration_s")
    every = _whole_steps(record_every_s, dt_s, "record_every_s")
    if n_steps > max_steps:
        raise ValueError(f"{n_steps} steps of {dt_s} s exceed the limit of {max_steps} steps")
    times = [drift.t_s]
    frames = [drift.positions_m]
    for i in range(1, n_steps + 1):
        drift.step(dt_s)
        if i % every == 0:
            times.append(drift.t_s)
            frames.append(drift.positions_m)
    return np.array(times), np.stack(frames)


def _whole_steps(duration_s: float, dt_s: float, name: str) -> int:
    steps = round(duration_s / dt_s)
    if steps < 1 or abs(steps * dt_s - duration_s) > 1e-9 * max(1.0, duration_s):
        raise ValueError(f"{name} ({duration_s} s) is not a whole number of {dt_s} s steps")
    return steps
