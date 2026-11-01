"""Surface current and 10 m wind fields, and the Ornstein-Uhlenbeck noise used on top of them.

Fields are deterministic mean fields. Random perturbations (current noise, gusts, leeway noise)
are `OUProcess` instances owned per particle by the drift model, so that every particle is an
independent realisation of the uncertain world.

Positions are (N, 2) arrays of ENU metres; velocities are (N, 2) arrays of (u, v) m/s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from seaswarm.config import (
    CurrentConfig,
    GyreCurrentConfig,
    LinearGradientCurrentConfig,
    UniformCurrentConfig,
    WindConfig,
)
from seaswarm.units import SECONDS_PER_HOUR, from_vector, towards_vector

Array = NDArray[np.float64]


# --------------------------------------------------------------------------- OU noise


@dataclass
class OUProcess:
    """Ornstein-Uhlenbeck process, one independent component per array element.

    dx = -x / tau dt + sigma * sqrt(2 / tau) dW, with stationary distribution N(0, sigma^2) and
    autocorrelation exp(-|lag| / tau). Stepped with the exact discretisation, so the statistics do
    not depend on dt.

    Attributes:
        sigma: Stationary standard deviation (units of the perturbed quantity).
        tau_s: Correlation time, s.
        value: Current value of every component.
    """

    sigma: float
    tau_s: float
    value: Array

    @classmethod
    def stationary(
        cls, sigma: float, tau_s: float, shape: int | tuple[int, ...], rng: np.random.Generator
    ) -> OUProcess:
        """Start from a draw of the stationary distribution N(0, sigma^2)."""
        return cls(sigma, tau_s, sigma * rng.standard_normal(shape))

    def step(self, dt_s: float, rng: np.random.Generator) -> Array:
        """Advance by dt_s seconds and return the new value."""
        decay = np.exp(-dt_s / self.tau_s)
        kick = self.sigma * np.sqrt(1.0 - decay**2)
        self.value = decay * self.value + kick * rng.standard_normal(self.value.shape)
        return self.value


# --------------------------------------------------------------------------- currents


class CurrentField(Protocol):
    """Mean surface current, (u, v) "towards" components."""

    def velocity(self, xy_m: Array, t_s: float) -> Array:
        """Current velocity (N, 2) m/s at positions xy_m (N, 2) m and time t_s."""
        ...


def _unit_towards(to_bearing_deg: float) -> Array:
    return np.array(towards_vector(1.0, to_bearing_deg))


@dataclass(frozen=True)
class UniformCurrent:
    """Same current everywhere."""

    u_mps: float
    v_mps: float

    def velocity(self, xy_m: Array, t_s: float) -> Array:
        """Current velocity (N, 2) m/s; independent of position and time."""
        return np.broadcast_to([self.u_mps, self.v_mps], xy_m.shape).copy()


@dataclass(frozen=True)
class LinearGradientCurrent:
    """Shear current: constant direction, speed varying linearly across the flow.

    speed(x) = speed_mps + shear_per_s * (x . n), n the unit vector 90 deg to the left of the flow.
    The speed gradient is perpendicular to the flow, so the divergence is zero.
    """

    speed_mps: float
    to_bearing_deg: float
    shear_per_s: float

    def velocity(self, xy_m: Array, t_s: float) -> Array:
        """Current velocity (N, 2) m/s at positions xy_m (N, 2) m."""
        e = _unit_towards(self.to_bearing_deg)
        left = np.array([-e[1], e[0]])  # e rotated +90 deg (counter-clockwise)
        speed = self.speed_mps + self.shear_per_s * (xy_m @ left)
        return speed[:, None] * e


@dataclass(frozen=True)
class GyreCurrent:
    """Gaussian eddy from the stream function psi = A exp(-r^2 / (2 R^2)).

    u = -dpsi/dy, v = dpsi/dx, so div(u, v) = 0 exactly. The speed is |A| r / R^2 exp(-r^2/(2R^2)),
    largest at r = R, where it equals max_speed_mps. A > 0 turns clockwise.
    """

    center_x_m: float
    center_y_m: float
    radius_m: float
    max_speed_mps: float
    clockwise: bool

    def velocity(self, xy_m: Array, t_s: float) -> Array:
        """Current velocity (N, 2) m/s at positions xy_m (N, 2) m."""
        r_m = self.radius_m
        amplitude = (1.0 if self.clockwise else -1.0) * self.max_speed_mps * r_m * np.exp(0.5)
        dx = xy_m[:, 0] - self.center_x_m
        dy = xy_m[:, 1] - self.center_y_m
        g = amplitude / r_m**2 * np.exp(-(dx**2 + dy**2) / (2.0 * r_m**2))
        return np.column_stack([g * dy, -g * dx])


def make_current(config: CurrentConfig) -> CurrentField:
    """Build the mean current field described by a scenario."""
    match config:
        case UniformCurrentConfig():
            return UniformCurrent(*towards_vector(config.speed_mps, config.to_bearing_deg))
        case LinearGradientCurrentConfig():
            return LinearGradientCurrent(
                config.speed_mps, config.to_bearing_deg, config.shear_per_s
            )
        case GyreCurrentConfig():
            return GyreCurrent(
                config.center_x_m,
                config.center_y_m,
                config.radius_m,
                config.max_speed_mps,
                config.clockwise,
            )
    raise TypeError(f"unknown current config {type(config).__name__}")


# --------------------------------------------------------------------------- wind


@dataclass(frozen=True)
class WindField:
    """Spatially uniform mean 10 m wind whose direction may rotate at a constant rate.

    Attributes:
        speed_mps: Mean wind speed, m/s.
        from_bearing_deg: Direction the wind blows from at t = 0, compass degrees.
        rotation_deg_per_h: Rate of change of the from-bearing; > 0 veers, < 0 backs.
    """

    speed_mps: float
    from_bearing_deg: float
    rotation_deg_per_h: float = 0.0

    @classmethod
    def from_config(cls, config: WindConfig) -> WindField:
        """Build the mean wind of a scenario (gusts are handled per particle by the drift)."""
        return cls(config.speed_mps, config.from_bearing_deg, config.rotation_deg_per_h)

    def from_bearing_at(self, t_s: float) -> float:
        """Direction the wind blows from at time t_s, compass degrees in [0, 360)."""
        return (self.from_bearing_deg + self.rotation_deg_per_h * t_s / SECONDS_PER_HOUR) % 360.0

    def downwind_unit(self, t_s: float) -> Array:
        """Unit (u, v) vector pointing downwind at time t_s (defined even at zero speed)."""
        return np.array(from_vector(1.0, self.from_bearing_at(t_s)))

    def velocity(self, t_s: float) -> Array:
        """Mean wind (u, v) m/s at time t_s."""
        return self.speed_mps * self.downwind_unit(t_s)
