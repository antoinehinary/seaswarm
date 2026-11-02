"""Unicycle kinematics with heading-rate, speed and acceleration limits.

Continuous model:

    psi_dot = clip(k_psi * wrap(psi_cmd - psi), -r_max, r_max)
    v_dot   = clip((v_cmd - v) / tau_v, -a_max, a_max),   0 <= v <= v_max
    x_dot   = v cos(psi) + u_drift,   y_dot = v sin(psi) + v_drift

The two first-order loops are discretised exactly: over a step dt the heading error shrinks by
exp(-k_psi dt) and the speed error by exp(-dt / tau_v), then the rate and acceleration limits
apply. A forward-Euler step would give err[n+1] = (1 - k dt) err[n], which oscillates for
k dt > 1 and never settles at k dt = 2 (the drone's k = 2 /s at dt = 1 s did exactly that).

`drift` is the surface current for a USV and zero for a drone (wind on drones is ignored in v1).
Heading and speed are updated first, then the position (semi-implicit Euler).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seaswarm.config import VehicleTypeConfig
from seaswarm.units import wrap_angle_rad


@dataclass(frozen=True)
class KinematicLimits:
    """Limits and gains of the unicycle model, SI units."""

    max_speed_mps: float
    max_turn_rate_rad_s: float
    max_accel_mps2: float
    heading_gain_per_s: float
    speed_time_constant_s: float

    @classmethod
    def from_config(cls, config: VehicleTypeConfig) -> KinematicLimits:
        """Limits of a vehicle type."""
        return cls(
            config.max_speed_mps,
            np.radians(config.max_turn_rate_deg_s),
            config.max_accel_mps2,
            config.heading_gain_per_s,
            config.speed_time_constant_s,
        )

    def min_turn_radius_m(self, speed_mps: float) -> float:
        """Tightest turn at a given speed, metres."""
        return speed_mps / self.max_turn_rate_rad_s


@dataclass(frozen=True)
class Pose:
    """Planar pose and speed: ENU metres, yaw rad (CCW from east), m/s."""

    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float


def step_unicycle(
    pose: Pose,
    yaw_cmd_rad: float,
    speed_cmd_mps: float,
    limits: KinematicLimits,
    dt_s: float,
    drift_mps: tuple[float, float] = (0.0, 0.0),
) -> Pose:
    """Advance a pose by dt_s seconds towards a commanded heading and speed."""
    yaw_error = wrap_angle_rad(yaw_cmd_rad - pose.yaw_rad)
    yaw_rate = np.clip(
        yaw_error * -np.expm1(-limits.heading_gain_per_s * dt_s) / dt_s,
        -limits.max_turn_rate_rad_s,
        limits.max_turn_rate_rad_s,
    )
    accel = np.clip(
        (speed_cmd_mps - pose.speed_mps) * -np.expm1(-dt_s / limits.speed_time_constant_s) / dt_s,
        -limits.max_accel_mps2,
        limits.max_accel_mps2,
    )
    yaw = float(wrap_angle_rad(pose.yaw_rad + yaw_rate * dt_s))
    speed = float(np.clip(pose.speed_mps + accel * dt_s, 0.0, limits.max_speed_mps))
    x = pose.x_m + (speed * np.cos(yaw) + drift_mps[0]) * dt_s
    y = pose.y_m + (speed * np.sin(yaw) + drift_mps[1]) * dt_s
    return Pose(float(x), float(y), yaw, speed)


def heading_to(pose: Pose, x_m: float, y_m: float) -> float:
    """Yaw (rad) pointing from the pose to (x_m, y_m)."""
    return float(np.arctan2(y_m - pose.y_m, x_m - pose.x_m))
