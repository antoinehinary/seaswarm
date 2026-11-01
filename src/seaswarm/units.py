"""Unit conversions, angle conventions and constants.

Internal frame: local ENU, origin at the base, x east, y north, metres. Yaw is in radians,
counter-clockwise from +x (east). Compass bearings are in degrees, clockwise from north, and only
appear at the user-facing edges (scenario files, agent replies).

Vector fields are stored as (u, v) "towards" components: u east, v north, m/s. Meteorology gives
wind as the bearing it blows *from*, oceanography gives current as the bearing it flows *towards*;
both are converted here, once.

All functions accept scalars or numpy arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

KNOT_MPS = 1852.0 / 3600.0  # 1 knot = 1 nautical mile (1852 m, exact) per hour
SECONDS_PER_HOUR = 3600.0

FloatOrArray = float | NDArray[np.float64]
UV = tuple[FloatOrArray, FloatOrArray]  # (u east, v north) components


def knots_to_mps(speed_kn: ArrayLike) -> FloatOrArray:
    """Convert a speed from knots to m/s."""
    return np.asarray(speed_kn, dtype=float) * KNOT_MPS


def mps_to_knots(speed_mps: ArrayLike) -> FloatOrArray:
    """Convert a speed from m/s to knots."""
    return np.asarray(speed_mps, dtype=float) / KNOT_MPS


def wrap_angle_rad(angle_rad: ArrayLike) -> FloatOrArray:
    """Wrap an angle to [-pi, pi) radians."""
    return (np.asarray(angle_rad, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def bearing_deg_to_yaw_rad(bearing_deg: ArrayLike) -> FloatOrArray:
    """Convert a compass bearing (deg, clockwise from north) to ENU yaw (rad, CCW from east).

    Returns:
        Yaw in [-pi, pi) radians. North (0 deg) is +pi/2, east (90 deg) is 0.
    """
    return wrap_angle_rad(np.radians(90.0 - np.asarray(bearing_deg, dtype=float)))


def yaw_rad_to_bearing_deg(yaw_rad: ArrayLike) -> FloatOrArray:
    """Convert ENU yaw (rad, CCW from east) to a compass bearing in [0, 360) degrees."""
    return (90.0 - np.degrees(np.asarray(yaw_rad, dtype=float))) % 360.0


def towards_vector(speed_mps: ArrayLike, to_bearing_deg: ArrayLike) -> UV:
    """ENU (u, v) components of a flow moving *towards* a compass bearing.

    Args:
        speed_mps: Flow speed, m/s.
        to_bearing_deg: Bearing the flow moves towards, degrees clockwise from north.

    Returns:
        (u_mps, v_mps): east and north components, m/s.
    """
    speed = np.asarray(speed_mps, dtype=float)
    bearing_rad = np.radians(np.asarray(to_bearing_deg, dtype=float))
    return speed * np.sin(bearing_rad), speed * np.cos(bearing_rad)


def from_vector(speed_mps: ArrayLike, from_bearing_deg: ArrayLike) -> UV:
    """ENU (u, v) "towards" components of a flow given by the bearing it comes *from*.

    Meteorological convention: wind "from 270 deg" blows from the west, so it moves towards the
    east and u > 0.

    Args:
        speed_mps: Flow speed, m/s.
        from_bearing_deg: Bearing the flow comes from, degrees clockwise from north.

    Returns:
        (u_mps, v_mps): east and north components, m/s.
    """
    return towards_vector(speed_mps, np.asarray(from_bearing_deg, dtype=float) + 180.0)
