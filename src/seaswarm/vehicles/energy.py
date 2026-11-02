"""Battery / fuel model and return-to-base reserve.

Energy is a fraction of a full battery or tank, drained at

    rate = (idle + (1 - idle) (v / v_cruise)^2) / endurance        [1/s]

which equals 1 / endurance at cruise speed. `idle` is the share of cruise power needed at zero
speed: high for a multirotor (hovering is expensive), low for a boat.

Return-to-base reserve: the vehicle must keep enough energy to fly home at cruise speed with a
margin, plus a fixed allowance for approach and landing:

    reserve_ok  <=>  energy * endurance >= t_rtb * (1 + margin) + t_fixed,   t_rtb = d / v_cruise
"""

from __future__ import annotations

from seaswarm.config import VehicleTypeConfig


def drain_rate_per_s(speed_mps: float, config: VehicleTypeConfig) -> float:
    """Fraction of a full battery used per second at a given speed."""
    load = (
        config.idle_power_fraction
        + (1.0 - config.idle_power_fraction) * (speed_mps / config.cruise_speed_mps) ** 2
    )
    return load / config.endurance_s


def rtb_energy_frac(distance_to_base_m: float, config: VehicleTypeConfig) -> float:
    """Energy fraction the reserve rule requires to return from `distance_to_base_m`."""
    t_rtb_s = distance_to_base_m / config.cruise_speed_mps
    return (t_rtb_s * (1.0 + config.rtb_margin) + config.rtb_fixed_s) / config.endurance_s


def reserve_ok(energy_frac: float, distance_to_base_m: float, config: VehicleTypeConfig) -> bool:
    """True if the vehicle can still return to base with the required reserve."""
    return energy_frac >= rtb_energy_frac(distance_to_base_m, config)
