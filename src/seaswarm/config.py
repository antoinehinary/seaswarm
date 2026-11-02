"""Typed configuration: project defaults (config/default.yaml) and scenarios (scenarios/*.yaml).

Models forbid unknown keys, so a typo in a YAML file fails loudly at start-up instead of silently
falling back to a default. All quantities are SI; bearings are compass degrees (clockwise from
north) and are converted to ENU vectors by `seaswarm.units`.

Frame: local ENU in metres, origin at the centre of the search area. The base is a point in that
frame and may lie outside the area.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- shared pieces


class Point(_Model):
    """A position in the local ENU frame, metres."""

    x_m: float
    y_m: float


class OUNoiseConfig(_Model):
    """Ornstein-Uhlenbeck noise: stationary standard deviation and correlation time."""

    sigma_mps: NonNegativeFloat
    tau_s: PositiveFloat


# --------------------------------------------------------------------------- default.yaml


class SimConfig(_Model):
    """Simulation stepping (default.yaml `sim`)."""

    dt_s: PositiveFloat


class LeewayObjectConfig(_Model):
    """Leeway coefficients of one search-object class, in SI (default.yaml `leeway_objects`).

    Leeway speed = slope * |W10| + offset, with W10 the 10 m wind speed. The crosswind term is
    symmetric (same slope on both sides of downwind), as for PIW-1.
    """

    description: str
    downwind_slope: NonNegativeFloat
    downwind_offset_mps: float
    downwind_sigma_mps: NonNegativeFloat
    crosswind_slope: NonNegativeFloat
    crosswind_offset_mps: float
    crosswind_sigma_mps: NonNegativeFloat
    jibe_probability_per_h: Probability


class BeliefConfig(_Model):
    """Probability-map parameters (default.yaml `belief`)."""

    predict_period_s: PositiveFloat
    pod_points_per_r_max: PositiveInt


class SensorConfig(_Model):
    """Lateral-range sensor (default.yaml `sensors`).

    Downward-looking drone cameras set `fov_deg` (their range limit grows with altitude);
    surface sensors set a fixed `r_max_m`. Exactly one of the two must be given.
    """

    sigma_r_m: PositiveFloat
    pod_max_day: Probability
    pod_max_night: Probability
    period_s: PositiveFloat
    fov_deg: Annotated[float, Field(gt=0.0, lt=180.0)] | None = None
    altitude_ref_m: PositiveFloat | None = None
    altitude_scale_m: PositiveFloat | None = None
    r_max_m: PositiveFloat | None = None

    @model_validator(mode="after")
    def _one_range_model(self) -> Self:
        camera = (self.fov_deg, self.altitude_ref_m, self.altitude_scale_m)
        if self.r_max_m is None and None in camera:
            raise ValueError("set r_max_m, or fov_deg + altitude_ref_m + altitude_scale_m")
        if self.r_max_m is not None and any(v is not None for v in camera):
            raise ValueError("set either r_max_m or the camera fields, not both")
        return self


class VehicleTypeConfig(_Model):
    """Kinematics, energy and operations of one vehicle type (default.yaml `vehicles`)."""

    cruise_speed_mps: PositiveFloat
    max_speed_mps: PositiveFloat
    max_turn_rate_deg_s: PositiveFloat
    max_accel_mps2: PositiveFloat
    heading_gain_per_s: PositiveFloat
    speed_time_constant_s: PositiveFloat
    waypoint_radius_m: PositiveFloat
    endurance_s: PositiveFloat
    idle_power_fraction: Probability
    altitude_layers_m: list[NonNegativeFloat]
    search_altitude_m: NonNegativeFloat
    sensor: str
    rtb_margin: NonNegativeFloat
    rtb_fixed_s: NonNegativeFloat
    turnaround_s: NonNegativeFloat
    detection_sigma_m: PositiveFloat
    recovery_radius_m: PositiveFloat | None = None

    @model_validator(mode="after")
    def _speeds(self) -> Self:
        if self.cruise_speed_mps > self.max_speed_mps:
            raise ValueError("cruise_speed_mps exceeds max_speed_mps")
        if not self.altitude_layers_m:
            raise ValueError("altitude_layers_m needs at least one layer")
        return self


class VehiclesConfig(_Model):
    """The two vehicle types of the fleet."""

    drone: VehicleTypeConfig
    usv: VehicleTypeConfig


class DefaultConfig(_Model):
    """Project-wide parameters (config/default.yaml)."""

    sim: SimConfig
    leeway_objects: dict[str, LeewayObjectConfig]
    belief: BeliefConfig
    sensors: dict[str, SensorConfig]
    vehicles: VehiclesConfig


# --------------------------------------------------------------------------- scenario fields


class AreaConfig(_Model):
    """Square-celled search area centred on the origin."""

    width_m: PositiveFloat
    height_m: PositiveFloat
    cell_size_m: PositiveFloat

    @model_validator(mode="after")
    def _whole_cells(self) -> Self:
        for name, size_m in (("width_m", self.width_m), ("height_m", self.height_m)):
            cells = size_m / self.cell_size_m
            if abs(cells - round(cells)) > 1e-9:
                raise ValueError(
                    f"{name} ({size_m} m) is not a whole number of {self.cell_size_m} m cells"
                )
        return self

    def contains(self, x_m: float, y_m: float) -> bool:
        """True if (x_m, y_m) lies inside the area (boundary included)."""
        return abs(x_m) <= self.width_m / 2 and abs(y_m) <= self.height_m / 2


class LkpConfig(_Model):
    """Last known position of the person and its uncertainty."""

    x_m: float
    y_m: float
    sigma_m: PositiveFloat
    time_since_fall_s: NonNegativeFloat


class WindConfig(_Model):
    """Spatially uniform 10 m wind, optionally rotating slowly and gusting."""

    type: Literal["uniform"]
    speed_mps: NonNegativeFloat
    from_bearing_deg: float
    rotation_deg_per_h: float = 0.0  # > 0 veers (from-bearing turns clockwise), < 0 backs
    gusts: OUNoiseConfig | None = None


class UniformCurrentConfig(_Model):
    """Same surface current everywhere."""

    type: Literal["uniform"]
    speed_mps: NonNegativeFloat
    to_bearing_deg: float
    noise: OUNoiseConfig | None = None


class LinearGradientCurrentConfig(_Model):
    """Shear current: speed varies linearly across the flow, direction is constant.

    Speed at a point = speed_mps + shear_per_s * (signed distance to the left of the flow line
    through the origin). The gradient is perpendicular to the flow, so the field is divergence-free.
    """

    type: Literal["linear_gradient"]
    speed_mps: float
    to_bearing_deg: float
    shear_per_s: float
    noise: OUNoiseConfig | None = None


class GyreCurrentConfig(_Model):
    """Gaussian eddy derived from a stream function (divergence-free by construction)."""

    type: Literal["gyre"]
    center_x_m: float
    center_y_m: float
    radius_m: PositiveFloat  # radius of maximum speed
    max_speed_mps: NonNegativeFloat
    clockwise: bool = False
    noise: OUNoiseConfig | None = None


CurrentConfig = Annotated[
    UniformCurrentConfig | LinearGradientCurrentConfig | GyreCurrentConfig,
    Field(discriminator="type"),
]


class DriftConfig(_Model):
    """Which leeway object class drifts, and the correlation time of its leeway noise."""

    object: str
    tau_s: PositiveFloat


class FleetConfig(_Model):
    """Number of vehicles of each type."""

    drones_eo: NonNegativeInt
    drones_thermal: NonNegativeInt
    usvs: NonNegativeInt


class LimitsConfig(_Model):
    """Hard limits on one simulation run."""

    max_sim_time_s: PositiveFloat
    max_steps: PositiveInt


class Scenario(_Model):
    """One search scenario (scenarios/*.yaml)."""

    id: str
    seed: NonNegativeInt
    area: AreaConfig
    base: Point
    lkp: LkpConfig
    wind: WindConfig
    current: CurrentConfig
    drift: DriftConfig
    daylight: bool
    fleet: FleetConfig
    limits: LimitsConfig

    @model_validator(mode="after")
    def _lkp_inside_area(self) -> Self:
        if not self.area.contains(self.lkp.x_m, self.lkp.y_m):
            raise ValueError(
                f"last known position ({self.lkp.x_m}, {self.lkp.y_m}) m is outside the "
                f"{self.area.width_m} x {self.area.height_m} m search area"
            )
        return self


# --------------------------------------------------------------------------- loaders


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_default_config(path: Path | None = None) -> DefaultConfig:
    """Load and validate the project defaults (default: `config/default.yaml`)."""
    return DefaultConfig.model_validate(_load_yaml(path or CONFIG_DIR / "default.yaml"))


def load_scenario(path: Path, defaults: DefaultConfig | None = None) -> Scenario:
    """Load and validate a scenario file.

    Args:
        path: Scenario YAML file.
        defaults: Project defaults used to check that the drift object exists (loaded from
            `config/default.yaml` if omitted).

    Raises:
        pydantic.ValidationError: If the file does not match the schema.
        ValueError: If the drift object is not defined in the defaults.
    """
    scenario = Scenario.model_validate(_load_yaml(path))
    known = (defaults or load_default_config()).leeway_objects
    if scenario.drift.object not in known:
        raise ValueError(
            f"scenario {scenario.id!r} uses drift object {scenario.drift.object!r}, "
            f"not one of {sorted(known)}"
        )
    return scenario
