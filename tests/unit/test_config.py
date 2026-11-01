from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from seaswarm.config import (
    GyreCurrentConfig,
    LinearGradientCurrentConfig,
    Scenario,
    load_default_config,
    load_scenario,
)

SCENARIOS = Path(__file__).resolve().parents[2] / "scenarios"


def _raw(name: str = "calm_single") -> dict:
    return yaml.safe_load((SCENARIOS / f"{name}.yaml").read_text())


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_default_config_has_piw1_coefficients() -> None:
    piw = load_default_config().leeway_objects["PIW-1"]
    assert piw.downwind_slope == 0.0096
    assert piw.crosswind_slope == 0.0054
    assert piw.jibe_probability_per_h == 0.04


@pytest.mark.parametrize("name", ["calm_single", "windy_mixed"])
def test_demo_scenarios_load(name: str) -> None:
    assert load_scenario(SCENARIOS / f"{name}.yaml").id == name


def test_current_type_selects_the_model() -> None:
    assert isinstance(
        load_scenario(SCENARIOS / "windy_mixed.yaml").current, LinearGradientCurrentConfig
    )
    raw = _raw()
    raw["current"] = {
        "type": "gyre",
        "center_x_m": 0,
        "center_y_m": 0,
        "radius_m": 2000,
        "max_speed_mps": 0.5,
    }
    assert isinstance(Scenario.model_validate(raw).current, GyreCurrentConfig)


def test_unknown_key_is_rejected() -> None:
    raw = _raw()
    raw["wind"]["speed_kn"] = 10
    with pytest.raises(ValidationError, match="speed_kn"):
        Scenario.model_validate(raw)


def test_unknown_current_type_is_rejected() -> None:
    raw = _raw()
    raw["current"]["type"] = "tidal"
    with pytest.raises(ValidationError, match="tidal"):
        Scenario.model_validate(raw)


def test_negative_wind_speed_is_rejected() -> None:
    raw = _raw()
    raw["wind"]["speed_mps"] = -1.0
    with pytest.raises(ValidationError, match="speed_mps"):
        Scenario.model_validate(raw)


def test_lkp_outside_area_is_rejected() -> None:
    raw = _raw()
    raw["lkp"]["y_m"] = 6000
    with pytest.raises(ValidationError, match="outside"):
        Scenario.model_validate(raw)


def test_area_must_be_whole_cells() -> None:
    raw = _raw()
    raw["area"]["cell_size_m"] = 300
    with pytest.raises(ValidationError, match="whole number"):
        Scenario.model_validate(raw)


def test_unknown_drift_object_is_rejected(tmp_path: Path) -> None:
    raw = _raw()
    raw["drift"]["object"] = "PIW-99"
    with pytest.raises(ValueError, match="PIW-99"):
        load_scenario(_write(tmp_path, raw))
