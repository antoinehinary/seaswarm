import numpy as np
import pytest

from seaswarm.config import load_default_config
from seaswarm.search.pod import SensorModel

SENSORS = load_default_config().sensors


def _model(name: str, daylight: bool = True, altitude_m: float | None = 60.0) -> SensorModel:
    return SensorModel.from_config(SENSORS[name], daylight, altitude_m)


@pytest.mark.parametrize(
    ("name", "r_max_m", "sigma_r_m", "pod_day", "pod_night"),
    [
        ("drone_eo", 60 * np.tan(np.radians(35)), 25.0, 0.6, 0.1),
        ("drone_thermal", 60 * np.tan(np.radians(25)), 18.0, 0.7, 0.8),
        ("usv_mast", 150.0, 70.0, 0.5, 0.2),
    ],
)
def test_pod_matches_the_sensor_table(
    name: str, r_max_m: float, sigma_r_m: float, pod_day: float, pod_night: float
) -> None:
    altitude = None if name == "usv_mast" else 60.0
    day, night = _model(name, True, altitude), _model(name, False, altitude)
    assert day.r_max_m == pytest.approx(r_max_m)
    assert day.pod(0.0) == pytest.approx(pod_day)
    assert night.pod(0.0) == pytest.approx(pod_night)
    assert day.pod(sigma_r_m) == pytest.approx(pod_day * np.exp(-0.5))
    assert day.pod(r_max_m) == pytest.approx(pod_day * np.exp(-(r_max_m**2) / (2 * sigma_r_m**2)))
    assert day.pod(r_max_m + 1e-6) == 0.0


def test_drone_footprint_grows_and_pod_falls_with_altitude() -> None:
    low, high = _model("drone_eo", altitude_m=60.0), _model("drone_eo", altitude_m=160.0)
    assert high.r_max_m == pytest.approx(160 * np.tan(np.radians(35)))
    assert high.pod_max == pytest.approx(0.6 * np.exp(-1.0))
    assert low.pod_max == pytest.approx(0.6)


def test_pod_max_is_clipped_to_one() -> None:
    assert _model("drone_thermal", daylight=False, altitude_m=1.0).pod_max == 1.0


def test_camera_needs_an_altitude() -> None:
    with pytest.raises(ValueError, match="altitude"):
        _model("drone_eo", altitude_m=None)


def test_coverage_area_is_the_integral_of_pod() -> None:
    usv = _model("usv_mast", altitude_m=None)
    r = np.linspace(0.0, 200.0, 200001)
    numeric = np.trapezoid(usv.pod(r) * 2 * np.pi * r, r)
    assert usv.coverage_area_m2() == pytest.approx(numeric, rel=1e-4)
