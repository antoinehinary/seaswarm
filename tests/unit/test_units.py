import numpy as np
import pytest

from seaswarm import units


def test_one_knot_is_0_514444_mps() -> None:
    assert units.knots_to_mps(1.0) == pytest.approx(0.514444, abs=1e-6)


def test_knots_round_trip() -> None:
    speeds_kn = np.array([0.0, 1.0, 8.0, 35.5])
    np.testing.assert_allclose(units.mps_to_knots(units.knots_to_mps(speeds_kn)), speeds_kn)


@pytest.mark.parametrize(
    ("bearing_deg", "yaw_rad"),
    [(0.0, np.pi / 2), (90.0, 0.0), (180.0, -np.pi / 2), (270.0, np.pi)],
)
def test_bearing_to_yaw_at_cardinal_points(bearing_deg: float, yaw_rad: float) -> None:
    # compare on the circle: 270 deg may come back as -pi, which is the same heading as +pi
    error = units.wrap_angle_rad(units.bearing_deg_to_yaw_rad(bearing_deg) - yaw_rad)
    assert error == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    ("yaw_rad", "bearing_deg"),
    [(np.pi / 2, 0.0), (0.0, 90.0), (-np.pi / 2, 180.0), (np.pi, 270.0)],
)
def test_yaw_to_bearing_at_cardinal_points(yaw_rad: float, bearing_deg: float) -> None:
    assert units.yaw_rad_to_bearing_deg(yaw_rad) == pytest.approx(bearing_deg, abs=1e-12)


def test_bearing_yaw_round_trip() -> None:
    bearings_deg = np.linspace(0.0, 359.0, 360)
    back = units.yaw_rad_to_bearing_deg(units.bearing_deg_to_yaw_rad(bearings_deg))
    np.testing.assert_allclose(back, bearings_deg, atol=1e-9)


def test_wrap_angle_range() -> None:
    wrapped = units.wrap_angle_rad(np.linspace(-20.0, 20.0, 1001))
    assert np.all(wrapped >= -np.pi)
    assert np.all(wrapped < np.pi)


def test_towards_north_and_east() -> None:
    np.testing.assert_allclose(units.towards_vector(2.0, 0.0), (0.0, 2.0), atol=1e-12)
    np.testing.assert_allclose(units.towards_vector(2.0, 90.0), (2.0, 0.0), atol=1e-12)


def test_wind_from_270_blows_east() -> None:
    u_mps, v_mps = units.from_vector(10.0, 270.0)
    assert u_mps == pytest.approx(10.0)
    assert v_mps == pytest.approx(0.0, abs=1e-12)


def test_wind_from_north_blows_south() -> None:
    np.testing.assert_allclose(units.from_vector(5.0, 0.0), (0.0, -5.0), atol=1e-12)


def test_vectors_accept_arrays() -> None:
    u_mps, v_mps = units.towards_vector(np.array([1.0, 2.0]), np.array([0.0, 90.0]))
    np.testing.assert_allclose(u_mps, [0.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(v_mps, [1.0, 0.0], atol=1e-12)
