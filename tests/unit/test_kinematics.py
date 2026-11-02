import numpy as np
import pytest

from seaswarm.config import load_default_config
from seaswarm.vehicles.kinematics import KinematicLimits, Pose, heading_to, step_unicycle

VEHICLES = load_default_config().vehicles
DRONE = KinematicLimits.from_config(VEHICLES.drone)
USV = KinematicLimits.from_config(VEHICLES.usv)


def _run(pose: Pose, yaw_cmd: float, speed_cmd: float, limits: KinematicLimits, steps: int, **kw):
    poses = [pose]
    for _ in range(steps):
        poses.append(step_unicycle(poses[-1], yaw_cmd, speed_cmd, limits, 1.0, **kw))
    return poses


@pytest.mark.parametrize("limits", [DRONE, USV], ids=["drone", "usv"])
def test_turn_rate_and_acceleration_limits_hold(limits: KinematicLimits) -> None:
    poses = _run(Pose(0.0, 0.0, 0.0, 0.0), np.pi, 100.0, limits, 60)
    yaws = np.unwrap([p.yaw_rad for p in poses])
    speeds = np.array([p.speed_mps for p in poses])
    assert np.max(np.abs(np.diff(yaws))) <= limits.max_turn_rate_rad_s + 1e-12
    assert np.max(np.abs(np.diff(speeds))) <= limits.max_accel_mps2 + 1e-12
    assert speeds.max() == pytest.approx(limits.max_speed_mps)


def test_heading_converges_to_the_command() -> None:
    pose = _run(Pose(0.0, 0.0, 0.0, 15.0), np.radians(120.0), 15.0, DRONE, 20)[-1]
    assert pose.yaw_rad == pytest.approx(np.radians(120.0), abs=1e-3)


def test_straight_line_at_cruise() -> None:
    pose = _run(Pose(0.0, 0.0, np.pi / 2, 4.0), np.pi / 2, 4.0, USV, 100)[-1]
    assert (pose.x_m, pose.y_m) == pytest.approx((0.0, 400.0), abs=1e-9)


def test_current_carries_a_holding_usv() -> None:
    pose = _run(Pose(0.0, 0.0, 0.0, 0.0), 0.0, 0.0, USV, 100, drift_mps=(0.3, -0.1))[-1]
    assert (pose.x_m, pose.y_m) == pytest.approx((30.0, -10.0))


def test_turn_radius_fits_inside_the_waypoint_radius() -> None:
    # otherwise a vehicle could circle a waypoint forever
    for cfg, limits in ((VEHICLES.drone, DRONE), (VEHICLES.usv, USV)):
        assert limits.min_turn_radius_m(cfg.cruise_speed_mps) < cfg.waypoint_radius_m


def test_heading_to() -> None:
    assert heading_to(Pose(0.0, 0.0, 0.0, 0.0), 0.0, 10.0) == pytest.approx(np.pi / 2)
