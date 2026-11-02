import pytest

from seaswarm.config import load_default_config
from seaswarm.vehicles.energy import drain_rate_per_s, reserve_ok, rtb_energy_frac

DRONE = load_default_config().vehicles.drone
USV = load_default_config().vehicles.usv


@pytest.mark.parametrize("cfg", [DRONE, USV], ids=["drone", "usv"])
def test_full_battery_lasts_the_endurance_at_cruise(cfg) -> None:
    assert drain_rate_per_s(cfg.cruise_speed_mps, cfg) * cfg.endurance_s == pytest.approx(1.0)


def test_hovering_drone_still_drains() -> None:
    assert drain_rate_per_s(0.0, DRONE) == pytest.approx(0.7 / 1500.0)
    assert drain_rate_per_s(20.0, DRONE) > drain_rate_per_s(15.0, DRONE)


def test_reserve_rule() -> None:
    # 3 km from base at 15 m/s: 200 s * 1.2 + 120 s = 360 s = 24 % of 1500 s
    assert rtb_energy_frac(3000.0, DRONE) == pytest.approx(0.24)
    assert reserve_ok(0.25, 3000.0, DRONE)
    assert not reserve_ok(0.23, 3000.0, DRONE)
