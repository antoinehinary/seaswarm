from pathlib import Path

import numpy as np
import pytest

from seaswarm.config import FleetConfig, load_default_config, load_scenario
from seaswarm.rng import make_rngs
from seaswarm.vehicles.backend import FollowPath, GoTo, Hold, ReturnToBase, VehicleStatus
from seaswarm.vehicles.sim2d import Sim2DBackend, fleet_specs

SCENARIO = load_scenario(Path(__file__).resolve().parents[2] / "scenarios" / "windy_mixed.yaml")
DEFAULTS = load_default_config()
BASE = (SCENARIO.base.x_m, SCENARIO.base.y_m)


def _sim(seed: int = 0) -> Sim2DBackend:
    return Sim2DBackend.from_scenario(SCENARIO, DEFAULTS, make_rngs(seed))


def _run(sim: Sim2DBackend, seconds: float) -> list:
    reports = []
    for _ in range(round(seconds)):
        reports += sim.step(1.0)
    return reports


def test_fleet_ids_and_sensors() -> None:
    specs = fleet_specs(FleetConfig(drones_eo=1, drones_thermal=1, usvs=2), DEFAULTS)
    assert [s.vehicle_id for s in specs] == ["drone_1", "drone_2", "usv_1", "usv_2"]
    assert [s.sensor for s in specs] == ["drone_eo", "drone_thermal", "usv_mast", "usv_mast"]


def test_fleet_starts_at_base_when_the_search_starts() -> None:
    sim = _sim()
    assert sim.vehicle_ids() == ["drone_1", "drone_2", "usv_1"]
    assert sim.t_s == SCENARIO.lkp.time_since_fall_s
    state = sim.get_state("usv_1")
    assert (state.x_m, state.y_m, state.energy_frac, state.status) == (*BASE, 1.0, "idle")


def test_goto_reaches_the_point_and_holds() -> None:
    sim = _sim()
    sim.send(GoTo("drone_1", BASE[0], BASE[1] + 1500.0, 15.0))
    assert sim.get_state("drone_1").status is VehicleStatus.TRANSIT
    _run(sim, 150)
    state = sim.get_state("drone_1")
    assert np.hypot(state.x_m - BASE[0], state.y_m - BASE[1] - 1500.0) < 30.0
    assert state.status is VehicleStatus.IDLE
    assert 0.85 < state.energy_frac < 0.95  # ~100 s at cruise + hovering


def test_follow_path_visits_every_waypoint() -> None:
    sim = _sim()
    path = ((BASE[0], BASE[1] + 500.0), (BASE[0] + 500.0, BASE[1] + 500.0))
    sim.send(FollowPath("drone_1", path, 15.0))
    _run(sim, 40)
    assert sim.get_state("drone_1").waypoints_left == 1
    _run(sim, 60)
    state = sim.get_state("drone_1")
    assert state.waypoints_left == 0 and state.x_m > BASE[0] + 450.0


def test_return_to_base_then_battery_swap() -> None:
    sim = _sim()
    sim.send(GoTo("drone_1", BASE[0], BASE[1] + 1000.0, 15.0))
    _run(sim, 100)
    sim.send(ReturnToBase("drone_1"))
    _run(sim, 100)
    assert sim.get_state("drone_1").status is VehicleStatus.TURNAROUND
    _run(sim, DEFAULTS.vehicles.drone.turnaround_s)
    state = sim.get_state("drone_1")
    assert state.status is VehicleStatus.IDLE and state.energy_frac == 1.0


def test_parked_vehicles_use_no_energy_and_take_no_looks() -> None:
    sim = _sim()
    assert _run(sim, 600) == []
    assert all(sim.get_state(vid).energy_frac == 1.0 for vid in sim.vehicle_ids())


def test_holding_usv_drifts_with_the_current_and_drone_does_not() -> None:
    sim = _sim()
    sim.send(GoTo("usv_1", BASE[0], BASE[1] + 200.0, 4.0))
    sim.send(GoTo("drone_1", BASE[0], BASE[1] + 200.0, 4.0))
    _run(sim, 120)
    sim.send(Hold("usv_1"))
    usv0, drone0 = sim.get_state("usv_1"), sim.get_state("drone_1")
    _run(sim, 600)
    usv, drone = sim.get_state("usv_1"), sim.get_state("drone_1")
    assert np.hypot(usv.x_m - usv0.x_m, usv.y_m - usv0.y_m) > 100.0
    assert (drone.x_m, drone.y_m) == pytest.approx((drone0.x_m, drone0.y_m), abs=1.0)


def test_look_periods() -> None:
    sim = _sim()
    for vid in ("drone_1", "usv_1"):
        sim.send(GoTo(vid, BASE[0], BASE[1] + 5000.0, 4.0))
    reports = _run(sim, 60)
    counts = {vid: sum(r.vehicle_id == vid for r in reports) for vid in ("drone_1", "usv_1")}
    assert counts == {"drone_1": 30, "usv_1": 12}


def test_detections_follow_pod_and_carry_noisy_positions() -> None:
    sim = _sim(seed=4)
    hits, errors = 0, []
    for _ in range(300):  # hover right above the person: POD = pod_max = 0.6 per look
        sim._vehicles["drone_1"].pose = sim._vehicles["drone_1"].pose.__class__(
            *sim.truth_position_m(), 0.0, 0.0
        )
        for r in sim.step(1.0):
            if r.vehicle_id == "drone_1" and r.detected:
                hits += 1
                errors.append(np.hypot(*(np.array(r.measured_xy_m) - sim.truth_position_m())))
    assert hits / 150 == pytest.approx(0.6, abs=0.12)
    assert np.mean(errors) == pytest.approx(10.0 * np.sqrt(np.pi / 2), rel=0.3)  # Rayleigh mean


def test_drone_out_of_energy_is_stranded() -> None:
    sim = _sim()
    sim.send(GoTo("drone_1", BASE[0], BASE[1] + 20000.0, 15.0))
    _run(sim, 1600)
    state = sim.get_state("drone_1")
    assert state.status is VehicleStatus.STRANDED and state.energy_frac == 0.0


def test_usv_reaching_the_person_recovers_them() -> None:
    sim = _sim()
    assert not sim.person_recovered()
    for _ in range(4000):
        x, y = sim.truth_position_m()
        sim.send(GoTo("usv_1", float(x), float(y), 6.0))
        sim.step(1.0)
        if sim.person_recovered():
            break
    assert sim.person_recovered()
