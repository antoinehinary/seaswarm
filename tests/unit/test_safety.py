from pathlib import Path

import numpy as np
import pytest

from seaswarm.config import load_default_config, load_scenario
from seaswarm.rng import make_rngs
from seaswarm.safety.guard import Proposal, SafetyGuard
from seaswarm.safety.refusal import RefusalCode
from seaswarm.safety.rules import Geofence, position_along
from seaswarm.vehicles.backend import FollowPath, GoTo, ReturnToBase, VehicleStatus
from seaswarm.vehicles.kinematics import Pose
from seaswarm.vehicles.sim2d import Sim2DBackend

SCENARIO = load_scenario(Path(__file__).resolve().parents[2] / "scenarios" / "windy_mixed.yaml")
DEFAULTS = load_default_config()
BASE = (SCENARIO.base.x_m, SCENARIO.base.y_m)


def _world(drones: int = 2, usvs: int = 1) -> tuple[Sim2DBackend, SafetyGuard]:
    raw = SCENARIO.model_dump()
    raw["fleet"] = {"drones_eo": drones, "drones_thermal": 0, "usvs": usvs}
    scenario = SCENARIO.model_validate(raw)
    sim = Sim2DBackend.from_scenario(scenario, DEFAULTS, make_rngs(0))
    fence = Geofence.around(scenario.area, scenario.base, DEFAULTS.safety.geofence_margin_m)
    return sim, SafetyGuard(sim, fence, BASE, DEFAULTS)


@pytest.fixture
def world() -> tuple[Sim2DBackend, SafetyGuard]:
    return _world()


def _place(sim: Sim2DBackend, vid: str, x: float, y: float, yaw: float = 0.0, energy: float = 1.0):
    """Test fixture: teleport a vehicle (bypasses nothing: no command is sent)."""
    v = sim._vehicles[vid]
    v.pose, v.energy_frac, v.status = Pose(x, y, yaw, 0.0), energy, VehicleStatus.IDLE


def _steps(sim: Sim2DBackend, guard: SafetyGuard, n: int) -> list[str]:
    forced = []
    for _ in range(n):
        sim.step(1.0)
        forced += guard.check_state(sim.t_s)
    return forced


def test_geofence_refusal(world) -> None:
    sim, guard = world
    refusal = guard.submit(Proposal(GoTo("drone_1", 0.0, 7000.0, 15.0)), sim.t_s)
    assert refusal.code is RefusalCode.GEOFENCE
    assert refusal.reason == (
        "drone_1 waypoint (0, 7000) m is outside the geofence "
        "(x -6000 to 6000 m, y -7000 to 6000 m)"
    )
    assert sim.get_state("drone_1").status is VehicleStatus.IDLE  # never reached the backend


def test_energy_reserve_refusal(world) -> None:
    sim, guard = world
    _place(sim, "drone_1", 0.0, -3000.0, energy=0.30)
    refusal = guard.submit(Proposal(GoTo("drone_1", 0.0, 3000.0, 15.0)), sim.t_s)
    assert refusal.code is RefusalCode.ENERGY_RESERVE
    assert refusal.reason == (
        "drone_1 would finish this task with 3% energy, but needs 56% to return to base "
        "with the reserve"
    )


def test_tile_occupied_refusal_and_dispatch_exemption(world) -> None:
    sim, guard = world
    assert (
        guard.submit(Proposal(GoTo("drone_1", 0.0, 0.0, 15.0), tile_id="t_12_12"), sim.t_s) is None
    )
    refusal = guard.submit(Proposal(GoTo("drone_2", 500.0, 0.0, 15.0), tile_id="t_12_12"), sim.t_s)
    assert refusal.code is RefusalCode.TILE_OCCUPIED
    assert (
        refusal.reason == "tile t_12_12 is already being searched by drone_1; choose another tile"
    )
    dispatch = Proposal(GoTo("usv_1", 0.0, 0.0, 4.0), tile_id="t_12_12", dispatch=True)
    assert guard.submit(dispatch, sim.t_s) is None


def test_vehicle_unavailable_refusals(world) -> None:
    sim, guard = world
    _place(sim, "drone_1", 0.0, -3000.0)
    guard.submit(Proposal(ReturnToBase("drone_1")), sim.t_s)
    refusal = guard.submit(Proposal(GoTo("drone_1", 0.0, 0.0, 15.0)), sim.t_s)
    assert refusal.code is RefusalCode.VEHICLE_UNAVAILABLE
    assert refusal.reason == "drone_1 is returning to base and cannot take a task"
    unknown = guard.submit(Proposal(GoTo("drone_9", 0.0, 0.0, 15.0)), sim.t_s)
    assert unknown.code is RefusalCode.VEHICLE_UNAVAILABLE
    assert (
        unknown.reason == "there is no vehicle 'drone_9'; known vehicles: drone_1, drone_2, usv_1"
    )


def test_separation_refusal_is_predictive() -> None:
    sim, guard = _world(drones=4)
    # drone_1 and drone_4 share the 60 m layer
    _place(sim, "drone_1", -300.0, 0.0)
    _place(sim, "drone_4", 300.0, 0.0, yaw=np.pi)
    assert guard.submit(Proposal(GoTo("drone_1", 300.0, 0.0, 15.0)), sim.t_s) is None
    refusal = guard.submit(Proposal(GoTo("drone_4", -300.0, 0.0, 15.0)), sim.t_s)
    assert refusal.code is RefusalCode.SEPARATION
    assert refusal.reason == (
        "drone_4 would come within 0 m of drone_1 in 20 s; the minimum separation is 30 m"
    )
    # different layers, or a USV below a drone, are not conflicts
    _place(sim, "drone_2", 300.0, 0.0, yaw=np.pi)
    assert guard.submit(Proposal(GoTo("drone_2", -300.0, 0.0, 15.0)), sim.t_s) is None
    _place(sim, "usv_1", 300.0, 0.0)
    assert guard.submit(Proposal(GoTo("usv_1", -300.0, 0.0, 4.0)), sim.t_s) is None


def test_recall_is_always_accepted(world) -> None:
    sim, guard = world
    _place(sim, "drone_1", 0.0, 4000.0, energy=0.01)
    assert guard.submit(Proposal(ReturnToBase("drone_1")), sim.t_s) is None
    assert sim.get_state("drone_1").status is VehicleStatus.RTB


def test_drones_launched_together_are_in_different_layers(world) -> None:
    sim, guard = world
    assert [sim.get_state(v).altitude_m for v in ("drone_1", "drone_2")] == [60.0, 80.0]
    for vid in ("drone_1", "drone_2"):
        assert guard.submit(Proposal(GoTo(vid, 0.0, 0.0, 15.0)), sim.t_s) is None


def test_low_energy_forces_rtb_before_the_reserve_is_broken(world) -> None:
    sim, guard = world
    _place(sim, "drone_1", 0.0, 0.0, energy=0.50)
    guard.submit(Proposal(GoTo("drone_1", 0.0, 4000.0, 15.0)), sim.t_s)
    forced = _steps(sim, guard, 600)
    assert forced == ["drone_1"]
    _steps(sim, guard, 900)
    assert sim.get_state("drone_1").status in (VehicleStatus.TURNAROUND, VehicleStatus.IDLE)
    assert guard.violations.total == 0


def test_actual_separation_violations_are_counted_once_per_episode() -> None:
    sim, guard = _world(drones=4)
    _place(sim, "drone_1", 0.0, 0.0)
    _place(sim, "drone_4", 10.0, 0.0)  # same 60 m layer
    _steps(sim, guard, 5)
    assert guard.violations.separation == 1


def test_follow_path_releases_the_tile_when_done(world) -> None:
    sim, guard = world
    path = ((0.0, -5600.0), (200.0, -5600.0))
    guard.submit(Proposal(FollowPath("drone_1", path, 15.0), tile_id="t_0_0"), sim.t_s)
    assert guard.tile_of("drone_1") == "t_0_0"
    _steps(sim, guard, 120)
    assert guard.tile_of("drone_1") is None


def test_position_along_a_polyline() -> None:
    path = ((10.0, 0.0), (10.0, 10.0))
    assert position_along((0.0, 0.0), path, 15.0) == pytest.approx((10.0, 5.0))
    assert position_along((0.0, 0.0), path, 99.0) == pytest.approx((10.0, 10.0))
