"""Every replan trigger leads to a replan within one simulation step."""

from pathlib import Path

import pytest

from seaswarm.config import load_default_config, load_scenario
from seaswarm.mission import MissionState, SearchMission
from seaswarm.planning.base import Assignment, PriorityArea, ReplanTrigger, SearchTask
from seaswarm.vehicles.kinematics import Pose

DEFAULTS = load_default_config()
SCENARIO = load_scenario(Path(__file__).resolve().parents[2] / "scenarios" / "windy_mixed.yaml")


def _replans(mission: SearchMission, trigger: str) -> list[float]:
    return [e.t_s for e in mission.events if e.kind == "replan" and trigger in e.data["triggers"]]


def _started(strategy: str = "greedy") -> SearchMission:
    mission = SearchMission(SCENARIO, DEFAULTS, strategy)
    for _ in range(30):
        mission.step()
    return mission


@pytest.mark.parametrize("strategy", ["greedy", "auction"])
def test_operator_priority_replans_next_step(strategy: str) -> None:
    mission = _started(strategy)
    t0 = mission.t_s
    mission.set_priority_area(PriorityArea("a1", -2000, 0, 0, 2000, 3.0, t0 + 3600))
    mission.step()
    assert _replans(mission, "priority") == [t0 + 1.0]


def test_low_energy_replans_in_the_same_step() -> None:
    mission = _started()
    drone = mission.backend._vehicles["drone_1"]  # test fixture: drain the battery
    drone.energy_frac = 0.05
    t_before = mission.t_s
    mission.step()
    assert any(e.kind == "forced_rtb" and e.t_s == t_before + 1.0 for e in mission.events)
    assert _replans(mission, "low_energy") == [t_before + 1.0]


def test_detection_replans_in_the_same_step() -> None:
    mission = _started()
    drone = mission.backend._vehicles["drone_1"]  # test fixture: hover over the true person
    for _ in range(100):
        drone.pose = Pose(*mission.backend.truth_position_m(), 0.0, 0.0)
        mission.step()
        if mission.state is MissionState.FOUND:
            break
    found_at = next(e.t_s for e in mission.events if e.kind == "detection")
    assert _replans(mission, "detection") == [found_at]


class _OutOfBoundsOnce:
    """Stub planner: first proposes an illegal tile, then records what it is told."""

    name = "stub"

    def __init__(self) -> None:
        self.calls = 0
        self.excluded: list[dict] = []

    def plan(self, view, pmap, vehicles, prefs):
        self.calls += 1
        self.excluded.append(view.excluded)
        if self.calls == 1:
            task = SearchTask("t_0_0", ((0.0, 99_000.0),), 0.1, 0.0)
            return {"drone_1": Assignment("drone_1", (task,))}
        return {}


def test_refusal_replans_next_step_without_repeating_the_refused_choice() -> None:
    mission = SearchMission(SCENARIO, DEFAULTS, "greedy")
    stub = _OutOfBoundsOnce()
    mission.planner = stub
    mission.step()  # start replan: the stub's proposal is refused (geofence)
    assert mission.refusals[0].code.value == "GEOFENCE"
    mission.step()
    assert _replans(mission, "refusal") == [mission.t_s]
    assert stub.excluded[-1] == {"drone_1": frozenset({"t_0_0"})}


def test_vehicle_idle_replans_are_rate_limited() -> None:
    mission = _started()
    mission.set_priority_area(PriorityArea("a1", -2000, 0, 0, 2000, 3.0, 1e9))
    mission.step()  # immediate priority replan
    replanned_at = mission.t_s
    mission._pending.add(ReplanTrigger.VEHICLE_IDLE)
    for _ in range(9):
        mission.step()
    assert _replans(mission, "vehicle_idle") == []  # inside the 10 s cooldown
    mission.step()
    assert _replans(mission, "vehicle_idle") == [replanned_at + 10.0]
