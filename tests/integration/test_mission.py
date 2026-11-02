from pathlib import Path

import pytest

from seaswarm.config import load_default_config, load_scenario
from seaswarm.mission import MissionState, SearchMission

SCENARIOS = Path(__file__).resolve().parents[2] / "scenarios"
DEFAULTS = load_default_config()


def _mission(name: str = "windy_mixed", strategy: str = "greedy", **overrides) -> SearchMission:
    scenario = load_scenario(SCENARIOS / f"{name}.yaml")
    if overrides:
        raw = scenario.model_dump()
        raw.update(overrides)
        scenario = scenario.model_validate(raw)
    return SearchMission(scenario, DEFAULTS, strategy)


@pytest.mark.parametrize("strategy", ["lawnmower", "greedy"])
def test_mission_ends_safely_within_its_limits(strategy: str) -> None:
    mission = _mission(strategy=strategy)
    state = mission.run()
    assert state in (MissionState.RECOVERED, MissionState.TIMEOUT)
    assert mission.steps <= mission.scenario.limits.max_steps
    assert mission.guard.violations.total == 0
    assert abs(mission.belief.total() - 1.0) < 1e-9


def test_detection_leads_to_recovery_by_a_usv() -> None:
    mission = _mission()
    assert mission.run() is MissionState.RECOVERED
    assert mission.time_to_detection_s <= mission.time_to_recovery_s
    kinds = [e.kind for e in mission.events]
    assert kinds.index("detection") < len(kinds) - 1
    assert mission.detections[0].detection_id == "d_1"


def test_same_seed_same_run() -> None:
    a, b = _mission(), _mission()
    a.run()
    b.run()
    assert [(e.t_s, e.kind, e.data) for e in a.events] == [
        (e.t_s, e.kind, e.data) for e in b.events
    ]
    assert a.success.pos_cum == b.success.pos_cum


def test_time_limit_ends_the_mission() -> None:
    mission = _mission(limits={"max_sim_time_s": 300.0, "max_steps": 20000})
    assert mission.run() is MissionState.TIMEOUT
    assert mission.t_s - mission.start_s == pytest.approx(300.0)


def test_step_limit_ends_the_mission() -> None:
    mission = _mission(limits={"max_sim_time_s": 10800.0, "max_steps": 50})
    assert mission.run() is MissionState.TIMEOUT
    assert mission.steps == 50


def test_periodic_replans_happen_while_searching() -> None:
    mission = _mission(limits={"max_sim_time_s": 600.0, "max_steps": 20000})
    mission.run()
    periodic = [
        e for e in mission.events if e.kind == "replan" and "periodic" in e.data["triggers"]
    ]
    assert len(periodic) >= 4


def test_unknown_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown strategy"):
        _mission(strategy="random")
