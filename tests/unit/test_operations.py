import json
from pathlib import Path

import pytest

from seaswarm.config import load_default_config, load_scenario
from seaswarm.mission import MissionState, SearchMission
from seaswarm.operations import Operations
from seaswarm.vehicles.kinematics import Pose

SCENARIO = load_scenario(Path(__file__).resolve().parents[2] / "scenarios" / "windy_mixed.yaml")
DEFAULTS = load_default_config()


@pytest.fixture
def ops() -> Operations:
    return Operations(SearchMission(SCENARIO, DEFAULTS, "greedy"))


def _found(ops: Operations) -> Operations:
    """Test fixture: hover drone_1 over the true person until it reports a detection."""
    drone = ops.m.backend._vehicles["drone_1"]
    for _ in range(200):
        drone.pose = Pose(*ops.m.backend.truth_position_m(), 0.0, 0.0)
        ops.m.step()
        if ops.m.detections:
            return ops
    raise AssertionError("no detection")


def test_situation_is_json_and_has_the_essentials(ops: Operations) -> None:
    sit = ops.situation()
    json.dumps(sit)  # serialisable
    assert sit["mission_state"] == "SEARCHING" and sit["sim_time_s"] == 0.0
    assert [v["id"] for v in sit["vehicles"]] == ["drone_1", "drone_2", "usv_1"]
    assert {"bearing_from_base_deg", "distance_from_base_m"} <= sit["best_estimate"].keys()


def test_probability_map_summary(ops: Operations) -> None:
    summary = ops.probability_map(top_k=5)
    assert len(summary["top_cells"]) == 5
    assert summary["top_cells"][0]["p"] >= summary["top_cells"][-1]["p"]
    assert summary["area_90pct_km2"] > 0.0
    assert ops.probability_map(top_k=0)["refusal"]["code"] == "INVALID_ARGUMENT"


def test_priority_sector_is_one_ninth_and_changes_no_probability(ops: Operations) -> None:
    before = ops.m.belief.p.copy()
    result = ops.set_priority_sector("N", 3.0, 1800.0)
    assert result["ok"] and result["rect_m"] == pytest.approx(
        [-5000 / 3, 5000 / 3, 5000 / 3, 5000.0]
    )
    assert (ops.m.belief.p == before).all()  # a preference, not evidence
    ops.advance(1.0)
    assert any(e.kind == "replan" and "priority" in e.data["triggers"] for e in ops.m.events)
    assert ops.situation()["priority_areas"][0]["weight"] == 3.0
    assert ops.clear_priority_areas() == {"ok": True, "cleared": 1}


@pytest.mark.parametrize(
    "call",
    [
        lambda o: o.set_priority_area(0, 0, -1, 1, 1.0, 60.0),
        lambda o: o.set_priority_area(0, 0, 1, 1, 9.0, 60.0),
        lambda o: o.set_priority_area(0, 0, 1, 1, 1.0, 9999.0),
        lambda o: o.set_priority_sector("X", 1.0, 60.0),
        lambda o: o.set_strategy("random"),
        lambda o: o.advance(5000.0),
        lambda o: o.assign_vehicle("drone_1", 0.0, 0.0, radius_m=10.0),
        lambda o: o.dispatch_to_detection("d_9", "usv_1"),
    ],
)
def test_invalid_arguments_are_explained_not_raised(ops: Operations, call) -> None:
    result = call(ops)
    assert result["ok"] is False and result["refusal"]["code"] == "INVALID_ARGUMENT"
    assert result["refusal"]["reason"]


def test_unknown_vehicle_is_refused_with_the_known_ids(ops: Operations) -> None:
    result = ops.recall_vehicle("drone_9")
    assert result["refusal"]["code"] == "VEHICLE_UNAVAILABLE"
    assert "drone_1, drone_2, usv_1" in result["refusal"]["reason"]


def test_assign_vehicle_takes_it_away_from_the_planner(ops: Operations) -> None:
    result = ops.assign_vehicle("usv_1", -1000.0, -2000.0, radius_m=300.0)
    assert result["ok"], result
    ops.advance(300.0)
    usv = next(v for v in ops.situation()["vehicles"] if v["id"] == "usv_1")
    assert usv["current_task"].startswith("operator: search")
    replans = [e for e in ops.m.events if e.kind == "replan"]
    assert all("usv_1" not in e.data["assigned"] for e in replans[1:])


def test_assignment_outside_the_geofence_is_refused(ops: Operations) -> None:
    result = ops.assign_vehicle("drone_1", 0.0, 9000.0, radius_m=300.0)
    assert result["ok"] is False and result["refusal"]["code"] == "GEOFENCE"
    assert "outside the geofence" in result["refusal"]["reason"]


def test_set_strategy_switches_the_planner(ops: Operations) -> None:
    assert ops.set_strategy("auction") == {"ok": True, "strategy": "auction"}
    assert ops.m.planner.name == "auction"


def test_recall_is_always_accepted(ops: Operations) -> None:
    ops.advance(120.0)
    assert ops.recall_vehicle("drone_2")["ok"]
    ops.advance(2.0)
    drone = next(v for v in ops.situation()["vehicles"] if v["id"] == "drone_2")
    assert drone["status"] == "rtb"


def test_dispatch_a_usv_to_a_detection_recovers_the_person(ops: Operations) -> None:
    _found(ops)
    detection = ops.situation()["detections"][0]
    result = ops.dispatch_to_detection(detection["id"], "usv_1")
    assert result["ok"], result
    assert ops.m.rescuer_id == "usv_1"
    for _ in range(10):
        summary = ops.advance(600.0)
        if summary["ended"]:
            break
    assert ops.m.state is MissionState.RECOVERED


def test_advance_reports_what_happened(ops: Operations) -> None:
    summary = ops.advance(130.0)
    assert summary["ok"] and summary["advanced_s"] == 130.0
    assert summary["events"]["replan"] >= 2
    json.dumps(summary)
