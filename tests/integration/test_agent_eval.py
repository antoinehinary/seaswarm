"""The evaluation harness itself, with scripted agents (no LLM)."""

from seaswarm.agent.models import ScriptedModel
from seaswarm.eval.agent_eval import load_cases, run_case

CASES = {c.id: c for c in load_cases()}


def test_there_are_ten_cases_with_known_checks() -> None:
    assert len(CASES) == 10
    assert CASES["boat_to_drone_detection"].seed == 14


def test_a_correct_agent_passes() -> None:
    model = ScriptedModel.from_code(
        [
            'print(set_priority_sector(sector="N", weight=3.0, duration_s=1800.0))',
            'final_answer("north sector prioritised")',
        ]
    )
    outcome = run_case(CASES["north_sector"], model, max_steps=4)
    assert outcome.passed, outcome.failures


def test_a_wrong_agent_fails_with_a_reason() -> None:
    model = ScriptedModel.from_code(
        [
            'print(set_priority_sector(sector="S", weight=3.0, duration_s=1800.0))',
            'final_answer("done")',
        ]
    )
    outcome = run_case(CASES["north_sector"], model, max_steps=4)
    assert not outcome.passed
    assert outcome.failures == ["no priority area with weight > 1 on sector N"]


def test_report_checks_read_the_answer() -> None:
    model = ScriptedModel.from_code(['final_answer("the order was refused: outside the geofence")'])
    assert run_case(CASES["refused_outside_geofence"], model, max_steps=2).passed
