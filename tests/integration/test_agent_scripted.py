"""The whole agent stack without an LLM: smolagents MCPClient + CodeAgent + ScriptedModel."""

from pathlib import Path

from smolagents import LogLevel

from seaswarm.agent.cli import ServerOptions, run_instruction
from seaswarm.agent.models import ScriptedModel

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = ServerOptions(scenario=ROOT / "scenarios" / "windy_mixed.yaml", strategy="greedy")


def test_scripted_agent_changes_the_mission_through_mcp() -> None:
    model = ScriptedModel.from_code(
        [
            "situation = get_situation()\n"
            'print(situation["mission_state"], len(situation["vehicles"]))',
            'result = set_priority_sector(sector="N", weight=3.0, duration_s=1800.0)\n'
            "print(result)",
            'refused = assign_vehicle(vehicle_id="drone_1", x_m=0.0, y_m=9000.0)\n'
            'print(refused["ok"], refused["refusal"]["code"])',
            'print(advance_simulation(duration_s=60.0)["sim_time_s"])',
            'final_answer("north sector prioritised; drone_1 order refused by the geofence")',
        ]
    )
    result = run_instruction(
        "prioritise the north sector", model, OPTIONS, max_steps=6, verbosity=LogLevel.OFF
    )
    assert result.run.state == "success"
    assert "north sector prioritised" in result.run.output
    assert [a["id"] for a in result.situation["priority_areas"]] == ["area_1"]
    assert result.situation["sim_time_s"] == 60.0
    assert result.situation["last_refusals"][0]["code"] == "GEOFENCE"
    # the agent saw parsed dicts, not JSON strings: its printed observations prove it
    observations = " ".join(model.prompts)
    assert "SEARCHING 3" in observations and "False GEOFENCE" in observations
