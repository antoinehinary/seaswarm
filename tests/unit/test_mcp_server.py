"""Every MCP tool, called in-process through FastMCP's own dispatch (JSON in, JSON out)."""

import asyncio
import json
from pathlib import Path

import pytest

from seaswarm.mcp_server.server import build_server, make_operations

SCENARIO = Path(__file__).resolve().parents[2] / "scenarios" / "windy_mixed.yaml"
TOOLS = {
    "get_situation",
    "get_probability_map",
    "set_priority_area",
    "set_priority_sector",
    "clear_priority_areas",
    "assign_vehicle",
    "dispatch_to_detection",
    "recall_vehicle",
    "set_strategy",
    "advance_simulation",
}


@pytest.fixture
def server():
    return build_server(make_operations(SCENARIO, None, "greedy"))


def call(server, tool: str, **arguments) -> dict:
    content, _ = asyncio.run(server.call_tool(tool, arguments))
    return json.loads(content[0].text)  # what an agent with structured_output=False receives


def test_the_ten_tools_are_listed_with_descriptions(server) -> None:
    tools = asyncio.run(server.list_tools())
    assert {t.name for t in tools} == TOOLS
    assert all(t.description and len(t.description) > 40 for t in tools)
    schema = next(t for t in tools if t.name == "set_priority_sector").inputSchema
    assert "N" in json.dumps(schema)


def test_reading_tools(server) -> None:
    assert call(server, "get_situation")["mission_state"] == "SEARCHING"
    assert len(call(server, "get_probability_map", top_k=3)["top_cells"]) == 3


def test_priority_tools(server) -> None:
    assert call(server, "set_priority_sector", sector="NE", weight=2.0, duration_s=600)["ok"]
    assert (
        call(
            server,
            "set_priority_area",
            x_min_m=0,
            y_min_m=0,
            x_max_m=1000,
            y_max_m=1000,
            weight=0.5,
            duration_s=600,
        )["area_id"]
        == "area_2"
    )
    assert call(server, "clear_priority_areas")["cleared"] == 2


def test_vehicle_tools_and_refusals(server) -> None:
    assert call(server, "assign_vehicle", vehicle_id="drone_1", x_m=-1000, y_m=-1000)["ok"]
    refused = call(server, "assign_vehicle", vehicle_id="drone_2", x_m=0, y_m=9000)
    assert refused["ok"] is False and refused["refusal"]["code"] == "GEOFENCE"
    assert call(server, "recall_vehicle", vehicle_id="usv_1")["ok"]
    missing = call(server, "dispatch_to_detection", detection_id="d_1", vehicle_id="usv_1")
    assert missing["refusal"]["code"] == "INVALID_ARGUMENT"


def test_strategy_and_time(server) -> None:
    assert call(server, "set_strategy", name="auction") == {"ok": True, "strategy": "auction"}
    summary = call(server, "advance_simulation", duration_s=60)
    assert summary["ok"] and summary["sim_time_s"] == 60.0
    too_long = call(server, "advance_simulation", duration_s=4000)
    assert too_long["refusal"]["code"] == "INVALID_ARGUMENT"


def test_an_internal_error_is_returned_not_raised() -> None:
    ops = make_operations(SCENARIO, None, "greedy")

    def broken():
        raise RuntimeError("boom")

    ops.situation = broken
    result = call(build_server(ops), "get_situation")
    assert result == {
        "ok": False,
        "refusal": {"code": "ERROR", "reason": "boom", "vehicle_id": None},
    }
