"""Real MCP round trip: the server runs as a subprocess, the official client talks over stdio."""

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[2]


async def _round_trip() -> tuple[list[str], dict, dict]:
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "seaswarm.mcp_server.server",
            "--scenario",
            str(ROOT / "scenarios" / "windy_mixed.yaml"),
        ],
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        situation = await session.call_tool("get_situation", {})
        advanced = await session.call_tool("advance_simulation", {"duration_s": 30})
        return (
            sorted(t.name for t in tools.tools),
            json.loads(situation.content[0].text),
            json.loads(advanced.content[0].text),
        )


def test_stdio_round_trip_lists_tools_and_calls_them() -> None:
    names, situation, advanced = asyncio.run(asyncio.wait_for(_round_trip(), timeout=60))
    assert len(names) == 10 and "get_situation" in names
    assert situation["ok"] and len(situation["vehicles"]) == 3
    assert advanced["sim_time_s"] == 30.0
