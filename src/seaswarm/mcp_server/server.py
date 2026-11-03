"""MCP server: one search mission, operated through typed tools over stdio.

Run:
    python -m seaswarm.mcp_server.server --scenario scenarios/windy_mixed.yaml --seed 3

Tools take SI units, return JSON objects and never raise: a refused or invalid request comes
back as {"ok": false, "refusal": {"code", "reason", "vehicle_id"}}. Every vehicle command goes
through the mission's safety layer. Simulated time only advances through advance_simulation.

Adapted from skypilot-agent's server (build_server factory, logging on stderr).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from seaswarm.config import load_default_config, load_scenario
from seaswarm.mission import SearchMission
from seaswarm.operations import Operations

logger = logging.getLogger(__name__)

SERVER_NAME = "seaswarm"
INSTRUCTIONS = """\
Coordinates a search-and-rescue swarm (drones and small boats, "USVs") looking for a person in
the water. Frame: x east, y north, metres, origin at the centre of the search area; the base is
a point in that frame. Every tool returns a JSON object with "ok". Speeds are in m/s, times in
seconds, bearings in degrees clockwise from north. Vehicles search automatically from a
probability map; you can prioritise areas, give vehicles direct orders, and advance time.
Refusals come from the safety layer: read the reason and choose differently."""

Metres = Annotated[float, Field(description="metres in the local frame (x east, y north)")]
Weight = Annotated[
    float,
    Field(description="utility multiplier from 0 (avoid) to 5 (strongly prefer); 1 is neutral"),
]
Duration = Annotated[
    float, Field(description="how long the priority lasts, seconds (at most 3600)")
]
Result = dict[str, Any]


def build_server(ops: Operations) -> FastMCP:
    """Create the MCP server with one tool per operator action, bound to `ops`."""
    mcp = FastMCP(SERVER_NAME, instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(readOnlyHint=True)

    def safe(name: str, call) -> Result:
        try:
            return call()
        except Exception as error:  # never raise to the client: report the error as data
            logger.exception("tool %s failed", name)
            return {
                "ok": False,
                "refusal": {"code": "ERROR", "reason": str(error), "vehicle_id": None},
            }

    @mcp.tool(annotations=read_only)
    def get_situation() -> Result:
        """Current mission state. Call this first.

        Returns sim_time_s (since the search started), mission_state (SEARCHING, FOUND,
        RECOVERED or TIMEOUT), strategy, best_estimate of the person's position (x_m, y_m and
        bearing/distance from base), pos_cum (probability the search has already succeeded),
        vehicles (id, type, position, heading_deg, speed_mps, energy_pct, status, current_task),
        detections (id, vehicle_id, time_s, position), active priority_areas, last_refusals.
        """
        return safe("get_situation", ops.situation)

    @mcp.tool(annotations=read_only)
    def get_probability_map(
        top_k: Annotated[
            int, Field(description="number of most probable 100 m cells, 1 to 50")
        ] = 10,
    ) -> Result:
        """Summary of where the person probably is.

        Returns top_cells (x_m, y_m, probability, bearing/distance from base), p_out
        (probability the person left the map), entropy_bits, centroid_x_m, centroid_y_m,
        area_90pct_km2 (smallest area holding 90 % of the probability).
        """
        return safe("get_probability_map", lambda: ops.probability_map(top_k))

    @mcp.tool()
    def set_priority_area(
        x_min_m: Metres,
        y_min_m: Metres,
        x_max_m: Metres,
        y_max_m: Metres,
        weight: Weight,
        duration_s: Duration,
    ) -> Result:
        """Make the planner prefer (weight > 1) or avoid (weight < 1) a rectangle for a while.

        Changes how vehicles choose where to search, never the probability map. Returns area_id.
        """
        return safe(
            "set_priority_area",
            lambda: ops.set_priority_area(x_min_m, y_min_m, x_max_m, y_max_m, weight, duration_s),
        )

    @mcp.tool()
    def set_priority_sector(
        sector: Annotated[
            Literal["NW", "N", "NE", "W", "C", "E", "SW", "S", "SE"],
            Field(
                description="one ninth of the search area (3 x 3 split); N = top middle, C = centre"
            ),
        ],
        weight: Weight,
        duration_s: Duration,
    ) -> Result:
        """Same as set_priority_area for one sector of the search area. Returns area_id."""
        return safe(
            "set_priority_sector", lambda: ops.set_priority_sector(sector, weight, duration_s)
        )

    @mcp.tool()
    def clear_priority_areas() -> Result:
        """Remove every priority area. Returns how many were cleared."""
        return safe("clear_priority_areas", ops.clear_priority_areas)

    @mcp.tool()
    def assign_vehicle(
        vehicle_id: Annotated[
            str, Field(description="a vehicle id from get_situation, e.g. drone_1")
        ],
        x_m: Metres,
        y_m: Metres,
        radius_m: Annotated[
            float, Field(description="half the side of the square to search, metres, 50 to 2000")
        ] = 300.0,
    ) -> Result:
        """Order one vehicle to search the square around (x_m, y_m).

        The vehicle stays on this order (the automatic planner leaves it alone) until it is done,
        then rejoins the automatic search. May be refused by the safety layer.
        """
        return safe("assign_vehicle", lambda: ops.assign_vehicle(vehicle_id, x_m, y_m, radius_m))

    @mcp.tool()
    def dispatch_to_detection(
        detection_id: Annotated[
            str, Field(description="a detection id from get_situation, e.g. d_1")
        ],
        vehicle_id: Annotated[
            str, Field(description="a vehicle id from get_situation, e.g. usv_1")
        ],
    ) -> Result:
        """Send a vehicle at full speed to a detection. A USV that reaches the person recovers them.

        May be refused by the safety layer.
        """
        return safe(
            "dispatch_to_detection", lambda: ops.dispatch_to_detection(detection_id, vehicle_id)
        )

    @mcp.tool()
    def recall_vehicle(
        vehicle_id: Annotated[str, Field(description="a vehicle id from get_situation")],
    ) -> Result:
        """Order a vehicle back to base (never refused for a known vehicle)."""
        return safe("recall_vehicle", lambda: ops.recall_vehicle(vehicle_id))

    @mcp.tool()
    def set_strategy(
        name: Annotated[
            Literal["lawnmower", "greedy", "auction", "cbba"],
            Field(
                description="lawnmower: fixed strips; greedy, auction or cbba (decentralised "
                "auction): follow the probability map"
            ),
        ],
    ) -> Result:
        """Switch how the automatic planner allocates the search."""
        return safe("set_strategy", lambda: ops.set_strategy(name))

    @mcp.tool()
    def advance_simulation(
        duration_s: Annotated[float, Field(description="simulated seconds to run, 1 to 600")],
    ) -> Result:
        """Let the mission run for duration_s simulated seconds.

        Returns what happened: new_detections, events (counts), new_refusals, mission_state,
        sim_time_s, pos_cum, and ended (true when the mission is over).
        """
        return safe("advance_simulation", lambda: ops.advance(duration_s))

    return mcp


def make_operations(
    scenario: Path, seed: int | None, strategy: str, warmup_s: float = 0.0
) -> Operations:
    """Mission of a scenario (optionally another seed), advanced by `warmup_s` seconds."""
    loaded = load_scenario(scenario)
    if seed is not None:
        loaded = loaded.model_copy(update={"seed": seed})
    ops = Operations(SearchMission(loaded, load_default_config(), strategy))
    remaining = warmup_s
    while remaining > 0 and not ops.m.done:
        step = min(remaining, 600.0)
        ops.advance(step)
        remaining -= step
    return ops


def main(argv: list[str] | None = None) -> None:
    """Start the MCP server over stdio."""
    parser = argparse.ArgumentParser(description="SeaSwarm MCP server (stdio).")
    parser.add_argument("--scenario", type=Path, default=Path("scenarios/windy_mixed.yaml"))
    parser.add_argument("--seed", type=int, help="overrides the scenario seed")
    parser.add_argument("--strategy", default="auction", help="lawnmower, greedy, auction, cbba")
    parser.add_argument(
        "--warmup-s", type=float, default=0.0, help="simulate this long before serving"
    )
    parser.add_argument(
        "--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args(argv)
    # NOTE: stdout carries the MCP protocol; logging goes to stderr.
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(name)s %(message)s")
    ops = make_operations(args.scenario, args.seed, args.strategy, args.warmup_s)
    build_server(ops).run(transport="stdio")


if __name__ == "__main__":
    main()
