"""Direct the search in plain language with an LLM agent connected to the MCP server.

Usage:
    seaswarm-agent "prioritise the north sector" --scenario scenarios/windy_mixed.yaml --seed 3
    seaswarm-agent --interactive --scenario scenarios/windy_mixed.yaml

The agent never touches the simulator: it starts the MCP server as a subprocess and calls its
tools over stdio. Adapted from skypilot-agent/src/skypilot/agent/run_agent.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from mcp import StdioServerParameters
from smolagents import CodeAgent, LogLevel, MCPClient, Model
from smolagents.agents import RunResult

from seaswarm.agent.models import build_model
from seaswarm.agent.tools import OperatorTool
from seaswarm.config import load_agent_config

PROMPT = Path(__file__).with_name("prompts") / "system.md"


@dataclass(frozen=True)
class ServerOptions:
    """Mission served by the MCP server."""

    scenario: Path = Path("scenarios/windy_mixed.yaml")
    seed: int | None = None
    strategy: str = "auction"
    warmup_s: float = 0.0


def server_parameters(options: ServerOptions) -> StdioServerParameters:
    """Command that starts the MCP server with the current interpreter."""
    args = ["-m", "seaswarm.mcp_server.server", "--scenario", str(options.scenario)]
    args += ["--strategy", options.strategy, "--warmup-s", str(options.warmup_s)]
    if options.seed is not None:
        args += ["--seed", str(options.seed)]
    return StdioServerParameters(command=sys.executable, args=args)


def make_agent(tools: list, model: Model, max_steps: int, verbosity: LogLevel) -> CodeAgent:
    """CodeAgent with the operator instructions and the wrapped MCP tools."""
    return CodeAgent(
        tools=[OperatorTool(t) for t in tools],
        model=model,
        instructions=PROMPT.read_text(),
        max_steps=max_steps,
        additional_authorized_imports=["math"],
        code_block_tags="markdown",
        verbosity_level=verbosity,
    )


@dataclass(frozen=True)
class OperatorResult:
    """Agent run plus the situation read back from the server, independently of the LLM."""

    run: RunResult
    situation: dict


def run_instruction(
    instruction: str,
    model: Model,
    options: ServerOptions,
    max_steps: int,
    verbosity: LogLevel = LogLevel.INFO,
) -> OperatorResult:
    """Start the server, let the agent act on one instruction, then read the situation."""
    with MCPClient(server_parameters(options), structured_output=False) as tools:
        agent = make_agent(tools, model, max_steps, verbosity)
        run = agent.run(instruction, return_full_result=True)
        get_situation = next(t for t in tools if t.name == "get_situation")
        return OperatorResult(run, json.loads(get_situation()))


def interactive(model: Model, options: ServerOptions, max_steps: int) -> None:
    """One mission, many instructions; the agent keeps its memory between them."""
    with MCPClient(server_parameters(options), structured_output=False) as tools:
        agent = make_agent(tools, model, max_steps, LogLevel.INFO)
        reset = True
        while True:
            try:
                instruction = input("\noperator> ").strip()
            except EOFError:
                return
            if instruction in {"", "quit", "exit"}:
                return
            print(agent.run(instruction, reset=reset))
            reset = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="SeaSwarm operator agent.")
    parser.add_argument("instruction", nargs="?", help="plain-language instruction")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--scenario", type=Path, default=ServerOptions.scenario)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--strategy", default="auction", help="lawnmower, greedy, auction, cbba")
    parser.add_argument("--warmup-s", type=float, default=0.0, help="simulate this long first")
    parser.add_argument("--model", help="LiteLLM model id (default: config/agent.yaml)")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--quiet", action="store_true", help="only print the report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one instruction (or an interactive session) and print the verified situation."""
    load_dotenv()
    args = parse_args(argv)
    config = load_agent_config()
    if args.model:
        config = config.model_copy(update={"model_id": args.model})
    model = build_model(config)
    options = ServerOptions(args.scenario, args.seed, args.strategy, args.warmup_s)
    max_steps = args.max_steps or config.max_steps
    if args.interactive:
        interactive(model, options, max_steps)
        return 0
    if not args.instruction:
        raise SystemExit("give an instruction, or use --interactive")
    start = time.monotonic()
    verbosity = LogLevel.OFF if args.quiet else LogLevel.INFO
    result = run_instruction(args.instruction, model, options, max_steps, verbosity)
    sit = result.situation
    print(f"\n=== Report from the agent ({result.run.state}) ===\n{result.run.output}")
    estimate = sit["best_estimate"]
    print(
        f"\n=== Verified situation (read from the server) ===\n"
        f"{sit['mission_state']} at {sit['sim_time_s']:.0f} s; best estimate "
        f"{estimate['bearing_from_base_deg']:.0f} deg, {estimate['distance_from_base_m']:.0f} m "
        f"from base; detections {[d['id'] for d in sit['detections']]}; "
        f"priority areas {[a['id'] for a in sit['priority_areas']]}"
    )
    for v in sit["vehicles"]:
        print(f"  {v['id']}: {v['status']}, {v['energy_pct']:.0f} %, task: {v['current_task']}")
    steps = sum(1 for step in result.run.steps if "step_number" in step)
    print(f"\nSteps: {steps} | wall time: {time.monotonic() - start:.1f} s")
    return 0 if result.run.state == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
