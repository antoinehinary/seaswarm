"""Evaluate the operator agent on plain-language instructions with checkable outcomes.

    python -m seaswarm.eval.agent_eval
    python -m seaswarm.eval.agent_eval --only north_sector,recall_drone --model ollama_chat/qwen3:8b

Each instruction runs on a fresh mission (its own server process). It passes if every check
holds on the state read back from the server after the agent finished, never on the agent's own
claims, except for the checks that are about the report itself. Writes results/agent_eval.md.
"""

from __future__ import annotations

import argparse
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from smolagents import LogLevel, Model

from seaswarm.agent.cli import OperatorResult, ServerOptions, run_instruction
from seaswarm.agent.models import build_model
from seaswarm.config import load_agent_config
from seaswarm.mcp_server.server import make_operations
from seaswarm.operations import Operations

INSTRUCTIONS = Path("scenarios/agent_instructions.yaml")
DEFAULT_SCENARIO = Path("scenarios/windy_mixed.yaml")


@dataclass
class Case:
    """One instruction with its mission setup and checks."""

    id: str
    instruction: str
    checks: list[dict[str, Any]]
    scenario: Path = DEFAULT_SCENARIO
    seed: int | None = None
    strategy: str = "auction"
    warmup_s: float = 0.0

    @property
    def options(self) -> ServerOptions:
        """How to start this case's MCP server."""
        return ServerOptions(self.scenario, self.seed, self.strategy, self.warmup_s)


@dataclass
class Outcome:
    """Result of one case."""

    case: Case
    passed: bool
    failures: list[str] = field(default_factory=list)
    steps: int = 0
    wall_s: float = 0.0
    report: str = ""


def load_cases(path: Path = INSTRUCTIONS) -> list[Case]:
    """Read the instruction file."""
    raw = yaml.safe_load(path.read_text())
    return [Case(**{**c, "scenario": Path(c.get("scenario", DEFAULT_SCENARIO))}) for c in raw]


def _vehicle(situation: dict, vehicle_id: str) -> dict:
    return next(v for v in situation["vehicles"] if v["id"] == vehicle_id)


def _same_rect(rect: list[float], target: list[float]) -> bool:
    return all(abs(u - v) < 1.0 for u, v in zip(rect, target, strict=True))


def _check_sector(check: dict, sit: dict, start: Operations) -> str | None:
    r = start.sector_rect(check["sector"])
    target = [r.x_min_m, r.y_min_m, r.x_max_m, r.y_max_m]
    areas = [
        a for a in sit["priority_areas"] if a["weight"] > 1.0 and _same_rect(a["rect_m"], target)
    ]
    if not areas:
        return f"no priority area with weight > 1 on sector {check['sector']}"
    lasts = areas[0]["expires_in_s"]
    if "max_duration_s" in check and lasts > check["max_duration_s"]:
        return f"priority lasts {lasts:.0f} s, expected <= {check['max_duration_s']} s"
    return None


def _check_search_near(check: dict, sit: dict, start: Operations) -> str | None:
    task = _vehicle(sit, check["vehicle"])["current_task"] or ""
    match = re.search(r"around \((-?\d+), (-?\d+)\)", task)
    if not match:
        return f"{check['vehicle']} has no operator search task: {task!r}"
    x, y = float(match[1]), float(match[2])
    if "dx_from_estimate_m" in check:
        estimate = start.situation()["best_estimate"]
        want_x = estimate["x_m"] + check["dx_from_estimate_m"]
        want_y = estimate["y_m"] + check["dy_from_estimate_m"]
    else:
        want_x = start.m.base_xy_m[0] + check["x_from_base_m"]
        want_y = start.m.base_xy_m[1] + check["y_from_base_m"]
    if math.hypot(x - want_x, y - want_y) <= check["tolerance_m"]:
        return None
    return f"searches around ({x:.0f}, {y:.0f}), expected ({want_x:.0f}, {want_y:.0f})"


def _check_report_bearing(check: dict, report: str, start: Operations) -> str | None:
    want = start.situation()["best_estimate"]["bearing_from_base_deg"]
    for number in re.findall(r"\d+(?:\.\d+)?", report):
        gap = abs(float(number) - want) % 360.0
        if min(gap, 360.0 - gap) <= check["tolerance_deg"]:
            return None
    return f"no bearing within {check['tolerance_deg']} deg of {want:.0f} in the report"


def evaluate_check(check: dict, result: OperatorResult, start: Operations) -> str | None:
    """None if the check holds, else why it failed.

    Args:
        check: One check from the instruction file.
        result: The agent's run and the situation read back from the server afterwards.
        start: The same mission, rebuilt in-process at the moment the agent started (missions
            are deterministic), for expectations relative to the initial state.
    """
    sit, report = result.situation, str(result.run.output)
    match check["type"]:
        case "sector_priority":
            return _check_sector(check, sit, start)
        case "vehicle_status":
            status = _vehicle(sit, check["vehicle"])["status"]
            return None if status in check["status"] else f"{check['vehicle']} is {status}"
        case "vehicle_task":
            task = _vehicle(sit, check["vehicle"])["current_task"] or ""
            if "contains" in check and check["contains"] not in task:
                return f"{check['vehicle']} task is {task!r}"
            if "not_contains" in check and check["not_contains"] in task:
                return f"{check['vehicle']} task is {task!r}"
            return None
        case "operator_search_near":
            return _check_search_near(check, sit, start)
        case "strategy":
            return None if sit["strategy"] == check["name"] else f"strategy is {sit['strategy']}"
        case "sim_time_at_least":
            simulated = sit["sim_time_s"] - start.situation()["sim_time_s"]
            return None if simulated >= check["seconds"] else f"only {simulated:.0f} s simulated"
        case "report_mentions":
            if any(word.lower() in report.lower() for word in check["any"]):
                return None
            return f"report mentions none of {check['any']}"
        case "report_bearing":
            return _check_report_bearing(check, report, start)
    return f"unknown check {check['type']}"


def run_case(case: Case, model: Model, max_steps: int) -> Outcome:
    """Run one instruction on a fresh mission and evaluate its checks."""
    start = make_operations(case.scenario, case.seed, case.strategy, case.warmup_s)
    t0 = time.monotonic()
    try:
        result = run_instruction(case.instruction, model, case.options, max_steps, LogLevel.OFF)
    except Exception as error:  # an agent crash is a failed case, not a crashed evaluation
        return Outcome(case, False, [f"agent error: {error}"], wall_s=time.monotonic() - t0)
    failures = [f for c in case.checks if (f := evaluate_check(c, result, start)) is not None]
    steps = sum(1 for step in result.run.steps if "step_number" in step)
    wall_s = time.monotonic() - t0
    return Outcome(case, not failures, failures, steps, wall_s, str(result.run.output))


def write_report(outcomes: list[Outcome], model_id: str, path: Path) -> None:
    """Markdown table of the outcomes."""
    passed = sum(o.passed for o in outcomes)
    lines = [
        "# Operator agent evaluation",
        "",
        f"Model `{model_id}`: **{passed}/{len(outcomes)}** instructions passed.",
        "",
        "| Instruction | Result | Steps | Time | Why it failed |",
        "| --- | --- | --- | --- | --- |",
    ]
    for o in outcomes:
        verdict = "pass" if o.passed else "FAIL"
        why = "; ".join(o.failures).replace("|", "/")
        lines.append(f"| {o.case.instruction} | {verdict} | {o.steps} | {o.wall_s:.0f} s | {why} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the cases and write results/agent_eval.md."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Operator agent evaluation.")
    parser.add_argument("--model", help="LiteLLM model id (default: config/agent.yaml)")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--out", type=Path, default=Path("results/agent_eval.md"))
    args = parser.parse_args(argv)
    config = load_agent_config()
    if args.model:
        config = config.model_copy(update={"model_id": args.model})
    model = build_model(config)
    cases = load_cases()
    if args.only:
        cases = [c for c in cases if c.id in args.only.split(",")]
    outcomes = []
    for case in cases:
        outcome = run_case(case, model, config.max_steps)
        verdict = "pass" if outcome.passed else "FAIL"
        detail = "; ".join(outcome.failures)
        print(f"{verdict}  {case.id} ({outcome.steps} steps, {outcome.wall_s:.0f} s) {detail}")
        outcomes.append(outcome)
    write_report(outcomes, config.model_id, args.out)
    print(f"\n{sum(o.passed for o in outcomes)}/{len(outcomes)} passed; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
