"""Phase 3 check: compare search strategies over seeds; print a markdown table.

Two runs per (strategy, seed):
- a full run (stops at recovery or timeout) for the safety criteria;
- a `search_only` run cut at --minutes for POS_cum and "detected by then" on equal footing.

POS_cum is a belief metric: in search_only mode nothing about the truth feeds back into the
belief or the plans, so it is the same for every seed. The seeds change where the person really
is, hence whether and when they are detected.

    python scripts/compare_strategies.py --scenario scenarios/windy_mixed.yaml --seeds 0-19
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from seaswarm.config import Scenario, load_default_config, load_scenario
from seaswarm.mission import MissionState, SearchMission


def parse_seeds(text: str) -> list[int]:
    """'0-19' or '1,4,7' -> list of ints."""
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--seeds", default="0-19")
    parser.add_argument("--strategies", default="lawnmower,greedy,auction")
    parser.add_argument("--minutes", type=float, default=60.0)
    parser.add_argument("--fleet", default="2,0,1", help="drones_eo,drones_thermal,usvs")
    return parser.parse_args(argv)


def _variant(scenario: Scenario, seed: int, fleet: list[int], max_time_s: float | None) -> Scenario:
    raw = scenario.model_dump()
    raw["seed"] = seed
    raw["fleet"] = dict(zip(("drones_eo", "drones_thermal", "usvs"), fleet, strict=True))
    if max_time_s is not None:
        raw["limits"]["max_sim_time_s"] = max_time_s
    return Scenario.model_validate(raw)


def main(argv: list[str] | None = None) -> int:
    """Run every (strategy, seed) twice and print the comparison."""
    args = parse_args(argv)
    defaults = load_default_config()
    base = load_scenario(args.scenario)
    fleet = [int(n) for n in args.fleet.split(",")]
    seeds = parse_seeds(args.seeds)
    start = time.perf_counter()
    print(
        f"| {base.id}, fleet {args.fleet}, {len(seeds)} seeds | detected <= {args.minutes:.0f} min "
        f"| POS_cum @ {args.minutes:.0f} min | median detection (min) | recovered (full run) "
        "| violations | stranded | max steps |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for strategy in args.strategies.split(","):
        detected, pos, ttd, recovered, violations, stranded, steps = [], [], [], 0, 0, 0, 0
        for seed in seeds:
            full = SearchMission(_variant(base, seed, fleet, None), defaults, strategy)
            full.run()
            recovered += full.state is MissionState.RECOVERED
            violations += full.guard.violations.separation + full.guard.violations.geofence
            stranded += full.guard.violations.stranded
            steps = max(steps, full.steps)
            assert full.steps <= full.scenario.limits.max_steps

            short = SearchMission(
                _variant(base, seed, fleet, args.minutes * 60.0),
                defaults,
                strategy,
                search_only=True,
            )
            short.run()
            pos.append(short.success.pos_cum)
            hit = short.time_to_detection_s is not None
            detected.append(hit)
            ttd.append(short.time_to_detection_s / 60.0 if hit else np.inf)
        n = len(seeds)
        median = np.median(ttd)
        median_text = f"{median:.1f}" if np.isfinite(median) else f"> {args.minutes:.0f}"
        print(
            f"| {strategy} | {sum(detected)}/{n} | {np.mean(pos):.3f} (spread {np.ptp(pos):.0e}) "
            f"| {median_text} | {recovered}/{n} | {violations} | {stranded} | {steps} |"
        )
    print(f"\n({time.perf_counter() - start:.0f} s wall time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
