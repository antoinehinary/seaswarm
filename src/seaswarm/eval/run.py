"""Batch evaluation: strategies x fleets x seeds, in parallel, with reproducible output.

    python -m seaswarm.eval.run --strategy lawnmower,greedy,auction --fleet all \
        --seeds 0-49 --workers 4

Seed N runs scenario `scenarios/generated/gen_N.yaml`. Every job is a pure function of
(strategy, fleet, seed), results come back in job order and rows are sorted, so results.csv is
byte-identical for any --workers value.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from dataclasses import asdict, dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from seaswarm.config import Scenario, load_default_config, load_scenario
from seaswarm.mission import STRATEGIES, MissionState, SearchMission

FLEETS = {
    "2D+1U": {"drones_eo": 2, "drones_thermal": 0, "usvs": 1},
    "4D": {"drones_eo": 4, "drones_thermal": 0, "usvs": 0},
    "2U": {"drones_eo": 0, "drones_thermal": 0, "usvs": 2},
    "4D+2U": {"drones_eo": 4, "drones_thermal": 0, "usvs": 2},
}
POS_MINUTES = 120  # length of the search-only run that measures POS(t)
POS_SAMPLE_S = 60.0


@dataclass(frozen=True)
class Job:
    """One evaluation run."""

    strategy: str
    fleet: str
    seed: int
    scenarios_dir: str


@dataclass(frozen=True)
class Row:
    """One line of results.csv."""

    strategy: str
    fleet: str
    seed: int
    scenario: str
    daylight: bool
    wind_mps: float
    current_mps: float
    detected: bool
    time_to_detection_s: float  # censored at the time limit when not detected
    recovered: bool
    time_to_recovery_s: float  # censored at the time limit when not recovered
    pos_30min: float
    pos_60min: float
    pos_120min: float
    refusals: int
    tile_conflicts: int  # TILE_OCCUPIED refusals: two vehicles planned onto the same tile
    separation_violations: int
    geofence_violations: int
    stranded_vehicles: int
    steps: int


def parse_seeds(text: str) -> list[int]:
    """'0-49', '3' or '1,4,7' -> sorted seeds."""
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return sorted(int(s) for s in text.split(","))


def _scenario(job: Job) -> Scenario:
    scenario = load_scenario(Path(job.scenarios_dir) / f"gen_{job.seed:03d}.yaml")
    raw = scenario.model_dump()
    raw["fleet"] = FLEETS[job.fleet]
    return Scenario.model_validate(raw)


def run_job(job: Job) -> tuple[Row, list[float]]:
    """Full run (detection, recovery, safety) and search-only run (POS curve) of one job."""
    defaults = load_default_config()
    scenario = _scenario(job)
    full = SearchMission(scenario, defaults, job.strategy)
    full.run()
    limit_s = scenario.limits.max_sim_time_s
    detected = full.time_to_detection_s is not None
    recovered = full.state is MissionState.RECOVERED

    raw = scenario.model_dump()
    raw["limits"]["max_sim_time_s"] = POS_MINUTES * 60.0
    search = SearchMission(Scenario.model_validate(raw), defaults, job.strategy, search_only=True)
    curve = [0.0]
    while not search.done:
        search.step()
        if (search.t_s - search.start_s) % POS_SAMPLE_S == 0.0:
            curve.append(search.success.pos_cum)

    v = full.guard.violations
    row = Row(
        strategy=job.strategy,
        fleet=job.fleet,
        seed=job.seed,
        scenario=scenario.id,
        daylight=scenario.daylight,
        wind_mps=scenario.wind.speed_mps,
        current_mps=scenario.current.speed_mps,
        detected=detected,
        time_to_detection_s=full.time_to_detection_s if detected else limit_s,
        recovered=recovered,
        time_to_recovery_s=full.time_to_recovery_s if recovered else limit_s,
        pos_30min=round(curve[30], 6),
        pos_60min=round(curve[60], 6),
        pos_120min=round(curve[120], 6),
        refusals=len(full.refusals),
        tile_conflicts=sum(r.code.value == "TILE_OCCUPIED" for r in full.refusals),
        separation_violations=v.separation,
        geofence_violations=v.geofence,
        stranded_vehicles=v.stranded,
        steps=full.steps,
    )
    return row, curve


def run_all(jobs: list[Job], workers: int) -> list[tuple[Row, list[float]]]:
    """Run every job; results in job order whatever the number of workers."""
    if workers <= 1:
        return [run_job(job) for job in jobs]
    with Pool(workers) as pool:
        return pool.map(run_job, jobs, chunksize=1)


def write_results(results: list[tuple[Row, list[float]]], out: Path) -> None:
    """results.csv (one row per run) and pos_curves.npz (POS every minute)."""
    out.mkdir(parents=True, exist_ok=True)
    results = sorted(results, key=lambda r: (r[0].strategy, r[0].fleet, r[0].seed))
    with (out / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0][0])))
        writer.writeheader()
        for row, _ in results:
            writer.writerow(asdict(row))
    np.savez_compressed(
        out / "pos_curves.npz",
        strategy=np.array([r.strategy for r, _ in results]),
        fleet=np.array([r.fleet for r, _ in results]),
        seed=np.array([r.seed for r, _ in results]),
        minutes=np.arange(POS_MINUTES + 1),
        pos=np.array([c for _, c in results]),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description="Batch evaluation of search strategies.")
    parser.add_argument("--strategy", default=",".join(STRATEGIES), help="comma-separated")
    parser.add_argument(
        "--fleet", default="all", help=f"'all' or comma-separated from {list(FLEETS)}"
    )
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", default="0-49", help="e.g. 0-49 or 1,4,7")
    seeds.add_argument("--seed", type=int, help="a single seed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--scenarios-dir", default="scenarios/generated")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--no-report", action="store_true", help="skip summary.md and plots")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the grid, write results.csv, pos_curves.npz, summary.md and plots."""
    logging.basicConfig(level=logging.WARNING)
    args = parse_args(argv)
    strategies = args.strategy.split(",")
    unknown = {s for s in strategies if s.partition("@")[0] not in STRATEGIES}
    if unknown:
        raise SystemExit(f"unknown strategy {sorted(unknown)}; choose from {list(STRATEGIES)}")
    fleets = list(FLEETS) if args.fleet == "all" else args.fleet.split(",")
    seeds = [args.seed] if args.seed is not None else parse_seeds(args.seeds)
    jobs = [Job(s, f, n, args.scenarios_dir) for s in strategies for f in fleets for n in seeds]
    grid = f"{len(strategies)} strategies x {len(fleets)} fleets x {len(seeds)} seeds"
    print(f"{len(jobs)} runs ({grid}), {args.workers} workers")
    start = time.perf_counter()
    results = run_all(jobs, args.workers)
    write_results(results, args.out)
    print(f"wrote {args.out / 'results.csv'} in {time.perf_counter() - start:.0f} s")
    if not args.no_report:
        from seaswarm.eval.report import write_report

        write_report(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
