"""Phase 3 demo: run one search scenario with one strategy, log it and optionally render it.

    python scripts/run_search.py --scenario scenarios/windy_mixed.yaml --strategy greedy \
        --seed 3 --gif media/search_greedy_seed3.gif
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from seaswarm.config import load_default_config, load_scenario
from seaswarm.logging_utils import RunRecorder
from seaswarm.mission import SearchMission
from seaswarm.viz.gif import write_gif
from seaswarm.viz.replay import RunLog, render_frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--strategy", choices=["lawnmower", "greedy", "auction"], required=True)
    parser.add_argument("--seed", type=int, help="overrides the scenario seed")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--gif", type=Path, help="also render a replay GIF")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--frame-every-s", type=float, default=30.0)
    parser.add_argument("--show-truth", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run, print a summary, write the logs and the optional GIF."""
    args = parse_args(argv)
    start = time.perf_counter()
    scenario = load_scenario(args.scenario)
    if args.seed is not None:
        scenario = scenario.model_copy(update={"seed": args.seed})
    mission = SearchMission(scenario, load_default_config(), args.strategy)
    recorder = RunRecorder(mission)
    while not mission.done:
        mission.step()
        recorder.sample()
    sim_wall_s = time.perf_counter() - start

    run_dir = args.runs_dir / f"{scenario.id}_{args.strategy}_{scenario.seed}"
    meta = {
        "scenario": scenario.id,
        "strategy": args.strategy,
        "seed": scenario.seed,
        "state": mission.state.value,
        "time_to_detection_s": mission.time_to_detection_s,
        "time_to_recovery_s": mission.time_to_recovery_s,
    }
    recorder.write(run_dir, meta)
    minutes = (mission.t_s - mission.start_s) / 60.0
    label = f"{scenario.id} / {args.strategy} / seed {scenario.seed}"
    print(f"{label}: {mission.state.value} after {minutes:.0f} min")
    if mission.time_to_detection_s is not None:
        print(f"  detected after {mission.time_to_detection_s / 60:.1f} min")
    if mission.time_to_recovery_s is not None:
        print(f"  recovered after {mission.time_to_recovery_s / 60:.1f} min")
    violations = mission.guard.violations.total
    print(f"  POS_cum {mission.success.pos_cum:.2f}, refusals {len(mission.refusals)}, ", end="")
    print(f"violations {violations}")
    print(f"  simulation {sim_wall_s:.2f} s wall time; log in {run_dir}")
    if args.gif:
        frames = render_frames(
            RunLog.load(run_dir), args.frame_every_s, args.show_truth, log_scale=False
        )
        write_gif(frames, args.gif, args.fps)
        print(f"  wrote {args.gif} ({len(frames)} frames, {args.gif.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
