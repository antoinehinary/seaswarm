"""Render a GIF replay from a run directory written by run_search.py.

python scripts/render_replay.py runs/windy_mixed_greedy_3 --gif media/search_greedy_seed3.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

from seaswarm.viz.gif import write_gif
from seaswarm.viz.replay import RunLog, render_frames


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--gif", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--frame-every-s", type=float, default=30.0)
    parser.add_argument("--show-truth", action="store_true", help="draw the true person")
    parser.add_argument("--log-scale", action="store_true", help="log colour scale for the map")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Render and write the GIF."""
    args = parse_args(argv)
    frames = render_frames(
        RunLog.load(args.run_dir), args.frame_every_s, args.show_truth, args.log_scale
    )
    write_gif(frames, args.gif, args.fps)
    print(f"wrote {args.gif} ({len(frames)} frames, {args.gif.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
