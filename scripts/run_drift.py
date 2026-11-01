"""Phase 1 demo: drift a cloud of possible person-in-water positions and render it as a GIF.

Each particle is one independent realisation of the fall position (Gaussian around the LKP) and
of every noise process, so the cloud is a sample of where the person may be.

    python scripts/run_drift.py --scenario scenarios/calm_single.yaml --seed 1 \
        --gif media/drift_seed1.gif
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from seaswarm.config import load_default_config, load_scenario
from seaswarm.rng import make_rngs
from seaswarm.units import mps_to_knots, yaw_rad_to_bearing_deg
from seaswarm.viz.gif import figure_to_rgb, new_figure, write_gif
from seaswarm.world.drift import ParticleDrift, sample_around_lkp, simulate
from seaswarm.world.target import PersonInWater


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="overrides the scenario seed")
    parser.add_argument("--particles", type=int, default=500)
    parser.add_argument("--duration-s", type=float, help="default: the scenario time limit")
    parser.add_argument("--frame-every-s", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--gif", type=Path, help="write the animation here")
    parser.add_argument("--show-truth", action="store_true", help="mark the true person")
    return parser.parse_args(argv)


def _bearing_and_knots(u_mps: float, v_mps: float) -> str:
    bearing = yaw_rad_to_bearing_deg(np.arctan2(v_mps, u_mps))
    return f"{mps_to_knots(np.hypot(u_mps, v_mps)):.1f} kn towards {bearing:03.0f}°"


def _view_limits(frames_m: np.ndarray, lkp_m: np.ndarray) -> tuple[tuple, tuple]:
    """Square view (km) around 99 % of all particle positions and the LKP."""
    pts = np.vstack([frames_m.reshape(-1, 2), lkp_m[None]])
    lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
    centre, half = (lo + hi) / 2, max(hi - lo) / 2 * 1.2 + 200.0
    return tuple((centre[0] + [-half, half]) / 1e3), tuple((centre[1] + [-half, half]) / 1e3)


def render(
    times_s: np.ndarray,
    frames_m: np.ndarray,
    truth_m: np.ndarray | None,
    lkp_m: np.ndarray,
    title: str,
    caption: str,
) -> list[np.ndarray]:
    """Draw one frame per recorded time: particle cloud, LKP and optionally the true person."""
    fig = new_figure(6.0, 6.0, dpi=90)
    ax = fig.add_axes((0.12, 0.1, 0.83, 0.8))
    xlim, ylim = _view_limits(frames_m, lkp_m)
    ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="east (km)", ylabel="north (km)")
    ax.grid(alpha=0.3)
    ax.plot(*lkp_m / 1e3, "kx", ms=9, mew=2, label="last known position")
    cloud = ax.scatter([], [], s=4, alpha=0.4, color="tab:blue", label="possible positions")
    centroids_km = frames_m.mean(axis=1) / 1e3
    (track,) = ax.plot([], [], "-", color="navy", lw=1.5, label="cloud centroid")
    if truth_m is not None:
        (truth_dot,) = ax.plot([], [], "*", color="tab:red", ms=13, mec="k", label="true person")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.99, 0.01, caption, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    heading = ax.set_title("")
    images = []
    for i, t_s in enumerate(times_s):
        cloud.set_offsets(frames_m[i] / 1e3)
        track.set_data(centroids_km[: i + 1, 0], centroids_km[: i + 1, 1])
        if truth_m is not None:
            truth_dot.set_data([truth_m[i, 0] / 1e3], [truth_m[i, 1] / 1e3])
        hours, minutes = divmod(int(round(t_s / 60.0)), 60)
        heading.set_text(f"{title} — {hours}:{minutes:02d} after the fall")
        images.append(figure_to_rgb(fig))
    return images


def main(argv: list[str] | None = None) -> int:
    """Simulate the drift, print a summary and optionally write the GIF."""
    args = parse_args(argv)
    start_wall = time.perf_counter()
    scenario = load_scenario(args.scenario)
    defaults = load_default_config()
    seed = scenario.seed if args.seed is None else args.seed
    rngs = make_rngs(seed)
    dt_s = defaults.sim.dt_s
    duration_s = args.duration_s or scenario.limits.max_sim_time_s
    max_steps = scenario.limits.max_steps

    fall_m = sample_around_lkp(scenario.lkp, args.particles, rngs.particles)
    cloud = ParticleDrift.from_scenario(scenario, defaults, fall_m, rngs.particles)
    times_s, frames_m = simulate(cloud, duration_s, dt_s, args.frame_every_s, max_steps)

    person = PersonInWater.from_scenario(scenario, defaults, rngs)
    truth_m = [person.position_m]
    steps_per_frame = round(args.frame_every_s / dt_s)
    for _ in range(len(times_s) - 1):
        for _ in range(steps_per_frame):
            person.step(dt_s)
        truth_m.append(person.position_m)

    lkp_m = np.array([scenario.lkp.x_m, scenario.lkp.y_m])
    centroid_m = frames_m[-1].mean(axis=0)
    shift_m = centroid_m - frames_m[0].mean(axis=0)
    spread_m = np.sqrt(np.linalg.eigvalsh(np.cov(frames_m[-1].T)))
    sim_wall_s = time.perf_counter() - start_wall
    print(f"scenario {scenario.id}, seed {seed}, {args.particles} particles, {duration_s:.0f} s")
    print(f"centroid moved {np.hypot(*shift_m):.0f} m: {_bearing_and_knots(*shift_m / duration_s)}")
    print(f"final spread (1 sigma): {spread_m[1]:.0f} m x {spread_m[0]:.0f} m")
    print(f"truth ends {np.hypot(*(truth_m[-1] - centroid_m)):.0f} m from the cloud centroid")
    print(f"simulation: {sim_wall_s:.2f} s wall time")

    if args.gif:
        wind = scenario.wind
        current_mps = cloud.current.velocity(lkp_m[None], 0.0)[0]
        caption = (
            f"wind {mps_to_knots(wind.speed_mps):.0f} kn from {wind.from_bearing_deg:03.0f}°\n"
            f"current at LKP {_bearing_and_knots(*current_mps)}"
        )
        truth = np.array(truth_m) if args.show_truth else None
        images = render(times_s, frames_m, truth, lkp_m, scenario.id, caption)
        write_gif(images, args.gif, args.fps)
        size_mb = args.gif.stat().st_size / 1e6
        print(f"wrote {args.gif} ({len(images)} frames, {size_mb:.1f} MB)")
    print(f"total: {time.perf_counter() - start_wall:.2f} s wall time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
