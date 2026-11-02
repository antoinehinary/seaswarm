"""Phase 2 figure: probability map vs a Monte Carlo cloud of the truth drift model.

    python scripts/belief_vs_particles.py --scenario scenarios/windy_mixed.yaml \
        --png media/belief_vs_particles.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import Normalize

from seaswarm.config import load_default_config, load_scenario
from seaswarm.rng import make_rngs
from seaswarm.search.predict import belief_at
from seaswarm.search.probability_map import ProbabilityMap, hellinger
from seaswarm.viz.gif import new_figure
from seaswarm.world.drift import drift_cloud


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--seed", type=int, help="overrides the scenario seed")
    parser.add_argument("--particles", type=int, default=5000)
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--png", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build both distributions, print the comparison and save the figure."""
    args = parse_args(argv)
    scenario = load_scenario(args.scenario)
    defaults = load_default_config()
    seed = scenario.seed if args.seed is None else args.seed
    t_s = args.hours * 3600.0

    belief = belief_at(scenario, defaults, t_s)
    xy = drift_cloud(scenario, defaults, args.particles, make_rngs(seed).particles, t_s)
    cloud = ProbabilityMap.from_samples(belief.grid, xy)
    error_m = np.linalg.norm(belief.mean_m() - cloud.mean_m())
    h_cells, h_tiles = hellinger(belief, cloud), hellinger(belief.coarsened(4), cloud.coarsened(4))
    print(f"centroid error {error_m:.0f} m; Hellinger {h_cells:.3f} (100 m), {h_tiles:.3f} (400 m)")

    g = belief.grid
    extent_km = np.array([g.x_min_m, g.x_edges_m[-1], g.y_min_m, g.y_edges_m[-1]]) / 1e3
    centre, half = belief.mean_m() / 1e3, 3.0 * np.sqrt(np.trace(belief.covariance_m2())) / 1e3
    norm = Normalize(0.0, max(belief.p.max(), cloud.p.max()))
    fig = new_figure(11.0, 5.2, dpi=100)
    axes = fig.subplots(1, 2, sharex=True, sharey=True)
    panels = [
        (belief, "probability map (predict steps)"),
        (cloud, f"{args.particles} drift particles"),
    ]
    for ax, (pmap, label) in zip(axes, panels, strict=True):
        image = ax.imshow(pmap.p, origin="lower", extent=extent_km, cmap="viridis", norm=norm)
        ax.plot(scenario.lkp.x_m / 1e3, scenario.lkp.y_m / 1e3, "wx", ms=9, mew=2)
        ax.set(title=label, xlabel="east (km)", aspect="equal")
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
    axes[0].set_ylabel("north (km)")
    axes[1].scatter(xy[:, 0] / 1e3, xy[:, 1] / 1e3, s=0.5, c="w", alpha=0.25)
    fig.colorbar(image, ax=axes, label="probability per 100 m cell", shrink=0.85)
    fig.suptitle(
        f"{scenario.id}, {args.hours:g} h after the fall: centroid error {error_m:.0f} m, "
        f"Hellinger {h_cells:.2f} (100 m cells) / {h_tiles:.2f} (400 m tiles)"
    )
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png)
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
