"""Replay frames from a run log: belief heatmap, vehicle tracks and footprints, detections.

Each frame's heatmap is scaled to its own maximum (relative probability): after a detection the
belief collapses to a ~10 m spot, which would otherwise wash out every earlier frame.

The true person is drawn only on request: the point of the replay is that the swarm searches
from its belief, not from the truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Circle, Rectangle

from seaswarm.viz.gif import Frame, figure_to_rgb, new_figure

TRAIL_S = 600.0  # length of the track drawn behind each vehicle
COLOURS = {"drone": "tab:orange", "usv": "tab:red"}
MARKERS = {"drone": "^", "usv": "s"}


@dataclass
class RunLog:
    """Contents of a run's log.npz."""

    data: dict[str, np.ndarray]
    meta: dict

    @classmethod
    def load(cls, run_dir: Path) -> RunLog:
        """Read `run_dir/log.npz`."""
        with np.load(run_dir / "log.npz") as npz:
            data = {key: npz[key] for key in npz.files}
        return cls(data, json.loads(str(data["meta"])))

    def view_km(self, margin_m: float = 400.0) -> tuple[tuple[float, float], tuple[float, float]]:
        """Square view (km) around the vehicles' search tracks and the belief's mass."""
        d = self.data
        base = d["base_m"]
        xy = d["xy_m"].reshape(-1, 2)
        away = xy[np.hypot(*(xy - base).T) > 500.0]
        belief = d["belief"]
        x0, x1, y0, y1 = d["grid_extent_m"]
        ny, nx = belief.shape[1:]
        xs = np.linspace(x0, x1, nx + 1)[:-1] + (x1 - x0) / nx / 2
        ys = np.linspace(y0, y1, ny + 1)[:-1] + (y1 - y0) / ny / 2
        heavy = belief.max(axis=0) > 0.2 * belief.max()
        rows, cols = np.nonzero(heavy)
        pts = np.vstack([away, np.column_stack([xs[cols], ys[rows]])])
        lo, hi = pts.min(axis=0) - margin_m, pts.max(axis=0) + margin_m
        centre, half = (lo + hi) / 2, max(hi - lo) / 2
        return tuple((centre[0] + [-half, half]) / 1e3), tuple((centre[1] + [-half, half]) / 1e3)


def render_frames(
    log: RunLog, frame_every_s: float, show_truth: bool, log_scale: bool, dpi: int = 90
) -> list[Frame]:
    """One frame every `frame_every_s` simulated seconds."""
    d = log.data
    t = d["t_s"]
    stride = max(1, round(frame_every_s / (t[1] - t[0]))) if len(t) > 1 else 1
    extent_km = d["grid_extent_m"] / 1e3
    norm = LogNorm(1e-3, 1.0) if log_scale else Normalize(0.0, 1.0)

    fig = new_figure(7.0, 6.4, dpi=dpi)
    ax = fig.add_axes((0.1, 0.08, 0.78, 0.84))
    xlim, ylim = log.view_km()
    ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="east (km)", ylabel="north (km)")
    heat = ax.imshow(
        _relative(d["belief"][0]), origin="lower", extent=extent_km, cmap="viridis", norm=norm
    )
    ax.set_facecolor(heat.cmap(0.0))
    fig.colorbar(heat, ax=ax, fraction=0.035, pad=0.01, label="relative probability")
    fx0, fx1, fy0, fy1 = d["fence_m"] / 1e3
    ax.add_patch(Rectangle((fx0, fy0), fx1 - fx0, fy1 - fy0, fill=False, ls="--", color="w", lw=1))
    ax.plot(*d["base_m"] / 1e3, "wP", ms=10, mec="k", label="base")
    trails, dots, rings = [], [], []
    for i, vtype in enumerate(d["vehicle_types"]):
        colour = COLOURS[str(vtype)]
        trails.append(ax.plot([], [], "-", color=colour, lw=1, alpha=0.8)[0])
        label = str(vtype) if str(vtype) not in [str(v) for v in d["vehicle_types"][:i]] else None
        dots.append(
            ax.plot([], [], MARKERS[str(vtype)], color=colour, mec="k", ms=8, label=label)[0]
        )
        rings.append(
            ax.add_patch(Circle((0, 0), d["r_max_m"][i] / 1e3, fill=False, color=colour, lw=1))
        )
    found = ax.plot([], [], "wx", ms=10, mew=2, label="detection")[0]
    truth = (
        ax.plot([], [], "*", color="magenta", ms=12, mec="k", label="true person")[0]
        if show_truth
        else None
    )
    ax.legend(loc="upper left", fontsize=7, framealpha=0.6)
    title = ax.set_title("", fontsize=9)

    frames = []
    for k in range(0, len(t), stride):
        now = t[k]
        b = np.searchsorted(d["belief_t_s"], now, side="right") - 1
        heat.set_data(_relative(d["belief"][b]))
        start = np.searchsorted(t, now - TRAIL_S)
        for i in range(len(d["vehicle_ids"])):
            path = d["xy_m"][start : k + 1, i] / 1e3
            trails[i].set_data(path[:, 0], path[:, 1])
            dots[i].set_data([path[-1, 0]], [path[-1, 1]])
            rings[i].center = (path[-1, 0], path[-1, 1])
        seen = d["detections"][d["detections"][:, 0] <= now]
        found.set_data(seen[:, 1] / 1e3, seen[:, 2] / 1e3)
        if truth is not None:
            truth.set_data([d["truth_m"][k, 0] / 1e3], [d["truth_m"][k, 1] / 1e3])
        minutes = (now - t[0]) / 60.0
        state = "FOUND" if len(seen) else "SEARCHING"
        title.set_text(
            f"{log.meta['scenario']} / {log.meta['strategy']} / seed {log.meta['seed']} — "
            f"{minutes:.0f} min — {state} — POS {d['pos_cum'][k]:.0%}"
        )
        frames.append(figure_to_rgb(fig))
    return frames


def _relative(p: np.ndarray) -> np.ndarray:
    """Probabilities scaled to the frame's maximum, floored for the log scale."""
    return np.maximum(p / max(float(p.max()), 1e-300), 1e-3)
