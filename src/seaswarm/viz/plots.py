"""Evaluation plots: fraction found vs time (Kaplan-Meier) and POS(t), one panel per fleet.

Strategy colours are the first three slots of a validated categorical palette, in a fixed order
(colour follows the strategy, never its rank). The aqua slot is below 3:1 contrast on the light
surface, so every line also carries a direct label and its own line style, and every number is
repeated as a table in summary.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes

from seaswarm.viz.gif import new_figure

STYLE = {  # strategy -> (colour, line style); fixed order: lawnmower, greedy, auction
    "lawnmower": ("#2a78d6", (0, (1, 1.5))),
    "greedy": ("#eb6834", (0, (5, 2))),
    "auction": ("#1baf7a", "-"),
}
SURFACE = "#fcfcfb"
INK = "#1f1f1e"
MUTED = "#6b6a63"
GRID = "#e4e3de"


def _style_axes(ax: Axes, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _label_ends(ax: Axes, ends: dict[str, tuple[float, float]], gap: float = 0.06) -> None:
    """Direct labels at the line ends, spread at least `gap` apart and kept inside [0, 1].

    Text is in ink, not in the series colour; the line style beside it carries the identity.
    """
    names = sorted(ends, key=lambda n: ends[n][1])
    ys = [ends[n][1] for n in names]
    for i in range(1, len(ys)):  # push up to keep the gap
        ys[i] = max(ys[i], ys[i - 1] + gap)
    overflow = max(0.0, ys[-1] - 1.0) if ys else 0.0
    for name, y_text in zip(names, (y - overflow for y in ys), strict=True):  # then shift down
        x, y = ends[name]
        ax.annotate(
            name,
            (x, y),
            xytext=(x + 2.0, y_text),
            fontsize=8,
            color=INK,
            va="center",
            annotation_clip=False,
        )


def small_multiples(
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    fleets: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    step: bool,
) -> None:
    """2 x 2 panels (one per fleet) of one curve per strategy; same axes everywhere."""
    fig = new_figure(10.0, 7.2, dpi=110)
    fig.patch.set_facecolor(SURFACE)
    axes = fig.subplots(2, 2, sharex=True, sharey=True).ravel()
    for ax, fleet in zip(axes, fleets, strict=False):
        _style_axes(ax, f"Fleet {fleet}")
        ends = {}
        for strategy, (colour, dash) in STYLE.items():
            if strategy not in curves.get(fleet, {}):
                continue
            x, y = curves[fleet][strategy]
            ax.plot(
                x,
                y,
                color=colour,
                linestyle=dash,
                linewidth=2.0,
                label=strategy,
                drawstyle="steps-post" if step else "default",
            )
            ends[strategy] = (float(x[-1]), float(y[-1]))
        _label_ends(ax, ends)
        ax.set_ylim(0.0, 1.0)
    for ax in axes[len(fleets) :]:
        ax.set_visible(False)
    for ax in axes[2:]:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    for ax in axes[::2]:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, x=0.06, ha="left", fontsize=12, color=INK)
    fig.subplots_adjust(left=0.08, right=0.93, top=0.88, bottom=0.08, hspace=0.25, wspace=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
