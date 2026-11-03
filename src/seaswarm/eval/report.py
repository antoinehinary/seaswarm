"""summary.md and plots from results.csv and pos_curves.npz (fixed bootstrap seed)."""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from seaswarm.eval.metrics import (
    bootstrap_ci,
    found_by,
    kaplan_meier,
    km_median_s,
    paired_difference,
)
from seaswarm.viz.plots import STYLE, small_multiples

MINUTES = (30, 60, 120)


def _load(out: Path) -> list[dict]:
    with (out / "results.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        r["detected"] = r["detected"] == "True"
        r["recovered"] = r["recovered"] == "True"
        for key in (
            "time_to_detection_s",
            "time_to_recovery_s",
            "pos_30min",
            "pos_60min",
            "pos_120min",
        ):
            r[key] = float(r[key])
        for key in (
            "refusals",
            "separation_violations",
            "geofence_violations",
            "stranded_vehicles",
        ):
            r[key] = int(r[key])
    return rows


def _ordered(names: set[str], order: list[str]) -> list[str]:
    return [n for n in order if n in names] + sorted(names - set(order))


def write_report(out: Path) -> None:
    """Write summary.md and plots/*.png into `out`."""
    rows = _load(out)
    strategies = _ordered({r["strategy"] for r in rows}, list(STYLE))
    fleets = _ordered({r["fleet"] for r in rows}, ["2D+1U", "4D", "2U", "4D+2U"])
    group = {
        (s, f): [r for r in rows if r["strategy"] == s and r["fleet"] == f]
        for s in strategies
        for f in fleets
    }
    n_seeds = len({r["seed"] for r in rows})
    limit_min = max(r["time_to_detection_s"] for r in rows) / 60.0

    lines = [
        "# SeaSwarm evaluation summary",
        "",
        f"{len(rows)} runs: {len(strategies)} strategies x {len(fleets)} fleets x {n_seeds} "
        f"generated scenarios (seed N = scenario gen_N), {limit_min:.0f} min limit. "
        "Found = first detection. "
        "Brackets: 95 % bootstrap interval over scenarios (2000 resamples, fixed seed).",
        "",
        "## Outcomes",
        "",
        "| Fleet | Strategy | Found <= 30 min | Found <= 60 min | Found <= 120 min "
        "| Median time to find | Recovered | POS_cum @ 60 min | Refusals / run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    km_curves: dict[str, dict] = {f: {} for f in fleets}
    for f in fleets:
        for s in strategies:
            g = group[(s, f)]
            if not g:
                continue
            times = [r["time_to_detection_s"] for r in g]
            found = [r["detected"] for r in g]
            cells = []
            for m in MINUTES:
                est, lo, hi = bootstrap_ci(found_by(times, found, m * 60.0))
                cells.append(f"{est:.0%} [{lo:.0%}, {hi:.0%}]")
            median = km_median_s(times, found)
            median_text = (
                f"{median / 60:.1f} min" if np.isfinite(median) else f"> {limit_min:.0f} min"
            )
            recovered = sum(r["recovered"] for r in g)
            pos = np.mean([r["pos_60min"] for r in g])
            refusals = np.mean([r["refusals"] for r in g])
            lines.append(
                f"| {f} | {s} | {' | '.join(cells)} | {median_text} | {recovered}/{len(g)} "
                f"| {pos:.2f} | {refusals:.1f} |"
            )
            knots, frac = kaplan_meier(times, found)
            km_curves[f][s] = (np.append(knots / 60.0, limit_min), np.append(frac, frac[-1]))

    lines += [
        "",
        "## Paired comparison (same scenarios and seeds)",
        "",
        "Difference in the fraction found within 60 min, A - B, with a bootstrap interval of the "
        "mean paired difference. Pairing removes the scenario-to-scenario variance.",
        "",
        "| Fleet | A - B | Difference | A faster / B faster / tie (time to find) |",
        "| --- | --- | --- | --- |",
    ]
    for f in fleets:
        for a, b in combinations(strategies, 2):
            ga = sorted(group[(a, f)], key=lambda r: r["seed"])
            gb = sorted(group[(b, f)], key=lambda r: r["seed"])
            if not ga or [r["seed"] for r in ga] != [r["seed"] for r in gb]:
                continue
            fa = found_by(
                [r["time_to_detection_s"] for r in ga], [r["detected"] for r in ga], 3600.0
            )
            fb = found_by(
                [r["time_to_detection_s"] for r in gb], [r["detected"] for r in gb], 3600.0
            )
            est, lo, hi = paired_difference(fa, fb)
            ta = np.array([r["time_to_detection_s"] for r in ga])
            tb = np.array([r["time_to_detection_s"] for r in gb])
            lines.append(
                f"| {f} | {a} - {b} | {est:+.0%} [{lo:+.0%}, {hi:+.0%}] "
                f"| {np.sum(ta < tb)} / {np.sum(tb < ta)} / {np.sum(ta == tb)} |"
            )

    total = {
        k: sum(r[k] for r in rows)
        for k in ("separation_violations", "geofence_violations", "stranded_vehicles")
    }
    lines += [
        "",
        "## Safety",
        "",
        f"Separation violations: {total['separation_violations']}. Geofence violations: "
        f"{total['geofence_violations']}. Stranded vehicles: {total['stranded_vehicles']}. "
        f"Refusals (all runs): {sum(r['refusals'] for r in rows)}.",
        "",
        "## Plots",
        "",
        "![Fraction found vs time](plots/found_vs_time.png)",
        "",
        "![POS_cum vs time](plots/pos_vs_time.png)",
        "",
        "## Interpretation",
        "",
        "_To be written by the author (3 sentences)._",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines))

    small_multiples(
        km_curves,
        fleets,
        "Fraction of persons found vs time (Kaplan-Meier)",
        "minutes since the search started",
        "fraction found",
        out / "plots" / "found_vs_time.png",
        step=True,
    )
    with np.load(out / "pos_curves.npz") as npz:
        pos_curves: dict[str, dict] = {f: {} for f in fleets}
        for f in fleets:
            for s in strategies:
                mask = (npz["strategy"] == s) & (npz["fleet"] == f)
                if mask.any():
                    pos_curves[f][s] = (npz["minutes"].astype(float), npz["pos"][mask].mean(axis=0))
    small_multiples(
        pos_curves,
        fleets,
        "Cumulative probability of success of the planned search (mean over scenarios)",
        "minutes since the search started",
        "POS_cum",
        out / "plots" / "pos_vs_time.png",
        step=False,
    )
