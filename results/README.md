# Evaluation results

Only this file and `summary.md` are committed; everything else here is regenerated.

```bash
python scripts/generate_scenarios.py --n 60 --seed 0
python -m seaswarm.eval.run --strategy lawnmower,greedy,auction --fleet all --seeds 0-49 --workers 4
```

| File | Content |
| --- | --- |
| `results.csv` | one row per run (strategy, fleet, seed): detection and recovery times (censored at the 3 h limit), POS_cum at 30 / 60 / 120 min, refusals, tile conflicts, safety violations |
| `pos_curves.npz` | POS_cum every minute of each run's 120 min search-only run |
| `summary.md` | outcome table with 95 % bootstrap intervals, paired comparisons, safety totals |
| `plots/found_vs_time.png` | fraction of persons found vs time (Kaplan-Meier), one panel per fleet |
| `plots/pos_vs_time.png` | mean POS_cum(t) of the planned search, one panel per fleet |

`results.csv` is byte-identical for any `--workers` value: each run is a pure function of
(strategy, fleet, seed) and rows are sorted.

Fleets: `2D+1U` (2 EO drones + 1 USV), `4D`, `2U`, `4D+2U`. Seed N runs scenario
`scenarios/generated/gen_N.yaml`.

CBBA comparison (decentralised auction at several radio ranges), written to `results/cbba/`; its
table is reproduced in `docs/design.md` section 5.7:

```bash
python -m seaswarm.eval.run --strategy auction,cbba@inf,cbba@3000,cbba@1000,cbba@300 --fleet 2D+1U,4D+2U --seeds 0-49 --workers 4 --out results/cbba
```
