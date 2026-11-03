# SeaSwarm

A heterogeneous swarm of drones and autonomous surface vessels searches for a person in water.
SeaSwarm predicts the drift, keeps a Bayesian probability map of where the person may be, splits
the search between vehicles and replans after every sweep, while a human operator directs it in
plain language through an LLM agent over MCP.

Second project of a series on autonomy for drones, maritime and space systems, after
[skypilot-agent](https://github.com/antoinehinary/skypilot-agent).

![500 possible positions of a person in water drifting for 3 hours](media/drift_seed1.gif)

*500 equally likely realisations of the drift of a person in water over 3 hours: uncertain fall
position, current, and wind-driven leeway with time-correlated noise and random jibing.*

> Work in progress. Not an operational search-and-rescue tool: all drift and detection
> parameters are illustrative values from public literature.

## Quick start

```bash
conda create -n seaswarm python=3.12 -y && conda activate seaswarm
pip install -e ".[dev]"
python scripts/run_search.py --scenario scenarios/windy_mixed.yaml --strategy auction --seed 3 --gif media/search_auction_seed3.gif
```

## Status

| Phase | Name | Status |
| --- | --- | --- |
| 1 | World and drift | done |
| 2 | Probability map | done |
| 3 | Vehicles, safety and baseline search | done |
| 4 | Auction planner and evaluation | in review |
| 5 | Operator agent and MCP server | not started |
| 6 | Polish, publish and stretch | not started |

![Greedy Bayesian search with two drones and a USV](media/search_greedy_seed3.gif)

*Two drones and a USV search the probability map (greedy Bayesian strategy); after a detection
the nearest USV recovers the person. The true position is not shown: the swarm only knows the
map.*

### Results: 50 random scenarios x 4 fleets x 3 strategies

![Fraction of persons found vs time, per fleet and strategy](media/results_found_vs_time.png)

| Fleet | Found within 60 min: lawnmower | greedy | auction | Median time to find (best) |
| --- | --- | --- | --- | --- |
| 2 drones + 1 USV | 46% | 64% | 68% | 34 min (auction) |
| 4 drones | 38% | 60% | 52% | 36 min (greedy) |
| 2 USVs | 34% | 70% | 74% | 37 min (auction) |
| 4 drones + 2 USVs | 54% | 76% | 80% | 30 min (auction) |

Greedy search of the probability map beats the fixed lawnmower baseline in every fleet, and the
bundle auction in three of four (paired over the same scenarios; with 4 drones its +14 points
are not significant); greedy and auction are statistically tied. 600 runs, zero safety
violations. Full table with 95 % bootstrap intervals and paired comparisons:
[results/summary.md](results/summary.md).

![Probability map vs 5000 drift particles after 2 hours](media/belief_vs_particles.png)

*The Bayesian probability map (left), predicted with the exact variance of the drift noise,
against 5000 Monte Carlo particles of the drift model (right) after 2 hours.*

Design notes, equations and parameter sources: [docs/design.md](docs/design.md).

## Licence

[AGPL-3.0](LICENSE).
