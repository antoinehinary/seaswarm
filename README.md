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
python scripts/run_drift.py --scenario scenarios/calm_single.yaml --seed 1 --gif media/drift_seed1.gif
```

## Status

| Phase | Name | Status |
| --- | --- | --- |
| 1 | World and drift | done |
| 2 | Probability map | in review |
| 3 | Vehicles, safety and baseline search | not started |
| 4 | Auction planner and evaluation | not started |
| 5 | Operator agent and MCP server | not started |
| 6 | Polish, publish and stretch | not started |

![Probability map vs 5000 drift particles after 2 hours](media/belief_vs_particles.png)

*The Bayesian probability map (left), predicted with the exact variance of the drift noise,
against 5000 Monte Carlo particles of the drift model (right) after 2 hours.*

Design notes, equations and parameter sources: [docs/design.md](docs/design.md).

## Licence

[AGPL-3.0](LICENSE).
