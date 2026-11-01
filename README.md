# SeaSwarm

A heterogeneous swarm of drones and autonomous surface vessels searches for a person in water.
SeaSwarm predicts the drift, keeps a Bayesian probability map of where the person may be, splits
the search between vehicles and replans after every sweep, while a human operator directs it in
plain language through an LLM agent over MCP.

Second project of a series on autonomy for drones, maritime and space systems, after
[skypilot-agent](https://github.com/antoinehinary/skypilot-agent).

> Work in progress. Not an operational search-and-rescue tool: all drift and detection
> parameters are illustrative values from public literature.

## Status

| Phase | Name | Status |
| --- | --- | --- |
| 1 | World and drift | in progress |
| 2 | Probability map | not started |
| 3 | Vehicles, safety and baseline search | not started |
| 4 | Auction planner and evaluation | not started |
| 5 | Operator agent and MCP server | not started |
| 6 | Polish, publish and stretch | not started |

## Licence

[AGPL-3.0](LICENSE).
