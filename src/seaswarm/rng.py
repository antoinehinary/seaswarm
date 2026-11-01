"""Named, independent random generators derived from one scenario seed.

Each consumer of randomness gets its own stream, so adding draws in one place (say, a new sensor
roll) never shifts the numbers another place sees (say, the drift noise). Same seed => the same
numbers in every stream => bit-identical runs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass(frozen=True, slots=True)
class ScenarioRng:
    """One generator per consumer, all spawned from the same scenario seed.

    Streams are spawned in field order. Only ever append new fields at the end: inserting one
    in the middle would change the seed of every stream after it.
    """

    drift: np.random.Generator
    sensors: np.random.Generator
    initial_position: np.random.Generator
    planner_tiebreak: np.random.Generator


def make_rngs(seed: int) -> ScenarioRng:
    """Spawn the named generators of a scenario from its integer seed."""
    children = np.random.SeedSequence(seed).spawn(len(fields(ScenarioRng)))
    return ScenarioRng(*(np.random.default_rng(child) for child in children))
