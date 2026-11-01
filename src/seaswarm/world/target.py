"""Truth state of the person in water.

Only the simulator and the sensor model may import this module: the planner and the probability
map work from beliefs, never from the true position (enforced by tests/unit/test_layering.py,
Phase 3).

Time origin: t = 0 is the moment of the fall. The search starts at t = lkp.time_since_fall_s.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from seaswarm.config import DefaultConfig, Scenario
from seaswarm.rng import ScenarioRng
from seaswarm.world.drift import ParticleDrift, sample_around_lkp


class PersonInWater:
    """The one true drifting person: a single-particle `ParticleDrift`."""

    def __init__(self, drift: ParticleDrift) -> None:
        """Wrap a drift model holding exactly one particle."""
        if len(drift.positions_m) != 1:
            raise ValueError(f"a person is one particle, got {len(drift.positions_m)}")
        self._drift = drift

    @classmethod
    def from_scenario(
        cls, scenario: Scenario, defaults: DefaultConfig, rngs: ScenarioRng
    ) -> PersonInWater:
        """Place the person at a fall position drawn around the LKP, at t = 0.

        The fall position uses the `initial_position` stream and the drift noise the `drift`
        stream, so changing one never changes the other.
        """
        fall_m = sample_around_lkp(scenario.lkp, 1, rngs.initial_position)
        return cls(ParticleDrift.from_scenario(scenario, defaults, fall_m, rngs.drift))

    @property
    def position_m(self) -> NDArray[np.float64]:
        """True ENU position (2,), metres."""
        return self._drift.positions_m[0]

    @property
    def t_s(self) -> float:
        """Simulation time since the fall, s."""
        return self._drift.t_s

    def step(self, dt_s: float) -> None:
        """Advance the person by one step of dt_s seconds."""
        self._drift.step(dt_s)
