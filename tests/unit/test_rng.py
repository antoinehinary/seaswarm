from dataclasses import fields

import numpy as np

from seaswarm.rng import ScenarioRng, make_rngs

STREAMS = [f.name for f in fields(ScenarioRng)]


def _draws(rngs: ScenarioRng) -> dict[str, np.ndarray]:
    return {name: getattr(rngs, name).standard_normal(100) for name in STREAMS}


def test_same_seed_gives_identical_streams() -> None:
    first, second = _draws(make_rngs(7)), _draws(make_rngs(7))
    for name in STREAMS:
        np.testing.assert_array_equal(first[name], second[name])


def test_different_seed_gives_different_streams() -> None:
    first, second = _draws(make_rngs(7)), _draws(make_rngs(8))
    for name in STREAMS:
        assert not np.array_equal(first[name], second[name])


def test_streams_differ_from_each_other() -> None:
    draws = _draws(make_rngs(7))
    for i, a in enumerate(STREAMS):
        for b in STREAMS[i + 1 :]:
            assert not np.array_equal(draws[a], draws[b])


def test_drawing_from_one_stream_does_not_shift_another() -> None:
    untouched = make_rngs(7).drift.standard_normal(10)
    rngs = make_rngs(7)
    rngs.sensors.standard_normal(1000)
    np.testing.assert_array_equal(rngs.drift.standard_normal(10), untouched)
