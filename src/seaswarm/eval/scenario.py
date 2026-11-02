"""Random scenario generator (the schema itself is `seaswarm.config.Scenario`).

Ranges follow the brief: wind 2-15 m/s, current 0-0.8 m/s, random directions, night 30 % of the
time. The last known position is placed upstream of the expected drift, so that the predicted
position half-way through the mission lies near the centre of the search area: with a 0.8 m/s
current a person drifts ~9 km in 3 h, which would otherwise leave a 10 km area.
"""

from __future__ import annotations

import numpy as np

from seaswarm.config import DefaultConfig, Scenario
from seaswarm.units import from_vector, towards_vector

WIND_MPS = (2.0, 15.0)
CURRENT_MPS = (0.0, 0.8)
NIGHT_PROBABILITY = 0.3
LKP_SIGMA_M = (150.0, 400.0)
TIME_SINCE_FALL_S = (300.0, 1800.0)
MID_MISSION_S = 5400.0  # aim the expected drift at the area centre at 1.5 h
MID_OFFSET_M = 1500.0  # random offset of that aim point
EDGE_MARGIN_M = 1000.0  # keep the LKP this far inside the area

TEMPLATE = {
    "area": {"width_m": 10000.0, "height_m": 10000.0, "cell_size_m": 100.0},
    "base": {"x_m": 0.0, "y_m": -6000.0},
    "drift": {"object": "PIW-1", "tau_s": 1800.0},
    "fleet": {"drones_eo": 2, "drones_thermal": 0, "usvs": 1},
    "limits": {"max_sim_time_s": 10800.0, "max_steps": 20000},
}


def generate_scenario(index: int, master_seed: int, defaults: DefaultConfig) -> Scenario:
    """Scenario `gen_<index>`: drawn from its own child of `SeedSequence(master_seed)`.

    The scenario's own seed (truth and sensor randomness at run time) is `index`, so seed N of an
    evaluation is scenario gen_N.
    """
    rng = np.random.default_rng(np.random.SeedSequence(master_seed).spawn(index + 1)[index])
    wind_speed = rng.uniform(*WIND_MPS)
    wind_from = rng.uniform(0.0, 360.0)
    current_speed = rng.uniform(*CURRENT_MPS)
    current_to = rng.uniform(0.0, 360.0)

    leeway = defaults.leeway_objects["PIW-1"].downwind_slope * wind_speed
    drift = np.array(towards_vector(current_speed, current_to)) + leeway * np.array(
        from_vector(1.0, wind_from)
    )
    aim = rng.uniform(-MID_OFFSET_M, MID_OFFSET_M, 2)
    half = TEMPLATE["area"]["width_m"] / 2 - EDGE_MARGIN_M
    lkp = np.clip(aim - drift * MID_MISSION_S, -half, half)

    raw = {
        "id": f"gen_{index:03d}",
        "seed": index,
        **TEMPLATE,
        "lkp": {
            "x_m": round(float(lkp[0]), 1),
            "y_m": round(float(lkp[1]), 1),
            "sigma_m": round(float(rng.uniform(*LKP_SIGMA_M)), 1),
            "time_since_fall_s": round(float(rng.uniform(*TIME_SINCE_FALL_S))),
        },
        "wind": {
            "type": "uniform",
            "speed_mps": round(float(wind_speed), 2),
            "from_bearing_deg": round(float(wind_from), 1),
        },
        "current": {
            "type": "uniform",
            "speed_mps": round(float(current_speed), 3),
            "to_bearing_deg": round(float(current_to), 1),
        },
        "daylight": bool(rng.random() >= NIGHT_PROBABILITY),
    }
    return Scenario.model_validate(raw)
