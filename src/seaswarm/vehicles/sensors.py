"""Sensor footprints on the belief grid, and the detection roll on the truth side.

Belief side: the likelihood of "no detection" given the person is in cell i needs the POD
averaged over the cell, because the person may be anywhere in it. Evaluating POD only at the cell
centre breaks down when the footprint is smaller than a cell (a drone camera at 60 m sees 42 m,
the cells are 100 m): a drone flying between centres would update nothing. The average is
computed with a midpoint rule whose spacing is tied to the sensor (r_max / points_per_r_max), not
to the cell, because the hard cut-off at r_max is what limits the accuracy.

Truth side: a look detects the person with probability POD(distance to the true position).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from seaswarm.search.grid import Grid
from seaswarm.search.pod import SensorModel


@dataclass(frozen=True)
class Footprint:
    """Cell-averaged POD of one look, stored sparsely: only cells with POD > 0."""

    rows: NDArray[np.intp]
    cols: NDArray[np.intp]
    pod: NDArray[np.float64]


def footprint(
    grid: Grid, sensor: SensorModel, x_m: float, y_m: float, points_per_r_max: int
) -> Footprint:
    """Cell-averaged POD of a look taken from (x_m, y_m).

    Args:
        grid: Belief grid.
        sensor: POD model of the sensor.
        x_m, y_m: Horizontal position of the sensor, ENU metres.
        points_per_r_max: Midpoint-rule points per r_max along each axis.
    """
    h = grid.cell_size_m
    subsamples = int(np.ceil(points_per_r_max * h / sensor.r_max_m))
    reach = sensor.r_max_m
    c0 = max(int(np.floor((x_m - reach - grid.x_min_m) / h)), 0)
    c1 = min(int(np.floor((x_m + reach - grid.x_min_m) / h)), grid.nx - 1)
    r0 = max(int(np.floor((y_m - reach - grid.y_min_m) / h)), 0)
    r1 = min(int(np.floor((y_m + reach - grid.y_min_m) / h)), grid.ny - 1)
    if c0 > c1 or r0 > r1:
        return Footprint(np.empty(0, np.intp), np.empty(0, np.intp), np.empty(0))

    offsets = (np.arange(subsamples) + 0.5) / subsamples  # midpoints in [0, 1)
    xs = grid.x_min_m + h * (np.arange(c0, c1 + 1)[:, None] + offsets).ravel()
    ys = grid.y_min_m + h * (np.arange(r0, r1 + 1)[:, None] + offsets).ravel()
    pod = sensor.pod(np.hypot(xs[None, :] - x_m, ys[:, None] - y_m))
    n_rows, n_cols = r1 - r0 + 1, c1 - c0 + 1
    cell_pod = pod.reshape(n_rows, subsamples, n_cols, subsamples).mean(axis=(1, 3))

    rows, cols = np.nonzero(cell_pod)
    return Footprint(rows + r0, cols + c0, cell_pod[rows, cols])


def roll_detection(
    sensor: SensorModel,
    sensor_xy_m: NDArray[np.float64],
    person_xy_m: NDArray[np.float64],
    rng: np.random.Generator,
) -> bool:
    """Truth-side Bernoulli roll: does this look detect the person? Uses the `sensors` stream."""
    distance_m = float(np.hypot(*(np.asarray(person_xy_m) - np.asarray(sensor_xy_m))))
    return bool(rng.random() < sensor.pod(distance_m))
