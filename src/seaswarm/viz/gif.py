"""Headless frame capture and GIF export.

Figures are built with `matplotlib.figure.Figure` on an Agg canvas rather than through pyplot, so
rendering never depends on a GUI backend or on pyplot's global state.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


def new_figure(width_in: float, height_in: float, dpi: int) -> Figure:
    """Create a figure attached to an Agg canvas."""
    fig = Figure(figsize=(width_in, height_in), dpi=dpi)
    FigureCanvasAgg(fig)
    return fig


def figure_to_rgb(fig: Figure) -> Frame:
    """Render a figure and return its pixels as an (H, W, 3) uint8 array."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def write_gif(frames: Sequence[Frame], path: Path, fps: float) -> None:
    """Write frames to a looping GIF at `fps` frames per second."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Pillow takes the frame duration in milliseconds (GIF timing has 10 ms resolution)
    iio.imwrite(path, np.stack(frames), extension=".gif", duration=1000.0 / fps, loop=0)
