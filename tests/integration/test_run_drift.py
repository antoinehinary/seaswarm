import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio

ROOT = Path(__file__).resolve().parents[2]


def test_run_drift_writes_a_gif(tmp_path: Path) -> None:
    gif = tmp_path / "drift.gif"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_drift.py"),
            "--scenario",
            str(ROOT / "scenarios" / "windy_mixed.yaml"),
            "--particles",
            "50",
            "--duration-s",
            "600",
            "--frame-every-s",
            "120",
            "--gif",
            str(gif),
            "--show-truth",
        ],
        capture_output=True,
        text=True,
        env={"MPLBACKEND": "Agg", "PATH": ""},
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "centroid moved" in result.stdout
    assert iio.imread(gif, index=None).shape[0] == 6  # t = 0 plus 5 frames
