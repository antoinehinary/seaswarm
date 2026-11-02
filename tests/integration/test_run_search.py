import json
import subprocess
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def test_run_search_writes_logs_and_a_replay(tmp_path: Path) -> None:
    gif = tmp_path / "replay.gif"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_search.py"),
            "--scenario",
            str(ROOT / "scenarios" / "windy_mixed.yaml"),
            "--strategy",
            "greedy",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--gif",
            str(gif),
            "--frame-every-s",
            "120",
        ],
        capture_output=True,
        text=True,
        env={"MPLBACKEND": "Agg", "PATH": ""},
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    run_dir = tmp_path / "runs" / "windy_mixed_greedy_3"
    with np.load(run_dir / "log.npz") as log:
        assert log["xy_m"].shape[1:] == (3, 2)
        assert np.allclose(log["belief"].sum(axis=(1, 2)) <= 1.0 + 1e-6, True)
        meta = json.loads(str(log["meta"]))
    assert meta["state"] == "RECOVERED"
    kinds = {
        json.loads(line)["kind"] for line in (run_dir / "events.jsonl").read_text().splitlines()
    }
    assert {"replan", "detection", "safety_decision"} <= kinds
    assert iio.imread(gif, index=None).shape[0] >= 5
