from pathlib import Path

import yaml

from seaswarm.config import load_default_config
from seaswarm.eval.run import main
from seaswarm.eval.scenario import generate_scenario


def _scenarios(tmp_path: Path) -> Path:
    out = tmp_path / "scenarios"
    out.mkdir()
    defaults = load_default_config()
    for i in range(2):
        raw = generate_scenario(i, 0, defaults).model_dump(mode="json")
        raw["limits"]["max_sim_time_s"] = 1800.0  # keep the test short
        (out / f"gen_{i:03d}.yaml").write_text(yaml.safe_dump(raw))
    return out


def test_results_are_identical_for_any_number_of_workers(tmp_path: Path) -> None:
    scenarios = _scenarios(tmp_path)
    csvs = []
    for workers in (1, 2):
        out = tmp_path / f"results_{workers}"
        args = ["--strategy", "greedy,auction", "--fleet", "2D+1U", "--seeds", "0-1"]
        args += ["--workers", str(workers), "--scenarios-dir", str(scenarios), "--out", str(out)]
        assert main([*args, "--no-report"]) == 0
        csvs.append((out / "results.csv").read_bytes())
    assert csvs[0] == csvs[1]
    lines = csvs[0].decode().splitlines()
    assert len(lines) == 1 + 4
    assert lines[1].startswith("auction,2D+1U,0,gen_000")  # sorted by strategy, fleet, seed
