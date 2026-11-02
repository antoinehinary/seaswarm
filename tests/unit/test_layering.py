"""Architecture rules, checked statically on the import graph (ast, no imports executed)."""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE = "seaswarm"
TRUTH = "seaswarm.world.target"


def _module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return {name for name in found if name.startswith(PACKAGE)}


GRAPH = {_module_name(p): _imports(p) for p in (SRC / PACKAGE).rglob("*.py")}


def _reachable(start: str) -> set[str]:
    seen, stack = set(), [start]
    while stack:
        module = stack.pop()
        for dep in GRAPH.get(module, ()):
            if dep in GRAPH and dep not in seen:
                seen.add(dep)
                stack.append(dep)
    return seen


@pytest.mark.parametrize(
    "module", sorted(m for m in GRAPH if m.startswith(("seaswarm.planning", "seaswarm.search")))
)
def test_planner_and_belief_never_reach_the_truth(module: str) -> None:
    assert TRUTH not in _reachable(module) | {module}


def test_the_graph_is_not_empty() -> None:
    # guards against a silently vacuous test (wrong paths)
    assert "seaswarm.search.bayes" in GRAPH and TRUTH in GRAPH
    assert TRUTH in _reachable("seaswarm.vehicles.sim2d")
