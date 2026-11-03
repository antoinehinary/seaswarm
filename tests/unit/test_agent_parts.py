import pytest
from smolagents import Tool

from seaswarm.agent.models import ScriptedModel, build_model
from seaswarm.agent.tools import OperatorTool
from seaswarm.config import load_agent_config


class _Echo(Tool):
    name = "assign_vehicle"
    description = "Echo the arguments as JSON."
    inputs = {
        "vehicle_id": {"type": "string", "description": "id"},
        "x_m": {"type": "number", "description": "x"},
    }
    output_type = "string"

    def forward(self, vehicle_id: str, x_m: float) -> str:
        return f'{{"ok": true, "vehicle_id": "{vehicle_id}", "x_m": {x_m}}}'


def test_operator_tool_parses_json_and_maps_positional_arguments() -> None:
    tool = OperatorTool(_Echo())
    assert tool("drone_1", 5.0) == {"ok": True, "vehicle_id": "drone_1", "x_m": 5.0}
    assert tool(vehicle_id="usv_1", x_m=1.0)["vehicle_id"] == "usv_1"
    with pytest.raises(TypeError, match="multiple values"):
        tool("drone_1", vehicle_id="drone_2", x_m=0.0)
    with pytest.raises(TypeError, match="takes 2 argument"):
        tool("a", 1.0, 2.0)


def test_agent_config_defaults_to_local_ollama_with_a_large_context() -> None:
    config = load_agent_config()
    assert config.model_id == "ollama_chat/qwen3:8b"
    assert config.ollama.num_ctx == 32768 and config.ollama.think is False


def test_environment_overrides_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEASWARM_MODEL_ID", "anthropic/claude-sonnet-5-5")
    config = load_agent_config()
    assert config.model_id == "anthropic/claude-sonnet-5-5"
    model = build_model(config)
    assert model.model_id == "anthropic/claude-sonnet-5-5"


def test_scripted_model_replays_in_order() -> None:
    model = ScriptedModel.from_code(["print(1)"])
    assert "print(1)" in model.generate([]).content
    assert "out of script" in model.generate([]).content
