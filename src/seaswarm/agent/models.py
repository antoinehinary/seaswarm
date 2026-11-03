"""LLM clients for the operator agent, and a scripted stand-in for tests."""

from __future__ import annotations

from typing import Any, override

from smolagents import LiteLLMModel, Model
from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import TokenUsage

from seaswarm.config import AgentConfig


def build_model(config: AgentConfig) -> LiteLLMModel:
    """LiteLLM client for `config.model_id` (local Ollama by default, any LiteLLM provider else).

    Cloud providers read their API key from the environment (for example ANTHROPIC_API_KEY);
    nothing here requires one.
    """
    kwargs: dict[str, Any] = {
        "timeout": config.request_timeout_s,
        "temperature": config.temperature,
    }
    if config.seed is not None:
        kwargs["seed"] = config.seed
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    if config.model_id.startswith("ollama"):
        kwargs |= {"num_ctx": config.ollama.num_ctx, "think": config.ollama.think}
        return LiteLLMModel(model_id=config.model_id, api_base=config.api_base, **kwargs)
    return LiteLLMModel(model_id=config.model_id, **kwargs)


def code_reply(source: str) -> str:
    """A model reply with one Python code block, in the format CodeAgent parses."""
    return f"Thought: next step.\n```python\n{source}\n```"


class ScriptedModel(Model):
    """Replays pre-written replies in order and records every prompt (tests, no LLM).

    Adapted from skypilot-agent/src/skypilot/eval/scripted.py.
    """

    def __init__(self, replies: list[str]) -> None:
        """Replies to return, one per agent step."""
        super().__init__(model_id="scripted")
        self.replies = list(replies)
        self.prompts: list[str] = []

    @override
    def generate(self, messages: list, stop_sequences: Any = None, **kwargs: Any) -> ChatMessage:
        self.prompts.append(str(messages))
        reply = self.replies.pop(0) if self.replies else "Thought: out of script.\n"
        return ChatMessage(role=MessageRole.ASSISTANT, content=reply, token_usage=TokenUsage(10, 5))

    @classmethod
    def from_code(cls, steps: list[str]) -> ScriptedModel:
        """One code block per agent step."""
        return cls([code_reply(step) for step in steps])
