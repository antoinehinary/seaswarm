"""Adapter between MCP tools and the code the agent writes."""

from __future__ import annotations

import json
from typing import Any, override

from smolagents import Tool


class OperatorTool(Tool):
    """Wrap an MCP tool so the agent's code gets a dict and may pass positional arguments.

    With `structured_output=False`, mcpadapt returns the tool's JSON result as text; parsing it
    here lets the agent write `situation["vehicles"]` instead of string manipulation. Positional
    arguments are mapped to the inputs in order: mcpadapt only accepts keywords, and small models
    often write `recall_vehicle("drone_1")`, then retry the failing call.
    Adapted from skypilot-agent/src/skypilot/agent/tools.py.
    """

    skip_forward_signature_validation = True

    def __init__(self, tool: Tool) -> None:
        """Wrap `tool`, keeping its name, description and input schema."""
        self.name = tool.name
        self.description = tool.description + " Returns a dict."
        self.inputs = tool.inputs
        self.output_type = "object"
        self.output_schema = None
        self.is_initialized = True
        self._tool = tool

    @override
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        names = list(self.inputs)
        if len(args) > len(names):
            raise TypeError(
                f"{self.name}() takes {len(names)} argument(s) ({', '.join(names) or 'none'}) "
                f"but {len(args)} were given"
            )
        for name, value in zip(names, args, strict=False):
            if name in kwargs:
                raise TypeError(f"{self.name}() got multiple values for argument '{name}'")
            kwargs[name] = value
        result = self._tool(**kwargs)
        if isinstance(result, str):
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return result
        return result
