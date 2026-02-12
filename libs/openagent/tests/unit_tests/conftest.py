"""Shared test fixtures for unit tests."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from openagent.tools.base import BaseAgentTool
from openagent.types import ToolResult


class StubParams(BaseModel):
    """Minimal parameter schema for mock tools."""

    arg: str = ""


def make_tool(name: str, *, instruction: str = "") -> BaseAgentTool[StubParams]:
    """Create a stub tool with the given name.

    Use this instead of writing per-tool mock classes.
    """

    class _Tool(BaseAgentTool[StubParams]):
        args_schema = StubParams

        async def execute(self, params: StubParams) -> ToolResult:
            return ToolResult(output="")

    _Tool.name = name
    _Tool.instruction = instruction
    return _Tool()


def core_tools() -> list[BaseAgentTool[Any]]:
    """Return the six core mock tools (Bash, Read, Edit, Write, Glob, Grep)."""
    return [make_tool(n) for n in ("Bash", "Read", "Edit", "Write", "Glob", "Grep")]
