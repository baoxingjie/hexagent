"""System prompt assembler for building agent system prompts.

This module provides the SystemPromptAssembler class which builds system
prompts from typed data sources in a defined order.

Design follows Claude Code's pattern: ORDER is explicit data, not buried
in code logic. To reorder sections, just change the ORDER list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from openagent.tools.base import BaseAgentTool
    from openagent.types import MCPServer, Skill


class SystemPromptAssembler:
    """Assembles system prompt from sources in defined order.

    The assembler takes typed data (tools, skills, etc.) and formats each
    section according to its type. The order of sections is defined by the
    ORDER class attribute - visible, obvious, easy to reorder.

    Examples:
        ```python
        assembler = SystemPromptAssembler()

        prompt = assembler.assemble(
            base="You are a helpful assistant.",
            tools=registry.get_tools(),
            skills=registry.get_skills(),
            environment={"platform": "darwin", "cwd": "/project"},
            user_instructions="Focus on Python code.",
        )
        ```

        To reorder sections, subclass and change ORDER::

            class CustomAssembler(SystemPromptAssembler):
                ORDER = ["base", "environment", "tools", "skills", "mcps", "user_instructions"]
    """

    # Order is explicit data - visible, obvious, easy to reorder
    ORDER: ClassVar[list[str]] = ["base", "tools", "skills", "mcps", "environment", "user_instructions"]

    def assemble(
        self,
        *,
        base: str,
        tools: list[BaseAgentTool[Any]] | None = None,
        skills: list[Skill] | None = None,
        mcps: list[MCPServer] | None = None,
        environment: dict[str, str] | None = None,
        user_instructions: str | None = None,
    ) -> str:
        """Assemble prompt from sources in ORDER sequence.

        Args:
            base: The base agent persona/instructions.
            tools: List of tools to include.
            skills: List of skills to include.
            mcps: List of MCP servers to include.
            environment: Environment context as key-value pairs.
            user_instructions: Additional user instructions.

        Returns:
            The assembled system prompt.
        """
        # Map section names to (data, formatter)
        sections: dict[str, tuple[Any, Callable[[Any], str]]] = {
            "base": (base, self._format_base),
            "tools": (tools, self._format_tools),
            "skills": (skills, self._format_skills),
            "mcps": (mcps, self._format_mcps),
            "environment": (environment, self._format_environment),
            "user_instructions": (user_instructions, self._format_user_instructions),
        }

        result = []
        for name in self.ORDER:
            data, formatter = sections[name]
            if data:
                result.append(formatter(data))

        return "\n\n".join(result)

    def _format_base(self, base: str) -> str:
        """Format the base prompt section."""
        return base

    def _format_tools(self, tools: list[BaseAgentTool[Any]]) -> str:
        """Format the tools section."""
        lines = ["## Tools", ""]
        lines.extend(f"- **{t.name}**: {t.description}" for t in tools)
        return "\n".join(lines)

    def _format_skills(self, skills: list[Skill]) -> str:
        """Format the skills section."""
        lines = ["## Skills", ""]
        lines.extend(f"- **{s.name}**: {s.description}" for s in skills)
        return "\n".join(lines)

    def _format_mcps(self, mcps: list[MCPServer]) -> str:
        """Format the MCP servers section."""
        lines = ["## MCP Servers", ""]
        lines.extend(f"- **{m.name}**: {m.description}" for m in mcps)
        return "\n".join(lines)

    def _format_environment(self, env: dict[str, str]) -> str:
        """Format the environment context section."""
        lines = ["## Environment", ""]
        lines.extend(f"- {k}: {v}" for k, v in env.items())
        return "\n".join(lines)

    def _format_user_instructions(self, instructions: str) -> str:
        """Format the user instructions section."""
        return f"## User Instructions\n\n{instructions}"
