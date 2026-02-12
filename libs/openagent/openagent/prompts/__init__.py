"""Prompt infrastructure for building agent system prompts.

Three layers:
- Content: ``load``/``find`` for read-only .md fragment text
- Sections: self-contained functions producing prompt sections
- Compose: ``compose()`` joins sections via profiles
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from openagent.prompts import sections
from openagent.prompts.content import find, load, substitute
from openagent.types import AgentContext, GitContext

SectionFn = Callable[[AgentContext], str | None]
"""A section function: takes context, returns content or None to opt out."""


def compose(profile: Sequence[SectionFn], ctx: AgentContext) -> str:
    """Compose a system prompt from a profile and context.

    Args:
        profile: Ordered list of section functions.
        ctx: Runtime state snapshot.

    Returns:
        The assembled prompt string.
    """
    return "\n\n".join(section_text for fn in profile if (section_text := fn(ctx)) is not None)


FRESH_SESSION: Sequence[SectionFn] = [
    sections.identity,
    sections.doing_tasks,
    sections.executing_actions,
    sections.tone_and_style,
    sections.tool_usage_policy,
    sections.tool_instructions,
    sections.skills,
    sections.mcps,
    sections.environment,
    sections.git_status,
    sections.scratchpad,
]
"""Profile for a new conversation session."""

RESUMED_SESSION: Sequence[SectionFn] = [
    sections.identity,
    sections.doing_tasks,
    sections.executing_actions,
    sections.tone_and_style,
    sections.tool_usage_policy,
    sections.tool_instructions,
    sections.skills,
    sections.mcps,
    sections.environment,
    sections.git_status,
    sections.scratchpad,
]
"""Profile for a session resumed from compaction. Same as FRESH for now."""

__all__ = [
    "FRESH_SESSION",
    "RESUMED_SESSION",
    "AgentContext",
    "GitContext",
    "SectionFn",
    "compose",
    "find",
    "load",
    "sections",
    "substitute",
]
