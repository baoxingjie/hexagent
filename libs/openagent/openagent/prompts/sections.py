"""Self-contained section functions for system prompt assembly.

Each section function has the signature ``(PromptContext) -> str | None``.
It loads its own content, resolves its own variables, and decides its own
inclusion (returning ``None`` to opt out).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from openagent.prompts.content import find, load

if TYPE_CHECKING:
    from openagent.tools.base import BaseAgentTool
    from openagent.types import MCPServer, Skill


# ---------------------------------------------------------------------------
# Context types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GitContext:
    """Git repository snapshot for prompt assembly."""

    current_branch: str
    main_branch: str
    status: str
    recent_commits: str


@dataclass(frozen=True)
class PromptContext:
    """Runtime state snapshot for prompt assembly.

    Created at session boundaries (new conversation or resumed session).
    Frozen — represents a point-in-time snapshot, not live state.
    """

    tools: list[BaseAgentTool[Any]] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)
    mcps: list[MCPServer] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    user_instructions: str | None = None
    git: GitContext | None = None
    scratchpad_dir: str | None = None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _substitute(template: str, **variables: str) -> str:
    r"""Replace ${key} placeholders with values.

    Unlike ``string.Template``, this does NOT interpret ``$`` as a
    special character.  Only explicit ``${key}`` patterns matching
    a provided keyword argument are replaced.  All other ``$``
    characters are left as-is (no escaping needed in .md files).
    """
    result = template
    for key, value in variables.items():
        result = result.replace(f"${{{key}}}", value)
    return result


def _tool_vars(ctx: PromptContext) -> dict[str, str]:
    """Build tool name cross-reference variables from context.

    Returns a dict like ``{"BASH_TOOL_NAME": "bash", "READ_TOOL_NAME": "read", ...}``
    for all tools in ``ctx.tools``.  Used by section functions that load
    ``.md`` fragments containing ``${*_TOOL_NAME}`` placeholders.
    """
    return {f"{t.name.upper()}_TOOL_NAME": t.name for t in ctx.tools}


# ---------------------------------------------------------------------------
# Section functions — each is (PromptContext) -> str | None
# ---------------------------------------------------------------------------


def identity(_ctx: PromptContext) -> str | None:
    """Agent identity and help links. Always included."""
    return load("system_prompt_identity")


def doing_tasks(_ctx: PromptContext) -> str | None:
    """Software engineering task guidance. Always included."""
    return load("system_prompt_doing_tasks")


def executing_actions(_ctx: PromptContext) -> str | None:
    """Reversibility and blast-radius policy. Always included."""
    return load("system_prompt_executing_actions_with_care")


def tone_and_style(ctx: PromptContext) -> str | None:
    """Output style, objectivity, no time estimates. Always included."""
    return _substitute(load("system_prompt_tone_and_style"), **_tool_vars(ctx))


def tool_usage_policy(ctx: PromptContext) -> str | None:
    """Parallel calling and tool-over-bash preference."""
    if not ctx.tools:
        return None
    return _substitute(load("system_prompt_tool_usage_policy"), **_tool_vars(ctx))


def tool_instructions(ctx: PromptContext) -> str | None:
    """Per-tool usage instructions with supplementary fragments."""
    if not ctx.tools:
        return None

    today = date.today()  # noqa: DTZ011
    shared_vars = {
        **_tool_vars(ctx),
        "CURRENT_DATE": today.isoformat(),
        "CURRENT_YEAR": str(today.year),
        "PREVIOUS_YEAR": str(today.year - 1),
        "GIT_PARALLEL_NOTE": (
            "You can call multiple tools in a single response. "
            "When multiple independent pieces of information are requested "
            "and all commands are likely to succeed, "
            "run multiple tool calls in parallel for optimal performance."
        ),
    }

    tool_sections: list[str] = []

    for tool in ctx.tools:
        main_key = f"tool_instruction_{tool.name}"

        # Try .md file first, fall back to tool.instruction
        try:
            raw = load(main_key)
        except KeyError:
            if tool.instruction:
                tool_sections.append(f"## {tool.name}\n\n{tool.instruction}")
                continue
            msg = (
                f"Missing tool instruction: no '{main_key}.md' file "
                f"and tool '{tool.name}' has no instruction text. "
                f"Create '{main_key}.md' in prompts/ or set "
                f"tool.instruction on the tool class."
            )
            raise KeyError(msg)  # noqa: B904

        parts: list[str] = [raw]

        # Discover and append supplementary fragments
        prefix = f"tool_instruction_{tool.name}_"
        parts.extend(load(supp_key) for supp_key in find(prefix))

        content = "\n\n".join(parts)

        # Apply all shared substitutions (tool names, dates,
        # git_parallel_note). str.replace is a safe no-op for
        # vars not present in a given fragment.
        content = _substitute(content, **shared_vars)

        tool_sections.append(f"## {tool.name}\n\n{content}")

    return "# Tools\n\n" + "\n\n---\n\n".join(tool_sections)


def skills(ctx: PromptContext) -> str | None:
    """Skill listing."""
    if not ctx.skills:
        return None
    items = "\n".join(f"- **{s.name}**: {s.description}" for s in ctx.skills)
    return f"# Skills\n\nThe following skills extend the agent with specialized workflows. Use the skill tool to invoke them by name.\n\n{items}"


def mcps(ctx: PromptContext) -> str | None:
    """MCP server listing."""
    if not ctx.mcps:
        return None
    items = "\n".join(f"- **{m.name}**: {m.description}" for m in ctx.mcps)
    return f"# MCP Servers\n\nThe following MCP (Model Context Protocol) servers provide additional capabilities and tools.\n\n{items}"


def environment(ctx: PromptContext) -> str | None:
    """Environment key-value pairs."""
    if not ctx.environment:
        return None
    items = "\n".join(f"- {k}: {v}" for k, v in ctx.environment.items())
    return f"# Environment\n\nThe following context describes your operating environment.\n\n{items}"


def git_status(ctx: PromptContext) -> str | None:
    """Git repository snapshot."""
    if ctx.git is None:
        return None
    return _substitute(
        load("system_prompt_git_status"),
        CURRENT_BRANCH=ctx.git.current_branch,
        MAIN_BRANCH=ctx.git.main_branch,
        GIT_STATUS=ctx.git.status or "(clean)",
        RECENT_COMMITS=ctx.git.recent_commits,
    )


def scratchpad(ctx: PromptContext) -> str | None:
    """Scratchpad directory path."""
    if ctx.scratchpad_dir is None:
        return None
    return _substitute(
        load("system_prompt_scratchpad_directory"),
        SCRATCHPAD_DIR=ctx.scratchpad_dir,
    )


def user_instructions(ctx: PromptContext) -> str | None:
    """User-provided instructions (raw text under heading)."""
    if ctx.user_instructions is None:
        return None
    return f"# User Instructions\n\n{ctx.user_instructions}"
