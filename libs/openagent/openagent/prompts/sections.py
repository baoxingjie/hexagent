"""Self-contained section functions for system prompt assembly.

Each section function has the signature ``(AgentContext) -> str | None``.
It loads its own content, resolves its own variables, and decides its own
inclusion (returning ``None`` to opt out).
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from openagent.prompts.content import find, load, substitute

if TYPE_CHECKING:
    from openagent.types import AgentContext


# ---------------------------------------------------------------------------
# Section functions — each is (AgentContext) -> str | None
# ---------------------------------------------------------------------------


def identity(_ctx: AgentContext) -> str | None:
    """Agent identity and help links. Always included."""
    return load("system_prompt_identity")


def doing_tasks(_ctx: AgentContext) -> str | None:
    """Software engineering task guidance. Always included."""
    return load("system_prompt_doing_tasks")


def executing_actions(_ctx: AgentContext) -> str | None:
    """Reversibility and blast-radius policy. Always included."""
    return load("system_prompt_executing_actions_with_care")


def tone_and_style(ctx: AgentContext) -> str | None:
    """Output style, objectivity, no time estimates. Always included."""
    return substitute(load("system_prompt_tone_and_style"), **ctx.tool_name_vars)


def tool_usage_policy(ctx: AgentContext) -> str | None:
    """Parallel calling and tool-over-bash preference."""
    if not ctx.tools:
        return None
    return substitute(load("system_prompt_tool_usage_policy"), **ctx.tool_name_vars)


def tool_instructions(ctx: AgentContext) -> str | None:
    """Per-tool usage instructions with supplementary fragments."""
    if not ctx.tools:
        return None

    today = date.today()  # noqa: DTZ011
    shared_vars = {
        **ctx.tool_name_vars,
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
        content = substitute(content, **shared_vars)

        tool_sections.append(f"## {tool.name}\n\n{content}")

    return "# Tools\n\n" + "\n\n---\n\n".join(tool_sections)


def skills(ctx: AgentContext) -> str | None:
    """Skill listing."""
    if not ctx.skills:
        return None
    items = "\n".join(f"- **{s.name}**: {s.description}" for s in ctx.skills)
    return f"# Skills\n\nThe following skills extend the agent with specialized workflows. Use the skill tool to invoke them by name.\n\n{items}"


def mcps(ctx: AgentContext) -> str | None:
    """MCP server listing."""
    if not ctx.mcps:
        return None
    items = "\n".join(f"- **{m.name}**: {m.description}" for m in ctx.mcps)
    return f"# MCP Servers\n\nThe following MCP (Model Context Protocol) servers provide additional capabilities and tools.\n\n{items}"


def environment(ctx: AgentContext) -> str | None:
    """Environment key-value pairs."""
    if not ctx.environment:
        return None
    items = "\n".join(f"- {k}: {v}" for k, v in ctx.environment.items())
    return f"# Environment\n\nThe following context describes your operating environment.\n\n{items}"


def git_status(ctx: AgentContext) -> str | None:
    """Git repository snapshot."""
    if ctx.git is None:
        return None
    return substitute(
        load("system_prompt_git_status"),
        CURRENT_BRANCH=ctx.git.current_branch,
        MAIN_BRANCH=ctx.git.main_branch,
        GIT_STATUS=ctx.git.status or "(clean)",
        RECENT_COMMITS=ctx.git.recent_commits,
    )


def scratchpad(ctx: AgentContext) -> str | None:
    """Scratchpad directory path."""
    if ctx.scratchpad_dir is None:
        return None
    return substitute(
        load("system_prompt_scratchpad_directory"),
        SCRATCHPAD_DIR=ctx.scratchpad_dir,
    )


def user_instructions(ctx: AgentContext) -> str | None:
    """User-provided instructions (raw text under heading)."""
    if ctx.user_instructions is None:
        return None
    return f"# User Instructions\n\n{ctx.user_instructions}"
