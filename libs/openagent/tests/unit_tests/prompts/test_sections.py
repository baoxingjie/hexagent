"""Tests for prompt sections, compose, and profiles."""

# ruff: noqa: S108, PLR2004

from datetime import date

import pytest

from openagent.prompts import FRESH_SESSION, RESUMED_SESSION, compose
from openagent.prompts.sections import (
    doing_tasks,
    environment,
    executing_actions,
    git_status,
    identity,
    mcps,
    scratchpad,
    skills,
    tone_and_style,
    tool_instructions,
    tool_usage_policy,
    user_instructions,
)
from openagent.types import AgentContext, GitContext, MCPServer, Skill

from ..conftest import core_tools, make_tool

# ---------------------------------------------------------------------------
# Section functions
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_includes_agent_name(self) -> None:
        result = identity(AgentContext())
        assert result is not None
        assert "OpenAgent" in result


class TestDoingTasks:
    def test_includes_heading(self) -> None:
        result = doing_tasks(AgentContext())
        assert result is not None
        assert "Doing tasks" in result


class TestExecutingActions:
    def test_includes_heading(self) -> None:
        result = executing_actions(AgentContext())
        assert result is not None
        assert "Executing actions" in result


class TestToneAndStyle:
    def test_includes_heading(self) -> None:
        result = tone_and_style(AgentContext(tools=[make_tool("bash")]))
        assert result is not None
        assert "Tone and style" in result

    def test_resolves_tool_name_placeholders(self) -> None:
        ctx = AgentContext(tools=[make_tool("bash")])
        result = tone_and_style(ctx)
        assert result is not None
        assert "${BASH_TOOL_NAME}" not in result


class TestToolUsagePolicy:
    def test_returns_none_without_tools(self) -> None:
        assert tool_usage_policy(AgentContext()) is None

    def test_includes_heading_with_tools(self) -> None:
        ctx = AgentContext(tools=core_tools())
        result = tool_usage_policy(ctx)
        assert result is not None
        assert "Tool usage policy" in result

    def test_resolves_tool_name_placeholders(self) -> None:
        ctx = AgentContext(tools=core_tools())
        result = tool_usage_policy(ctx)
        assert result is not None
        for name in ("read", "edit", "write"):
            assert name in result
        assert "${READ_TOOL_NAME}" not in result


class TestToolInstructions:
    def test_returns_none_without_tools(self) -> None:
        assert tool_instructions(AgentContext()) is None

    def test_renders_tool_heading_per_tool(self) -> None:
        ctx = AgentContext(tools=core_tools())
        result = tool_instructions(ctx)
        assert result is not None
        assert "# Tools" in result
        assert "## bash" in result
        assert "## read" in result
        assert "---" in result

    def test_includes_supplementary_fragments(self) -> None:
        ctx = AgentContext(tools=core_tools())
        result = tool_instructions(ctx)
        assert result is not None
        assert "Committing changes with git" in result

    def test_resolves_all_tool_name_cross_references(self) -> None:
        ctx = AgentContext(tools=core_tools())
        result = tool_instructions(ctx)
        assert result is not None
        for name in ("GLOB", "GREP", "READ", "EDIT", "WRITE"):
            assert f"${{{name}_TOOL_NAME}}" not in result

    def test_resolves_git_parallel_note(self) -> None:
        ctx = AgentContext(tools=core_tools())
        result = tool_instructions(ctx)
        assert result is not None
        assert "${GIT_PARALLEL_NOTE}" not in result
        assert "run multiple tool calls in parallel" in result

    def test_raises_for_tool_without_fragment_or_instruction(self) -> None:
        ctx = AgentContext(tools=[make_tool("nonexistent_tool_xyz")])
        with pytest.raises(KeyError, match="Missing tool instruction"):
            tool_instructions(ctx)

    def test_falls_back_to_inline_instruction(self) -> None:
        ctx = AgentContext(tools=[make_tool("custom", instruction="Custom inline docs.")])
        result = tool_instructions(ctx)
        assert result is not None
        assert "## custom" in result
        assert "Custom inline docs." in result

    def test_resolves_date_placeholders(self) -> None:
        ctx = AgentContext(tools=[make_tool("web_search")])
        result = tool_instructions(ctx)
        assert result is not None
        assert "${CURRENT_DATE}" not in result
        today = date.today()  # noqa: DTZ011
        assert today.isoformat() in result


class TestSkills:
    def test_returns_none_when_empty(self) -> None:
        assert skills(AgentContext()) is None

    def test_formats_skill_entry(self) -> None:
        ctx = AgentContext(skills=[Skill(name="commit", description="Git commits", path="/skills/commit")])
        result = skills(ctx)
        assert result is not None
        assert "# Skills" in result
        assert "**commit**" in result
        assert "Git commits" in result


class TestMcps:
    def test_returns_none_when_empty(self) -> None:
        assert mcps(AgentContext()) is None

    def test_formats_mcp_entry(self) -> None:
        ctx = AgentContext(mcps=[MCPServer(name="github", description="GitHub API")])
        result = mcps(ctx)
        assert result is not None
        assert "# MCP Servers" in result
        assert "**github**" in result


class TestEnvironment:
    def test_returns_none_when_empty(self) -> None:
        assert environment(AgentContext()) is None

    def test_formats_key_value_pairs(self) -> None:
        ctx = AgentContext(environment={"OS": "Linux"})
        result = environment(ctx)
        assert result is not None
        assert "# Environment" in result
        assert "OS: Linux" in result


class TestGitStatus:
    def test_returns_none_without_git(self) -> None:
        assert git_status(AgentContext()) is None

    def test_renders_branch_and_status(self) -> None:
        git = GitContext(
            current_branch="feat/test",
            main_branch="main",
            status="M file.py",
            recent_commits="abc123 Initial commit",
        )
        result = git_status(AgentContext(git=git))
        assert result is not None
        assert "feat/test" in result
        assert "M file.py" in result

    def test_empty_status_shows_clean(self) -> None:
        git = GitContext(current_branch="main", main_branch="main", status="", recent_commits="abc")
        result = git_status(AgentContext(git=git))
        assert result is not None
        assert "(clean)" in result


class TestScratchpad:
    def test_returns_none_without_scratchpad(self) -> None:
        assert scratchpad(AgentContext()) is None

    def test_renders_path(self) -> None:
        ctx = AgentContext(scratchpad_dir="/tmp/scratch")
        result = scratchpad(ctx)
        assert result is not None
        assert "/tmp/scratch" in result


class TestUserInstructions:
    def test_returns_none_without_instructions(self) -> None:
        assert user_instructions(AgentContext()) is None

    def test_renders_instructions(self) -> None:
        ctx = AgentContext(user_instructions="Be concise")
        result = user_instructions(ctx)
        assert result is not None
        assert "# User Instructions" in result
        assert "Be concise" in result


# ---------------------------------------------------------------------------
# Compose and profiles
# ---------------------------------------------------------------------------


class TestCompose:
    def test_empty_profile_returns_empty_string(self) -> None:
        assert compose([], AgentContext()) == ""

    def test_filters_none_sections(self) -> None:
        ctx = AgentContext()  # no tools → tool_usage_policy returns None
        result = compose([identity, tool_usage_policy], ctx)
        assert "OpenAgent" in result
        assert "Tool usage policy" not in result

    def test_joins_sections_with_double_newline(self) -> None:
        ctx = AgentContext(user_instructions="Be concise")
        result = compose([identity, user_instructions], ctx)
        parts = result.split("\n\n")
        assert len(parts) >= 2

    def test_fresh_session_includes_tools(self) -> None:
        result = compose(FRESH_SESSION, AgentContext(tools=core_tools()))
        assert "OpenAgent" in result
        assert "# Tools" in result
        assert "## bash" in result

    def test_section_ordering_matches_profile(self) -> None:
        ctx = AgentContext(
            tools=core_tools(),
            skills=[Skill(name="commit", description="desc", path="/p")],
            user_instructions="Be concise",
        )
        result = compose(FRESH_SESSION, ctx)
        identity_pos = result.index("OpenAgent")
        tools_pos = result.index("# Tools")
        skills_pos = result.index("# Skills")
        instructions_pos = result.index("# User Instructions")
        assert identity_pos < tools_pos < skills_pos < instructions_pos

    def test_custom_profile_only_includes_specified_sections(self) -> None:
        ctx = AgentContext(user_instructions="Be concise")
        result = compose([identity, user_instructions], ctx)
        assert "OpenAgent" in result
        assert "Be concise" in result
        assert "Doing tasks" not in result

    def test_resumed_session_matches_fresh(self) -> None:
        ctx = AgentContext(tools=core_tools())
        assert compose(FRESH_SESSION, ctx) == compose(RESUMED_SESSION, ctx)
