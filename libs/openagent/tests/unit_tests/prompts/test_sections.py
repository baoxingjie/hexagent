"""Tests for prompt sections, compose, and profiles."""

# ruff: noqa: ARG002, S108, PLR2004

from datetime import date

import pytest
from pydantic import BaseModel

from openagent.prompts import FRESH_SESSION, RESUMED_SESSION, compose
from openagent.prompts.sections import (
    GitContext,
    PromptContext,
    _substitute,
    _tool_vars,
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
from openagent.tools.base import BaseAgentTool
from openagent.types import MCPServer, Skill, ToolResult

# ---------------------------------------------------------------------------
# Mock tools
# ---------------------------------------------------------------------------


class MockParams(BaseModel):
    arg: str = ""


class MockBashTool(BaseAgentTool[MockParams]):
    name: str = "bash"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="")


class MockReadTool(BaseAgentTool[MockParams]):
    name: str = "read"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="")


class MockEditTool(BaseAgentTool[MockParams]):
    name: str = "edit"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="")


class MockWriteTool(BaseAgentTool[MockParams]):
    name: str = "write"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="")


class MockGlobTool(BaseAgentTool[MockParams]):
    name: str = "glob"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="")


class MockGrepTool(BaseAgentTool[MockParams]):
    name: str = "grep"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="")


class MockWebSearchTool(BaseAgentTool[MockParams]):
    name: str = "web_search"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="")


class MockCustomTool(BaseAgentTool[MockParams]):
    name: str = "custom"
    instruction: str = "This is a custom tool with inline instructions."
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class TestSubstitute:
    """Tests for _substitute helper."""

    def test_replaces_vars(self) -> None:
        assert _substitute("Hello ${name}!", name="World") == "Hello World!"

    def test_leaves_dollar_signs_alone(self) -> None:
        assert _substitute("$(cat file) and $HOME", name="x") == "$(cat file) and $HOME"

    def test_no_vars_is_noop(self) -> None:
        text = "No vars here: $dollar ${unknown}"
        assert _substitute(text) == text


class TestToolVars:
    """Tests for _tool_vars helper."""

    def test_builds_vars_from_tools(self) -> None:
        ctx = PromptContext(tools=[MockBashTool(), MockReadTool()])
        result = _tool_vars(ctx)
        assert result == {"BASH_TOOL_NAME": "bash", "READ_TOOL_NAME": "read"}

    def test_empty_tools(self) -> None:
        ctx = PromptContext()
        assert _tool_vars(ctx) == {}


# ---------------------------------------------------------------------------
# Context types
# ---------------------------------------------------------------------------


class TestContextTypes:
    """Tests for GitContext and PromptContext."""

    def test_git_context_creation(self) -> None:
        git = GitContext(
            current_branch="main",
            main_branch="main",
            status="M file.py",
            recent_commits="abc123 Initial commit",
        )
        assert git.current_branch == "main"
        assert git.status == "M file.py"

    def test_prompt_context_defaults(self) -> None:
        ctx = PromptContext()
        assert ctx.tools == []
        assert ctx.skills == []
        assert ctx.mcps == []
        assert ctx.environment == {}
        assert ctx.user_instructions is None
        assert ctx.git is None
        assert ctx.scratchpad_dir is None

    def test_prompt_context_full(self) -> None:
        git = GitContext(
            current_branch="feat",
            main_branch="main",
            status="clean",
            recent_commits="abc",
        )
        ctx = PromptContext(
            tools=[MockBashTool()],
            skills=[Skill(name="commit", description="desc", path="/p")],
            mcps=[MCPServer(name="gh", description="desc")],
            environment={"OS": "Linux"},
            user_instructions="Be concise",
            git=git,
            scratchpad_dir="/tmp/scratch",
        )
        assert len(ctx.tools) == 1
        assert len(ctx.skills) == 1
        assert ctx.git is not None
        assert ctx.scratchpad_dir == "/tmp/scratch"


# ---------------------------------------------------------------------------
# Section functions
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_returns_content(self) -> None:
        result = identity(PromptContext())
        assert result is not None
        assert "OpenAgent" in result


class TestDoingTasks:
    def test_returns_content(self) -> None:
        result = doing_tasks(PromptContext())
        assert result is not None
        assert "Doing tasks" in result


class TestExecutingActions:
    def test_returns_content(self) -> None:
        result = executing_actions(PromptContext())
        assert result is not None
        assert "Executing actions" in result


class TestToneAndStyle:
    def test_returns_content(self) -> None:
        result = tone_and_style(PromptContext())
        assert result is not None
        assert "Tone and style" in result

    def test_substitutes_tool_vars(self) -> None:
        ctx = PromptContext(tools=[MockBashTool()])
        result = tone_and_style(ctx)
        assert result is not None
        assert "${BASH_TOOL_NAME}" not in result


class TestToolUsagePolicy:
    def test_none_without_tools(self) -> None:
        assert tool_usage_policy(PromptContext()) is None

    def test_returns_content_with_tools(self) -> None:
        ctx = PromptContext(tools=[MockBashTool()])
        result = tool_usage_policy(ctx)
        assert result is not None
        assert "Tool usage policy" in result

    def test_substitutes_tool_vars(self) -> None:
        ctx = PromptContext(tools=[MockBashTool(), MockReadTool(), MockEditTool(), MockWriteTool()])
        result = tool_usage_policy(ctx)
        assert result is not None
        assert "${READ_TOOL_NAME}" not in result
        assert "read" in result
        assert "edit" in result
        assert "write" in result


class TestToolInstructions:
    def test_none_without_tools(self) -> None:
        assert tool_instructions(PromptContext()) is None

    def test_single_tool(self) -> None:
        ctx = PromptContext(tools=[MockBashTool()])
        result = tool_instructions(ctx)
        assert result is not None
        assert "# Tools" in result
        assert "## bash" in result

    def test_multiple_tools(self) -> None:
        ctx = PromptContext(tools=[MockBashTool(), MockReadTool()])
        result = tool_instructions(ctx)
        assert result is not None
        assert "## bash" in result
        assert "## read" in result
        assert "---" in result

    def test_supplementary_fragments(self) -> None:
        ctx = PromptContext(tools=[MockBashTool()])
        result = tool_instructions(ctx)
        assert result is not None
        assert "Committing changes with git" in result

    def test_tool_name_cross_references(self) -> None:
        ctx = PromptContext(
            tools=[
                MockBashTool(),
                MockGlobTool(),
                MockGrepTool(),
                MockReadTool(),
                MockEditTool(),
                MockWriteTool(),
            ]
        )
        result = tool_instructions(ctx)
        assert result is not None
        assert "${GLOB_TOOL_NAME}" not in result
        assert "${GREP_TOOL_NAME}" not in result
        assert "${READ_TOOL_NAME}" not in result
        assert "${EDIT_TOOL_NAME}" not in result
        assert "${WRITE_TOOL_NAME}" not in result

    def test_git_parallel_note_substitution(self) -> None:
        ctx = PromptContext(tools=[MockBashTool()])
        result = tool_instructions(ctx)
        assert result is not None
        assert "${GIT_PARALLEL_NOTE}" not in result
        assert "run multiple tool calls in parallel" in result

    def test_missing_fragment_raises(self) -> None:
        class NoFragmentTool(BaseAgentTool[MockParams]):
            name: str = "nonexistent_tool_xyz"
            args_schema = MockParams

            async def execute(self, params: MockParams) -> ToolResult:
                return ToolResult(output="")

        ctx = PromptContext(tools=[NoFragmentTool()])
        with pytest.raises(KeyError, match="Missing tool instruction"):
            tool_instructions(ctx)

    def test_custom_tool_instruction(self) -> None:
        ctx = PromptContext(tools=[MockCustomTool()])
        result = tool_instructions(ctx)
        assert result is not None
        assert "## custom" in result
        assert "This is a custom tool with inline instructions." in result

    def test_date_substitution(self) -> None:
        ctx = PromptContext(tools=[MockWebSearchTool()])
        result = tool_instructions(ctx)
        assert result is not None
        assert "${CURRENT_DATE}" not in result
        today = date.today()  # noqa: DTZ011
        assert today.isoformat() in result


class TestSkills:
    def test_none_when_empty(self) -> None:
        assert skills(PromptContext()) is None

    def test_formats_skills(self) -> None:
        ctx = PromptContext(skills=[Skill(name="commit", description="Git commits", path="/skills/commit")])
        result = skills(ctx)
        assert result is not None
        assert "# Skills" in result
        assert "**commit**" in result
        assert "Git commits" in result


class TestMcps:
    def test_none_when_empty(self) -> None:
        assert mcps(PromptContext()) is None

    def test_formats_mcps(self) -> None:
        ctx = PromptContext(mcps=[MCPServer(name="github", description="GitHub API")])
        result = mcps(ctx)
        assert result is not None
        assert "# MCP Servers" in result
        assert "**github**" in result


class TestEnvironment:
    def test_none_when_empty(self) -> None:
        assert environment(PromptContext()) is None

    def test_formats_environment(self) -> None:
        ctx = PromptContext(environment={"OS": "Linux"})
        result = environment(ctx)
        assert result is not None
        assert "# Environment" in result
        assert "OS: Linux" in result


class TestGitStatus:
    def test_none_without_git(self) -> None:
        assert git_status(PromptContext()) is None

    def test_renders_git_context(self) -> None:
        git = GitContext(
            current_branch="feat/test",
            main_branch="main",
            status="M file.py",
            recent_commits="abc123 Initial commit",
        )
        ctx = PromptContext(git=git)
        result = git_status(ctx)
        assert result is not None
        assert "feat/test" in result
        assert "M file.py" in result

    def test_empty_status_shows_clean(self) -> None:
        git = GitContext(
            current_branch="main",
            main_branch="main",
            status="",
            recent_commits="abc",
        )
        ctx = PromptContext(git=git)
        result = git_status(ctx)
        assert result is not None
        assert "(clean)" in result


class TestScratchpad:
    def test_none_without_scratchpad(self) -> None:
        assert scratchpad(PromptContext()) is None

    def test_renders_path(self) -> None:
        ctx = PromptContext(scratchpad_dir="/tmp/scratch")
        result = scratchpad(ctx)
        assert result is not None
        assert "/tmp/scratch" in result


class TestUserInstructions:
    def test_none_without_instructions(self) -> None:
        assert user_instructions(PromptContext()) is None

    def test_renders_instructions(self) -> None:
        ctx = PromptContext(user_instructions="Be concise")
        result = user_instructions(ctx)
        assert result is not None
        assert "# User Instructions" in result
        assert "Be concise" in result


# ---------------------------------------------------------------------------
# Compose and profiles
# ---------------------------------------------------------------------------


class TestCompose:
    def test_empty_profile(self) -> None:
        assert compose([], PromptContext()) == ""

    def test_filters_none(self) -> None:
        """Sections returning None are excluded from output."""
        ctx = PromptContext()  # no tools → tool_usage_policy returns None
        result = compose([identity, tool_usage_policy], ctx)
        assert "OpenAgent" in result
        assert "Tool usage policy" not in result

    def test_joins_with_double_newline(self) -> None:
        ctx = PromptContext(user_instructions="Be concise")
        result = compose([identity, user_instructions], ctx)
        # Both sections should be present, separated by \n\n
        parts = result.split("\n\n")
        assert len(parts) >= 2

    def test_fresh_session_minimal(self) -> None:
        result = compose(FRESH_SESSION, PromptContext())
        assert "OpenAgent" in result
        assert "# Tools" not in result  # No tools registered

    def test_fresh_session_with_tools(self) -> None:
        ctx = PromptContext(tools=[MockBashTool(), MockReadTool()])
        result = compose(FRESH_SESSION, ctx)
        assert "# Tools" in result
        assert "## bash" in result
        assert "## read" in result

    def test_section_order(self) -> None:
        ctx = PromptContext(
            tools=[MockBashTool()],
            skills=[Skill(name="commit", description="desc", path="/p")],
            user_instructions="Be concise",
        )
        result = compose(FRESH_SESSION, ctx)
        identity_pos = result.index("OpenAgent")
        tools_pos = result.index("# Tools")
        skills_pos = result.index("# Skills")
        instructions_pos = result.index("# User Instructions")
        assert identity_pos < tools_pos < skills_pos < instructions_pos

    def test_custom_profile(self) -> None:
        ctx = PromptContext(user_instructions="Be concise")
        result = compose([identity, user_instructions], ctx)
        assert "OpenAgent" in result
        assert "Be concise" in result
        assert "Doing tasks" not in result  # Not in profile

    def test_resumed_session_same_as_fresh(self) -> None:
        ctx = PromptContext(tools=[MockBashTool()])
        fresh = compose(FRESH_SESSION, ctx)
        resumed = compose(RESUMED_SESSION, ctx)
        assert fresh == resumed
