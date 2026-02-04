"""Tests for SystemPromptAssembler."""

# ruff: noqa: ARG002, RUF012

from pydantic import BaseModel

from openagent.runtime.prompt import SystemPromptAssembler
from openagent.tools.base import BaseAgentTool
from openagent.types import MCPServer, Skill, ToolResult


class MockParams(BaseModel):
    """Mock params for testing."""


class MockTool(BaseAgentTool[MockParams]):
    """Mock tool for testing."""

    name = "bash"
    description = "Execute shell commands"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="mock")


class AnotherTool(BaseAgentTool[MockParams]):
    """Another mock tool for testing."""

    name = "read"
    description = "Read files"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="mock")


class TestSystemPromptAssembler:
    """Tests for SystemPromptAssembler class."""

    def test_order_is_explicit(self) -> None:
        """Test ORDER is a visible class attribute."""
        assert hasattr(SystemPromptAssembler, "ORDER")
        assert isinstance(SystemPromptAssembler.ORDER, list)
        assert len(SystemPromptAssembler.ORDER) > 0

    def test_order_contains_expected_sections(self) -> None:
        """Test ORDER contains all expected sections."""
        expected = ["base", "tools", "skills", "mcps", "environment", "user_instructions"]
        assert expected == SystemPromptAssembler.ORDER

    def test_assemble_base_only(self) -> None:
        """Test assembling with only base prompt."""
        assembler = SystemPromptAssembler()
        result = assembler.assemble(base="You are a helpful assistant.")
        assert result == "You are a helpful assistant."

    def test_assemble_with_tools(self) -> None:
        """Test assembling with tools."""
        assembler = SystemPromptAssembler()
        tools = [MockTool(), AnotherTool()]
        result = assembler.assemble(
            base="You are a helpful assistant.",
            tools=tools,
        )
        assert "You are a helpful assistant." in result
        assert "## Tools" in result
        assert "**bash**" in result
        assert "Execute shell commands" in result
        assert "**read**" in result
        assert "Read files" in result

    def test_assemble_with_skills(self) -> None:
        """Test assembling with skills."""
        assembler = SystemPromptAssembler()
        skills = [
            Skill(name="commit", description="Create git commits"),
            Skill(name="review", description="Review code changes"),
        ]
        result = assembler.assemble(
            base="You are a helpful assistant.",
            skills=skills,
        )
        assert "## Skills" in result
        assert "**commit**" in result
        assert "Create git commits" in result
        assert "**review**" in result

    def test_assemble_with_mcps(self) -> None:
        """Test assembling with MCP servers."""
        assembler = SystemPromptAssembler()
        mcps = [
            MCPServer(name="context7", description="Documentation lookup"),
        ]
        result = assembler.assemble(
            base="You are a helpful assistant.",
            mcps=mcps,
        )
        assert "## MCP Servers" in result
        assert "**context7**" in result
        assert "Documentation lookup" in result

    def test_assemble_with_environment(self) -> None:
        """Test assembling with environment context."""
        assembler = SystemPromptAssembler()
        result = assembler.assemble(
            base="You are a helpful assistant.",
            environment={"platform": "darwin", "cwd": "/home/user"},
        )
        assert "## Environment" in result
        assert "platform: darwin" in result
        assert "cwd: /home/user" in result

    def test_assemble_with_user_instructions(self) -> None:
        """Test assembling with user instructions."""
        assembler = SystemPromptAssembler()
        result = assembler.assemble(
            base="You are a helpful assistant.",
            user_instructions="Focus on Python code.",
        )
        assert "## User Instructions" in result
        assert "Focus on Python code." in result

    def test_assemble_respects_order(self) -> None:
        """Test sections appear in ORDER sequence."""
        assembler = SystemPromptAssembler()
        result = assembler.assemble(
            base="BASE_MARKER",
            tools=[MockTool()],
            skills=[Skill(name="s1", description="d1")],
            mcps=[MCPServer(name="m1", description="d1")],
            environment={"key": "ENV_MARKER"},
            user_instructions="USER_MARKER",
        )
        # Check order: base, tools, skills, mcps, environment, user_instructions
        base_pos = result.index("BASE_MARKER")
        tools_pos = result.index("## Tools")
        skills_pos = result.index("## Skills")
        mcps_pos = result.index("## MCP Servers")
        env_pos = result.index("ENV_MARKER")
        user_pos = result.index("USER_MARKER")

        assert base_pos < tools_pos
        assert tools_pos < skills_pos
        assert skills_pos < mcps_pos
        assert mcps_pos < env_pos
        assert env_pos < user_pos

    def test_assemble_skips_none_sections(self) -> None:
        """Test None sections are skipped."""
        assembler = SystemPromptAssembler()
        result = assembler.assemble(
            base="Base prompt",
            tools=None,
            skills=None,
            user_instructions="Instructions",
        )
        assert "## Tools" not in result
        assert "## Skills" not in result
        assert "Base prompt" in result
        assert "Instructions" in result

    def test_assemble_skips_empty_lists(self) -> None:
        """Test empty lists are skipped."""
        assembler = SystemPromptAssembler()
        result = assembler.assemble(
            base="Base prompt",
            tools=[],
            skills=[],
            mcps=[],
        )
        assert "## Tools" not in result
        assert "## Skills" not in result
        assert "## MCP Servers" not in result

    def test_assemble_sections_separated_by_double_newline(self) -> None:
        """Test sections are separated by double newline."""
        assembler = SystemPromptAssembler()
        result = assembler.assemble(
            base="Base",
            tools=[MockTool()],
        )
        # Should have double newline between base and tools
        assert "Base\n\n## Tools" in result

    def test_subclass_can_reorder(self) -> None:
        """Test subclass can change ORDER."""

        class CustomAssembler(SystemPromptAssembler):
            ORDER = ["base", "user_instructions", "tools"]

        assembler = CustomAssembler()
        result = assembler.assemble(
            base="BASE",
            tools=[MockTool()],
            user_instructions="USER",
        )
        # With custom order: base, user_instructions, tools
        base_pos = result.index("BASE")
        user_pos = result.index("USER")
        tools_pos = result.index("## Tools")

        assert base_pos < user_pos
        assert user_pos < tools_pos
