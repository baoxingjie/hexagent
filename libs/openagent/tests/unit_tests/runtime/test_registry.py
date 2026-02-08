"""Tests for CapabilityRegistry."""

# ruff: noqa: ARG002, PLR2004

import pytest
from pydantic import BaseModel

from openagent.runtime.registry import CapabilityRegistry
from openagent.tools.base import BaseAgentTool
from openagent.types import MCPServer, Skill, ToolResult


class MockParams(BaseModel):
    """Mock params for testing."""


class MockTool(BaseAgentTool[MockParams]):
    """Mock tool for testing."""

    name = "mock_tool"
    description = "A mock tool for testing"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="mock output")


class AnotherMockTool(BaseAgentTool[MockParams]):
    """Another mock tool for testing."""

    name = "another_tool"
    description = "Another mock tool"
    args_schema = MockParams

    async def execute(self, params: MockParams) -> ToolResult:
        return ToolResult(output="another output")


class TestCapabilityRegistryTools:
    """Tests for tool registration."""

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        registry = CapabilityRegistry()
        tool = MockTool()
        registry.register_tool(tool)
        assert registry.has_tool("mock_tool")

    def test_register_tool_duplicate_raises(self) -> None:
        """Test registering duplicate tool raises ValueError."""
        registry = CapabilityRegistry()
        tool = MockTool()
        registry.register_tool(tool)

        with pytest.raises(ValueError, match="Tool already registered"):
            registry.register_tool(tool)

    def test_unregister_tool(self) -> None:
        """Test unregistering a tool."""
        registry = CapabilityRegistry()
        tool = MockTool()
        registry.register_tool(tool)
        registry.unregister_tool("mock_tool")
        assert not registry.has_tool("mock_tool")

    def test_unregister_tool_not_found_raises(self) -> None:
        """Test unregistering non-existent tool raises KeyError."""
        registry = CapabilityRegistry()
        with pytest.raises(KeyError, match="Tool not registered"):
            registry.unregister_tool("nonexistent")

    def test_get_tool(self) -> None:
        """Test getting a tool by name."""
        registry = CapabilityRegistry()
        tool = MockTool()
        registry.register_tool(tool)
        result = registry.get_tool("mock_tool")
        assert result is tool

    def test_get_tool_not_found_raises(self) -> None:
        """Test getting non-existent tool raises KeyError."""
        registry = CapabilityRegistry()
        with pytest.raises(KeyError, match="Tool not registered"):
            registry.get_tool("nonexistent")

    def test_get_tools_empty(self) -> None:
        """Test getting tools from empty registry."""
        registry = CapabilityRegistry()
        assert registry.get_tools() == []

    def test_get_tools_multiple(self) -> None:
        """Test getting multiple tools."""
        registry = CapabilityRegistry()
        tool1 = MockTool()
        tool2 = AnotherMockTool()
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        tools = registry.get_tools()
        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools

    def test_has_tool_true(self) -> None:
        """Test has_tool returns True for registered tool."""
        registry = CapabilityRegistry()
        registry.register_tool(MockTool())
        assert registry.has_tool("mock_tool") is True

    def test_has_tool_false(self) -> None:
        """Test has_tool returns False for unregistered tool."""
        registry = CapabilityRegistry()
        assert registry.has_tool("nonexistent") is False


class TestCapabilityRegistrySkills:
    """Tests for skill registration."""

    def test_register_skill(self) -> None:
        """Test registering a skill."""
        registry = CapabilityRegistry()
        skill = Skill(name="commit", description="Create git commits", path="/skills/commit")
        registry.register_skill(skill)
        assert registry.has_skill("commit")

    def test_register_skill_duplicate_raises(self) -> None:
        """Test registering duplicate skill raises ValueError."""
        registry = CapabilityRegistry()
        skill = Skill(name="commit", description="Create git commits", path="/skills/commit")
        registry.register_skill(skill)

        with pytest.raises(ValueError, match="Skill already registered"):
            registry.register_skill(skill)

    def test_unregister_skill(self) -> None:
        """Test unregistering a skill."""
        registry = CapabilityRegistry()
        skill = Skill(name="commit", description="Create git commits", path="/skills/commit")
        registry.register_skill(skill)
        registry.unregister_skill("commit")
        assert not registry.has_skill("commit")

    def test_unregister_skill_not_found_raises(self) -> None:
        """Test unregistering non-existent skill raises KeyError."""
        registry = CapabilityRegistry()
        with pytest.raises(KeyError, match="Skill not registered"):
            registry.unregister_skill("nonexistent")

    def test_get_skill(self) -> None:
        """Test getting a skill by name."""
        registry = CapabilityRegistry()
        skill = Skill(name="commit", description="Create git commits", path="/skills/commit")
        registry.register_skill(skill)
        result = registry.get_skill("commit")
        assert result is skill

    def test_get_skill_not_found_raises(self) -> None:
        """Test getting non-existent skill raises KeyError."""
        registry = CapabilityRegistry()
        with pytest.raises(KeyError, match="Skill not registered"):
            registry.get_skill("nonexistent")

    def test_get_skills_empty(self) -> None:
        """Test getting skills from empty registry."""
        registry = CapabilityRegistry()
        assert registry.get_skills() == []

    def test_get_skills_multiple(self) -> None:
        """Test getting multiple skills."""
        registry = CapabilityRegistry()
        skill1 = Skill(name="commit", description="Create git commits", path="/skills/commit")
        skill2 = Skill(name="review", description="Review code", path="/skills/review")
        registry.register_skill(skill1)
        registry.register_skill(skill2)
        skills = registry.get_skills()
        assert len(skills) == 2
        assert skill1 in skills
        assert skill2 in skills

    def test_has_skill_true(self) -> None:
        """Test has_skill returns True for registered skill."""
        registry = CapabilityRegistry()
        registry.register_skill(Skill(name="commit", description="desc", path="/skills/commit"))
        assert registry.has_skill("commit") is True

    def test_has_skill_false(self) -> None:
        """Test has_skill returns False for unregistered skill."""
        registry = CapabilityRegistry()
        assert registry.has_skill("nonexistent") is False


class TestCapabilityRegistryMCPs:
    """Tests for MCP server registration."""

    def test_register_mcp(self) -> None:
        """Test registering an MCP server."""
        registry = CapabilityRegistry()
        mcp = MCPServer(name="context7", description="Documentation lookup")
        registry.register_mcp(mcp)
        assert registry.has_mcp("context7")

    def test_register_mcp_duplicate_raises(self) -> None:
        """Test registering duplicate MCP raises ValueError."""
        registry = CapabilityRegistry()
        mcp = MCPServer(name="context7", description="Documentation lookup")
        registry.register_mcp(mcp)

        with pytest.raises(ValueError, match="MCP server already registered"):
            registry.register_mcp(mcp)

    def test_unregister_mcp(self) -> None:
        """Test unregistering an MCP server."""
        registry = CapabilityRegistry()
        mcp = MCPServer(name="context7", description="Documentation lookup")
        registry.register_mcp(mcp)
        registry.unregister_mcp("context7")
        assert not registry.has_mcp("context7")

    def test_unregister_mcp_not_found_raises(self) -> None:
        """Test unregistering non-existent MCP raises KeyError."""
        registry = CapabilityRegistry()
        with pytest.raises(KeyError, match="MCP server not registered"):
            registry.unregister_mcp("nonexistent")

    def test_get_mcp(self) -> None:
        """Test getting an MCP server by name."""
        registry = CapabilityRegistry()
        mcp = MCPServer(name="context7", description="Documentation lookup")
        registry.register_mcp(mcp)
        result = registry.get_mcp("context7")
        assert result is mcp

    def test_get_mcp_not_found_raises(self) -> None:
        """Test getting non-existent MCP raises KeyError."""
        registry = CapabilityRegistry()
        with pytest.raises(KeyError, match="MCP server not registered"):
            registry.get_mcp("nonexistent")

    def test_get_mcps_empty(self) -> None:
        """Test getting MCPs from empty registry."""
        registry = CapabilityRegistry()
        assert registry.get_mcps() == []

    def test_get_mcps_multiple(self) -> None:
        """Test getting multiple MCPs."""
        registry = CapabilityRegistry()
        mcp1 = MCPServer(name="context7", description="Documentation lookup")
        mcp2 = MCPServer(name="github", description="GitHub integration")
        registry.register_mcp(mcp1)
        registry.register_mcp(mcp2)
        mcps = registry.get_mcps()
        assert len(mcps) == 2
        assert mcp1 in mcps
        assert mcp2 in mcps

    def test_has_mcp_true(self) -> None:
        """Test has_mcp returns True for registered MCP."""
        registry = CapabilityRegistry()
        registry.register_mcp(MCPServer(name="context7", description="desc"))
        assert registry.has_mcp("context7") is True

    def test_has_mcp_false(self) -> None:
        """Test has_mcp returns False for unregistered MCP."""
        registry = CapabilityRegistry()
        assert registry.has_mcp("nonexistent") is False


class TestCapabilityRegistryClear:
    """Tests for clear functionality."""

    def test_clear_empty_registry(self) -> None:
        """Test clearing an empty registry is safe."""
        registry = CapabilityRegistry()
        registry.clear()  # Should not raise

    def test_clear_removes_all(self) -> None:
        """Test clear removes all capabilities."""
        registry = CapabilityRegistry()
        registry.register_tool(MockTool())
        registry.register_skill(Skill(name="commit", description="desc", path="/skills/commit"))
        registry.register_mcp(MCPServer(name="context7", description="desc"))

        registry.clear()

        assert registry.get_tools() == []
        assert registry.get_skills() == []
        assert registry.get_mcps() == []
