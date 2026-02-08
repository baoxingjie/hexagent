"""Capability registry for managing agent tools, skills, and MCP servers.

This module provides the CapabilityRegistry class which serves as the
single source of truth for what capabilities an agent has access to.

The registry is a pure data store. It does NOT handle formatting - the
prompt system handles that.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openagent.tools.base import BaseAgentTool
    from openagent.types import MCPServer, Skill


class CapabilityRegistry:
    """Registry for agent capabilities.

    Pure data container for tools, skills, and MCP servers.
    Does NOT handle formatting - the prompt system handles that.

    Examples:
        ```python
        registry = CapabilityRegistry()

        # Register tools
        registry.register_tool(bash_tool)
        registry.register_tool(read_tool)

        # Register skills
        registry.register_skill(Skill(name="commit", description="Create git commits"))

        # Get data for the assembler
        tools = registry.get_tools()
        skills = registry.get_skills()
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._tools: dict[str, BaseAgentTool[Any]] = {}
        self._skills: dict[str, Skill] = {}
        self._mcps: dict[str, MCPServer] = {}

    # --- Tools ---

    def register_tool(self, tool: BaseAgentTool[Any]) -> None:
        """Register a tool.

        Args:
            tool: The tool to register. Uses tool.name as the key.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            msg = f"Tool already registered: {tool.name}"
            raise ValueError(msg)
        self._tools[tool.name] = tool

    def unregister_tool(self, name: str) -> None:
        """Unregister a tool by name.

        Args:
            name: The name of the tool to unregister.

        Raises:
            KeyError: If no tool with the given name is registered.
        """
        if name not in self._tools:
            msg = f"Tool not registered: {name}"
            raise KeyError(msg)
        del self._tools[name]

    def get_tool(self, name: str) -> BaseAgentTool[Any]:
        """Get a tool by name.

        Args:
            name: The name of the tool.

        Returns:
            The registered tool.

        Raises:
            KeyError: If no tool with the given name is registered.
        """
        if name not in self._tools:
            msg = f"Tool not registered: {name}"
            raise KeyError(msg)
        return self._tools[name]

    def get_tools(self) -> list[BaseAgentTool[Any]]:
        """Get all registered tools.

        Returns:
            List of all registered tools.
        """
        return list(self._tools.values())

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: The name of the tool.

        Returns:
            True if the tool is registered.
        """
        return name in self._tools

    # --- Skills ---

    def register_skill(self, skill: Skill) -> None:
        """Register a skill.

        Args:
            skill: The skill to register.

        Raises:
            ValueError: If a skill with the same name is already registered.
        """
        if skill.name in self._skills:
            msg = f"Skill already registered: {skill.name}"
            raise ValueError(msg)
        self._skills[skill.name] = skill

    def unregister_skill(self, name: str) -> None:
        """Unregister a skill by name.

        Args:
            name: The name of the skill to unregister.

        Raises:
            KeyError: If no skill with the given name is registered.
        """
        if name not in self._skills:
            msg = f"Skill not registered: {name}"
            raise KeyError(msg)
        del self._skills[name]

    def get_skill(self, name: str) -> Skill:
        """Get a skill by name.

        Args:
            name: The name of the skill.

        Returns:
            The registered skill.

        Raises:
            KeyError: If no skill with the given name is registered.
        """
        if name not in self._skills:
            msg = f"Skill not registered: {name}"
            raise KeyError(msg)
        return self._skills[name]

    def get_skills(self) -> list[Skill]:
        """Get all registered skills.

        Returns:
            List of all registered skills.
        """
        return list(self._skills.values())

    def has_skill(self, name: str) -> bool:
        """Check if a skill is registered.

        Args:
            name: The name of the skill.

        Returns:
            True if the skill is registered.
        """
        return name in self._skills

    # --- MCP Servers ---

    def register_mcp(self, server: MCPServer) -> None:
        """Register an MCP server.

        Args:
            server: The MCP server to register.

        Raises:
            ValueError: If an MCP server with the same name is already registered.
        """
        if server.name in self._mcps:
            msg = f"MCP server already registered: {server.name}"
            raise ValueError(msg)
        self._mcps[server.name] = server

    def unregister_mcp(self, name: str) -> None:
        """Unregister an MCP server by name.

        Args:
            name: The name of the MCP server to unregister.

        Raises:
            KeyError: If no MCP server with the given name is registered.
        """
        if name not in self._mcps:
            msg = f"MCP server not registered: {name}"
            raise KeyError(msg)
        del self._mcps[name]

    def get_mcp(self, name: str) -> MCPServer:
        """Get an MCP server by name.

        Args:
            name: The name of the MCP server.

        Returns:
            The registered MCP server.

        Raises:
            KeyError: If no MCP server with the given name is registered.
        """
        if name not in self._mcps:
            msg = f"MCP server not registered: {name}"
            raise KeyError(msg)
        return self._mcps[name]

    def get_mcps(self) -> list[MCPServer]:
        """Get all registered MCP servers.

        Returns:
            List of all registered MCP servers.
        """
        return list(self._mcps.values())

    def has_mcp(self, name: str) -> bool:
        """Check if an MCP server is registered.

        Args:
            name: The name of the MCP server.

        Returns:
            True if the MCP server is registered.
        """
        return name in self._mcps

    # --- Bulk Operations ---

    def clear(self) -> None:
        """Clear all registered capabilities."""
        self._tools.clear()
        self._skills.clear()
        self._mcps.clear()
