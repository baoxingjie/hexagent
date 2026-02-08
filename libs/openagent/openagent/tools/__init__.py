"""Tool implementations for openagent.

This module provides concrete tool implementations that follow Anthropic's
tool patterns. Tools in this module depend on core/ for result types and
computer/ for Computer implementations.

Base class:
- BaseAgentTool: Abstract base class for agent tools

CLI tools:
- BashTool: Execute bash commands on a Computer
- ReadTool: Read file contents with line numbers
- WriteTool: Create or overwrite files
- EditTool: Perform string replacements in files
- GlobTool: Find files by pattern
- GrepTool: Search for patterns in files

Web tools:
- WebSearchTool: Search the web for information
- WebFetchTool: Fetch and extract content from web pages

Skill tools:
- SkillTool: Invoke specialized skills by name

Factory functions:
- create_bash_tool: Create the bash tool
- create_filesystem_tools: Create file operation tools (read, write, edit, glob, grep)
- create_cli_tools: Create all CLI tools sharing a Computer instance

For LangChain integration, see openagent.langchain module.
"""

from typing import Any

from openagent.tools.base import BaseAgentTool
from openagent.tools.cli import (
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    ReadTool,
    WriteTool,
    create_bash_tool,
    create_cli_tools,
    create_filesystem_tools,
)
from openagent.tools.skill import SkillTool
from openagent.tools.web import (
    WebFetchTool,
    WebSearchTool,
)

BUILTIN_TOOLS: tuple[type[BaseAgentTool[Any]], ...] = (
    BashTool,
    ReadTool,
    WriteTool,
    EditTool,
    GlobTool,
    GrepTool,
    WebSearchTool,
    WebFetchTool,
    SkillTool,
)
"""The canonical set of built-in tool classes.

OpenAgent always has a computer — these tools are non-negotiable.
Everything that needs to know "what tools exist" derives from this
tuple (template variables, prompt fragment lookup, etc.).
"""

__all__ = [
    "BUILTIN_TOOLS",
    "BaseAgentTool",
    "BashTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "ReadTool",
    "SkillTool",
    "WebFetchTool",
    "WebSearchTool",
    "WriteTool",
    "create_bash_tool",
    "create_cli_tools",
    "create_filesystem_tools",
]
