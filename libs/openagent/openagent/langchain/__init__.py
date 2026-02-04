"""LangChain integration for OpenAgent.

This module provides adapters and utilities for integrating OpenAgent's
framework-agnostic tools and computer abstractions with LangChain.

If you delete this directory, all core OpenAgent functionality
(tools, computer, types, runtime) should still work independently.

Main exports:
- create_agent: Create an OpenAgent agent using LangChain
- AgentMiddleware: LangChain middleware that wires runtime modules
- ApprovalCallback: Protocol for human-in-the-loop tool approval
- to_langchain_tool: Convert BaseAgentTool to LangChain StructuredTool
"""

from openagent.langchain.adapter import to_langchain_tool
from openagent.langchain.agent import create_agent
from openagent.langchain.middleware import AgentMiddleware, ApprovalCallback

__all__ = [
    "AgentMiddleware",
    "ApprovalCallback",
    "create_agent",
    "to_langchain_tool",
]
