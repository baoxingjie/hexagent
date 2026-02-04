"""OpenAgent package.

OpenAgent is an Agent SDK (supporting OpenAI-compatible LLMs) similar to
Anthropic's Claude Agent SDK.

Core Philosophy: Give agents a CLI-based computer, allowing them to work
like humans do.
"""

from openagent.langchain import create_agent
from openagent.runtime import (
    CapabilityRegistry,
    ContextManager,
    PermissionDecision,
    PermissionGate,
    PermissionResult,
    SafetyRule,
    SystemPromptAssembler,
    SystemReminder,
)

__all__ = [
    "CapabilityRegistry",
    "ContextManager",
    "PermissionDecision",
    "PermissionGate",
    "PermissionResult",
    "SafetyRule",
    "SystemPromptAssembler",
    "SystemReminder",
    "create_agent",
]
