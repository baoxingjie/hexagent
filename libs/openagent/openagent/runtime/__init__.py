"""Runtime infrastructure modules for OpenAgent.

This package contains framework-agnostic modules that manage
agent runtime behavior:

- CapabilityRegistry: Stores tools, skills, and MCP servers (pure data)
- SystemPromptAssembler: Builds system prompts with explicit ORDER
- ContextManager: Stateless compaction decision (threshold check)
- SystemReminder: Conditional message augmentation
- PermissionGate: Validates tool calls and handles approvals
"""

from openagent.runtime.context import ContextManager
from openagent.runtime.permission import (
    PermissionDecision,
    PermissionGate,
    PermissionResult,
    SafetyRule,
)
from openagent.runtime.prompt import SystemPromptAssembler
from openagent.runtime.registry import CapabilityRegistry
from openagent.runtime.reminder import Reminder, SystemReminder

__all__ = [
    "CapabilityRegistry",
    "ContextManager",
    "PermissionDecision",
    "PermissionGate",
    "PermissionResult",
    "Reminder",
    "SafetyRule",
    "SystemPromptAssembler",
    "SystemReminder",
]
