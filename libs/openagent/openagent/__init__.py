"""OpenAgent package.

OpenAgent is an Agent SDK (supporting OpenAI-compatible LLMs) similar to
Anthropic's Claude Agent SDK.

Core Philosophy: Give agents a CLI-based computer, allowing them to work
like humans do.
"""

from openagent.config import AgentConfig, CompactionConfig, SkillsConfig
from openagent.harness import (
    PermissionDecision,
    PermissionGate,
    PermissionResult,
    Reminder,
    SafetyRule,
    SkillResolver,
)
from openagent.langchain import create_agent
from openagent.prompts import (
    FRESH_SESSION,
    RESUMED_SESSION,
    compose,
)
from openagent.types import AgentContext, CompactionPhase, GitContext, SkillCatalog

__all__ = [
    "FRESH_SESSION",
    "RESUMED_SESSION",
    "AgentConfig",
    "AgentContext",
    "CompactionConfig",
    "CompactionPhase",
    "GitContext",
    "PermissionDecision",
    "PermissionGate",
    "PermissionResult",
    "Reminder",
    "SafetyRule",
    "SkillCatalog",
    "SkillResolver",
    "SkillsConfig",
    "compose",
    "create_agent",
]
