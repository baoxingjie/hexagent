"""OpenAgent package.

OpenAgent is an Agent SDK (supporting OpenAI-compatible LLMs) similar to
Anthropic's Claude Agent SDK.

Core Philosophy: Give agents a CLI-based computer, allowing them to work
like humans do.
"""

from openagent.config import AgentConfig, CompactionConfig, SkillsConfig
from openagent.langchain import create_agent
from openagent.prompts import (
    FRESH_SESSION,
    RESUMED_SESSION,
    GitContext,
    PromptContext,
    compose,
)
from openagent.runtime import (
    Append,
    CapabilityRegistry,
    CompactionController,
    CompactionPhase,
    Inject,
    Overwrite,
    PermissionDecision,
    PermissionGate,
    PermissionResult,
    SafetyRule,
    SkillResolver,
)

__all__ = [
    "FRESH_SESSION",
    "RESUMED_SESSION",
    "AgentConfig",
    "Append",
    "CapabilityRegistry",
    "CompactionConfig",
    "CompactionController",
    "CompactionPhase",
    "GitContext",
    "Inject",
    "Overwrite",
    "PermissionDecision",
    "PermissionGate",
    "PermissionResult",
    "PromptContext",
    "SafetyRule",
    "SkillResolver",
    "SkillsConfig",
    "compose",
    "create_agent",
]
