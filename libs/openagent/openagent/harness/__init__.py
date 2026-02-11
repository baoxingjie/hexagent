"""Agent harness infrastructure.

This package contains modules that augment the agent loop:
- Permission gating (safety rules, human-in-the-loop approval)
- Skill discovery and content loading
- System reminder rules for dynamic message annotation
"""

from openagent.harness.permission import (
    PermissionDecision,
    PermissionGate,
    PermissionResult,
    SafetyRule,
)
from openagent.harness.reminders import (
    BUILTIN_REMINDERS,
    REMINDER_TAG,
    Reminder,
    evaluate_reminders,
    initial_available_skills,
)
from openagent.harness.skills import SkillResolver

__all__ = [
    "BUILTIN_REMINDERS",
    "REMINDER_TAG",
    "PermissionDecision",
    "PermissionGate",
    "PermissionResult",
    "Reminder",
    "SafetyRule",
    "SkillResolver",
    "evaluate_reminders",
    "initial_available_skills",
]
