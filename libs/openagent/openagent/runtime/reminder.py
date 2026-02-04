"""System reminder injection for conversation augmentation.

This module provides the SystemReminder class which injects contextual
reminders into messages based on conditions.

Design follows Claude Code's pattern: tiny reminders at the right time
change agent behavior. Reminders are wrapped in <system-reminder> tags
and injected into user messages.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# Type alias for condition functions
# Uses `object` to accept any state type (strict mypy compatible)
ConditionFn = Callable[[object], bool]


@dataclass(frozen=True)
class Reminder:
    """A conditional reminder.

    Attributes:
        condition: Function that takes state and returns True if reminder should trigger.
        text: The reminder text to inject when condition is met.
    """

    condition: ConditionFn
    text: str


class SystemReminder:
    """Injects reminders into messages based on conditions.

    Follows Claude Code's <system-reminder> pattern for contextual nudges
    that keep the agent on track during long conversations.

    Examples:
        Basic usage:
        ```python
        reminders = SystemReminder()

        # Register conditional reminders
        reminders.add(
            condition=lambda s: s.turns_since_task_tool > 5,
            text="Consider using task tools to track progress.",
        )
        reminders.add(
            condition=lambda s: s.last_modified_file is not None,
            text="A file was recently modified. Take it into account.",
        )

        # In middleware, augment user message before sending to LLM
        augmented = reminders.inject(user_message, state)
        ```

        Custom tag:
        ```python
        reminders = SystemReminder(tag="agent-hint")
        # Produces: <agent-hint>...</agent-hint>
        ```
    """

    def __init__(self, tag: str = "system-reminder") -> None:
        """Initialize the reminder system.

        Args:
            tag: XML tag name for wrapping reminders. Default "system-reminder".
        """
        self.tag = tag
        self._reminders: list[Reminder] = []

    def add(self, condition: ConditionFn, text: str) -> None:
        """Register a conditional reminder.

        Args:
            condition: Function that takes state and returns True to trigger.
            text: Reminder text to inject when condition is met.
        """
        self._reminders.append(Reminder(condition=condition, text=text))

    def clear(self) -> None:
        """Remove all registered reminders."""
        self._reminders.clear()

    def check(self, state: object) -> list[str]:
        """Get all triggered reminder texts.

        Args:
            state: Conversation state to evaluate conditions against.

        Returns:
            List of reminder texts for all triggered conditions.
        """
        return [r.text for r in self._reminders if r.condition(state)]

    def format(self, text: str) -> str:
        """Wrap text in reminder tags.

        Args:
            text: The reminder text.

        Returns:
            Text wrapped in <tag>...</tag>.
        """
        return f"<{self.tag}>\n{text}\n</{self.tag}>"

    def inject(self, message: str, state: object) -> str:
        """Augment message with triggered reminders.

        Checks all conditions against state and prepends triggered
        reminders to the message.

        Args:
            message: The original message content.
            state: Conversation state to evaluate conditions against.

        Returns:
            Message with reminders prepended, or original if none triggered.
        """
        triggered = self.check(state)
        if not triggered:
            return message

        blocks = [self.format(text) for text in triggered]
        reminder_section = "\n\n".join(blocks)
        return f"{reminder_section}\n\n{message}"
