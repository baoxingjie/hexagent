"""Context manager for compaction decisions.

This module provides the ContextManager class which decides when
conversation context needs to be compacted based on token count.

Design principle: Stateless. The manager receives state and returns
a decision. No internal tracking that can get out of sync.
"""

from __future__ import annotations

DEFAULT_COMPACTION_THRESHOLD = 100_000

DEFAULT_COMPACTION_PROMPT = """Please summarize what you have accomplished so far and what remains to be done.

Focus on:
1. What tasks have been completed
2. What files have been modified
3. What tasks are still pending
4. Any important context that should be preserved

After providing the summary, I will start a fresh conversation with this summary as context."""


class ContextManager:
    """Decides when context needs compaction.

    Stateless design: Pass token count, get decision. No internal state
    to track or synchronize.

    Examples:
        Basic usage:
        ```python
        manager = ContextManager(threshold=100_000)

        # Calculate tokens from your message history
        token_count = count_tokens(messages)

        # Check if compaction needed
        if manager.needs_compaction(token_count):
            prompt = manager.compaction_prompt
            # Inject prompt to trigger summary...
        ```

        Custom threshold:
        ```python
        manager = ContextManager(threshold=50_000)  # More aggressive
        ```
    """

    def __init__(self, threshold: int = DEFAULT_COMPACTION_THRESHOLD) -> None:
        """Initialize the context manager.

        Args:
            threshold: Token count that triggers compaction. Default 100,000.
        """
        self.threshold = threshold

    def needs_compaction(self, token_count: int) -> bool:
        """Check if token count exceeds threshold.

        Args:
            token_count: Current total token count of the conversation.

        Returns:
            True if token_count >= threshold.
        """
        return token_count >= self.threshold

    @property
    def compaction_prompt(self) -> str:
        """Get the prompt to trigger agent summarization.

        Returns:
            The compaction request message.
        """
        return DEFAULT_COMPACTION_PROMPT
