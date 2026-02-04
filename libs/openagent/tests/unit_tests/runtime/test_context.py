"""Tests for ContextManager."""

# ruff: noqa: PLR2004

from openagent.runtime.context import (
    DEFAULT_COMPACTION_PROMPT,
    DEFAULT_COMPACTION_THRESHOLD,
    ContextManager,
)


class TestContextManager:
    """Tests for ContextManager class."""

    def test_init_default_threshold(self) -> None:
        """Test default threshold initialization."""
        manager = ContextManager()
        assert manager.threshold == DEFAULT_COMPACTION_THRESHOLD

    def test_init_custom_threshold(self) -> None:
        """Test custom threshold initialization."""
        manager = ContextManager(threshold=50_000)
        assert manager.threshold == 50_000

    def test_needs_compaction_below_threshold(self) -> None:
        """Test needs_compaction returns False below threshold."""
        manager = ContextManager(threshold=100)
        assert manager.needs_compaction(50) is False
        assert manager.needs_compaction(99) is False

    def test_needs_compaction_at_threshold(self) -> None:
        """Test needs_compaction returns True at exact threshold."""
        manager = ContextManager(threshold=100)
        assert manager.needs_compaction(100) is True

    def test_needs_compaction_above_threshold(self) -> None:
        """Test needs_compaction returns True above threshold."""
        manager = ContextManager(threshold=100)
        assert manager.needs_compaction(101) is True
        assert manager.needs_compaction(1000) is True

    def test_compaction_prompt(self) -> None:
        """Test compaction_prompt returns expected prompt."""
        manager = ContextManager()
        assert manager.compaction_prompt == DEFAULT_COMPACTION_PROMPT

    def test_compaction_prompt_is_property(self) -> None:
        """Test compaction_prompt is a property (not settable)."""
        manager = ContextManager()
        # Just verify it's accessible as a property
        prompt = manager.compaction_prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0
