"""Tests for SystemReminder."""

# ruff: noqa: ARG005, PLR2004

from dataclasses import dataclass

from openagent.runtime.reminder import Reminder, SystemReminder


@dataclass
class MockState:
    """Mock state for testing."""

    turn_count: int = 0
    has_error: bool = False


class TestReminder:
    """Tests for Reminder dataclass."""

    def test_reminder_creation(self) -> None:
        """Test creating a Reminder."""
        reminder = Reminder(
            condition=lambda s: True,
            text="Test reminder",
        )
        assert reminder.text == "Test reminder"
        assert reminder.condition(None) is True

    def test_reminder_is_frozen(self) -> None:
        """Test Reminder is immutable."""
        reminder = Reminder(
            condition=lambda s: True,
            text="Test reminder",
        )
        # frozen=True should prevent modification
        try:
            reminder.text = "New text"  # type: ignore[misc]
            msg = "Should have raised FrozenInstanceError"
            raise AssertionError(msg)
        except AttributeError:
            pass  # Expected


class TestSystemReminder:
    """Tests for SystemReminder class."""

    def test_init_default_tag(self) -> None:
        """Test default tag initialization."""
        system = SystemReminder()
        assert system.tag == "system-reminder"

    def test_init_custom_tag(self) -> None:
        """Test custom tag initialization."""
        system = SystemReminder(tag="agent-hint")
        assert system.tag == "agent-hint"

    def test_add_reminder(self) -> None:
        """Test adding a reminder."""
        system = SystemReminder()
        system.add(
            condition=lambda s: True,
            text="Test reminder",
        )
        assert len(system._reminders) == 1

    def test_clear_reminders(self) -> None:
        """Test clearing all reminders."""
        system = SystemReminder()
        system.add(condition=lambda s: True, text="Reminder 1")
        system.add(condition=lambda s: True, text="Reminder 2")
        assert len(system._reminders) == 2

        system.clear()
        assert len(system._reminders) == 0

    def test_check_no_reminders(self) -> None:
        """Test check returns empty list when no reminders registered."""
        system = SystemReminder()
        state = MockState()
        result = system.check(state)
        assert result == []

    def test_check_no_triggered(self) -> None:
        """Test check returns empty list when no conditions match."""
        system = SystemReminder()
        system.add(condition=lambda s: False, text="Never triggers")
        state = MockState()
        result = system.check(state)
        assert result == []

    def test_check_one_triggered(self) -> None:
        """Test check returns text when condition matches."""
        system = SystemReminder()
        system.add(condition=lambda s: True, text="Always triggers")
        state = MockState()
        result = system.check(state)
        assert result == ["Always triggers"]

    def test_check_multiple_triggered(self) -> None:
        """Test check returns all triggered texts."""
        system = SystemReminder()
        system.add(condition=lambda s: True, text="First")
        system.add(condition=lambda s: False, text="Never")
        system.add(condition=lambda s: True, text="Third")
        state = MockState()
        result = system.check(state)
        assert result == ["First", "Third"]

    def test_check_with_state_condition(self) -> None:
        """Test check evaluates condition against state."""
        system = SystemReminder()
        system.add(
            condition=lambda s: s.turn_count > 5,  # type: ignore[attr-defined]
            text="Many turns",
        )
        system.add(
            condition=lambda s: s.has_error,  # type: ignore[attr-defined]
            text="Has error",
        )

        # Neither condition met
        state = MockState(turn_count=3, has_error=False)
        assert system.check(state) == []

        # Only turn_count condition met
        state = MockState(turn_count=10, has_error=False)
        assert system.check(state) == ["Many turns"]

        # Both conditions met
        state = MockState(turn_count=10, has_error=True)
        assert system.check(state) == ["Many turns", "Has error"]

    def test_format_default_tag(self) -> None:
        """Test format wraps text in default tag."""
        system = SystemReminder()
        result = system.format("Test message")
        assert result == "<system-reminder>\nTest message\n</system-reminder>"

    def test_format_custom_tag(self) -> None:
        """Test format wraps text in custom tag."""
        system = SystemReminder(tag="hint")
        result = system.format("Test message")
        assert result == "<hint>\nTest message\n</hint>"

    def test_inject_no_triggered(self) -> None:
        """Test inject returns original message when nothing triggered."""
        system = SystemReminder()
        system.add(condition=lambda s: False, text="Never")
        state = MockState()
        result = system.inject("Hello world", state)
        assert result == "Hello world"

    def test_inject_one_triggered(self) -> None:
        """Test inject prepends single reminder."""
        system = SystemReminder()
        system.add(condition=lambda s: True, text="Remember this")
        state = MockState()
        result = system.inject("Hello world", state)
        expected = "<system-reminder>\nRemember this\n</system-reminder>\n\nHello world"
        assert result == expected

    def test_inject_multiple_triggered(self) -> None:
        """Test inject prepends multiple reminders."""
        system = SystemReminder()
        system.add(condition=lambda s: True, text="First reminder")
        system.add(condition=lambda s: True, text="Second reminder")
        state = MockState()
        result = system.inject("Hello world", state)
        expected = "<system-reminder>\nFirst reminder\n</system-reminder>\n\n<system-reminder>\nSecond reminder\n</system-reminder>\n\nHello world"
        assert result == expected

    def test_inject_empty_reminders(self) -> None:
        """Test inject returns original when no reminders registered."""
        system = SystemReminder()
        state = MockState()
        result = system.inject("Hello world", state)
        assert result == "Hello world"
