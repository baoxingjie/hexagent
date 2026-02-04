"""Tests for PermissionGate."""

# ruff: noqa: ARG002

import pytest

from openagent.runtime.permission import (
    PermissionDecision,
    PermissionGate,
    PermissionResult,
    SafetyRule,
)


class BlockDangerousCommand(SafetyRule):
    """Rule that blocks dangerous commands."""

    def check(
        self,
        tool_name: str,
        tool_args: dict[str, object],
    ) -> PermissionDecision | None:
        if tool_name == "bash":
            command = str(tool_args.get("command", ""))
            if "rm -rf" in command:
                return PermissionDecision(
                    result=PermissionResult.DENIED,
                    reason="Destructive command blocked",
                )
        return None


class RequireApprovalForSudo(SafetyRule):
    """Rule that requires approval for sudo commands."""

    def check(
        self,
        tool_name: str,
        tool_args: dict[str, object],
    ) -> PermissionDecision | None:
        if tool_name == "bash":
            command = str(tool_args.get("command", ""))
            if command.startswith("sudo"):
                return PermissionDecision(
                    result=PermissionResult.NEEDS_APPROVAL,
                    approval_prompt="Running sudo command requires approval",
                )
        return None


class AllowEverything(SafetyRule):
    """Rule that allows everything explicitly."""

    def check(
        self,
        tool_name: str,
        tool_args: dict[str, object],
    ) -> PermissionDecision | None:
        return PermissionDecision(result=PermissionResult.ALLOWED)


class TestPermissionResult:
    """Tests for PermissionResult enum."""

    def test_allowed_value(self) -> None:
        """Test ALLOWED has correct value."""
        assert PermissionResult.ALLOWED.value == "allowed"

    def test_denied_value(self) -> None:
        """Test DENIED has correct value."""
        assert PermissionResult.DENIED.value == "denied"

    def test_needs_approval_value(self) -> None:
        """Test NEEDS_APPROVAL has correct value."""
        assert PermissionResult.NEEDS_APPROVAL.value == "needs_approval"


class TestPermissionDecision:
    """Tests for PermissionDecision dataclass."""

    def test_minimal_decision(self) -> None:
        """Test creating a minimal decision."""
        decision = PermissionDecision(result=PermissionResult.ALLOWED)
        assert decision.result == PermissionResult.ALLOWED
        assert decision.reason is None
        assert decision.approval_prompt is None

    def test_decision_with_reason(self) -> None:
        """Test creating a decision with reason."""
        decision = PermissionDecision(
            result=PermissionResult.DENIED,
            reason="Command is dangerous",
        )
        assert decision.result == PermissionResult.DENIED
        assert decision.reason == "Command is dangerous"

    def test_decision_with_approval_prompt(self) -> None:
        """Test creating a decision requiring approval."""
        decision = PermissionDecision(
            result=PermissionResult.NEEDS_APPROVAL,
            approval_prompt="Do you want to proceed?",
        )
        assert decision.result == PermissionResult.NEEDS_APPROVAL
        assert decision.approval_prompt == "Do you want to proceed?"


class TestPermissionGate:
    """Tests for PermissionGate class."""

    @pytest.mark.asyncio
    async def test_no_rules_allows_all(self) -> None:
        """Test empty gate allows all."""
        gate = PermissionGate()
        decision = await gate.check("bash", {"command": "rm -rf /"})
        assert decision.result == PermissionResult.ALLOWED

    @pytest.mark.asyncio
    async def test_register_rule(self) -> None:
        """Test registering a rule."""
        gate = PermissionGate()
        gate.register_rule(BlockDangerousCommand())
        assert len(gate._rules) == 1

    @pytest.mark.asyncio
    async def test_rule_blocks_command(self) -> None:
        """Test rule can block command."""
        gate = PermissionGate()
        gate.register_rule(BlockDangerousCommand())

        decision = await gate.check("bash", {"command": "rm -rf /"})
        assert decision.result == PermissionResult.DENIED
        assert decision.reason == "Destructive command blocked"

    @pytest.mark.asyncio
    async def test_rule_allows_safe_command(self) -> None:
        """Test rule allows safe commands."""
        gate = PermissionGate()
        gate.register_rule(BlockDangerousCommand())

        decision = await gate.check("bash", {"command": "ls -la"})
        assert decision.result == PermissionResult.ALLOWED

    @pytest.mark.asyncio
    async def test_rule_requires_approval(self) -> None:
        """Test rule can require approval."""
        gate = PermissionGate()
        gate.register_rule(RequireApprovalForSudo())

        decision = await gate.check("bash", {"command": "sudo rm file"})
        assert decision.result == PermissionResult.NEEDS_APPROVAL
        assert decision.approval_prompt == "Running sudo command requires approval"

    @pytest.mark.asyncio
    async def test_first_matching_rule_wins(self) -> None:
        """Test first matching rule determines outcome."""
        gate = PermissionGate()
        gate.register_rule(BlockDangerousCommand())  # This matches first
        gate.register_rule(AllowEverything())  # This would allow, but comes second

        decision = await gate.check("bash", {"command": "rm -rf /"})
        assert decision.result == PermissionResult.DENIED

    @pytest.mark.asyncio
    async def test_rules_checked_in_order(self) -> None:
        """Test rules are checked in registration order."""
        gate = PermissionGate()
        gate.register_rule(AllowEverything())  # Allows everything first
        gate.register_rule(BlockDangerousCommand())  # Never reached

        decision = await gate.check("bash", {"command": "rm -rf /"})
        assert decision.result == PermissionResult.ALLOWED

    @pytest.mark.asyncio
    async def test_clear_rules(self) -> None:
        """Test clearing rules."""
        gate = PermissionGate()
        gate.register_rule(BlockDangerousCommand())
        gate.clear_rules()

        # Should allow now since no rules
        decision = await gate.check("bash", {"command": "rm -rf /"})
        assert decision.result == PermissionResult.ALLOWED

    @pytest.mark.asyncio
    async def test_rule_for_different_tool(self) -> None:
        """Test rule only applies to matching tool."""
        gate = PermissionGate()
        gate.register_rule(BlockDangerousCommand())  # Only checks bash

        # Should allow read tool even with suspicious args
        decision = await gate.check("read", {"file_path": "rm -rf"})
        assert decision.result == PermissionResult.ALLOWED
