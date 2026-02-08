"""Tests for prompt content loading."""

# ruff: noqa: PLR2004

from collections.abc import Generator

import pytest

from openagent.prompts import content


@pytest.fixture(autouse=True)
def _clear_caches() -> Generator[None]:
    """Clear content caches after each test for isolation."""
    yield
    content.load.cache_clear()
    content._scan_package_keys.cache_clear()


class TestLoad:
    """Tests for content.load()."""

    def test_load_existing(self) -> None:
        result = content.load("system_prompt_identity")
        assert isinstance(result, str)
        assert "OpenAgent" in result

    def test_load_missing_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Prompt fragment not found"):
            content.load("nonexistent_key_xyz")

    def test_load_system_prompt_fragments(self) -> None:
        keys = [
            "system_prompt_identity",
            "system_prompt_doing_tasks",
            "system_prompt_executing_actions_with_care",
            "system_prompt_tone_and_style",
            "system_prompt_tool_usage_policy",
            "system_prompt_git_status",
            "system_prompt_scratchpad_directory",
        ]
        for key in keys:
            result = content.load(key)
            assert isinstance(result, str)
            assert len(result) > 0, f"Fragment '{key}' is empty"

    def test_load_tool_instruction_fragments(self) -> None:
        keys = [
            "tool_instruction_bash",
            "tool_instruction_bash_git_commit_and_pr_creation_instructions",
            "tool_instruction_edit",
            "tool_instruction_glob",
            "tool_instruction_grep",
            "tool_instruction_read",
            "tool_instruction_write",
            "tool_instruction_web_search",
            "tool_instruction_web_fetch",
            "tool_instruction_skill",
        ]
        for key in keys:
            result = content.load(key)
            assert isinstance(result, str)
            assert len(result) > 0, f"Fragment '{key}' is empty"

    def test_load_user_prompt_fragments(self) -> None:
        for key in ("user_prompt_compaction_request", "user_prompt_compaction_summary_rebuild"):
            result = content.load(key)
            assert isinstance(result, str)
            assert len(result) > 0, f"Fragment '{key}' is empty"

    def test_temporary_dir_excluded(self) -> None:
        all_keys = content.find("")
        assert not any(k.startswith("_temporary") for k in all_keys)


class TestFind:
    """Tests for content.find()."""

    def test_find_by_prefix(self) -> None:
        result = content.find("system_prompt_")
        assert len(result) == 7

    def test_find_tool_supplements(self) -> None:
        result = content.find("tool_instruction_bash_")
        assert "tool_instruction_bash_git_commit_and_pr_creation_instructions" in result

    def test_find_no_match(self) -> None:
        result = content.find("nonexistent_prefix_")
        assert result == []

    def test_find_sorted(self) -> None:
        result = content.find("system_prompt_")
        assert result == sorted(result)
