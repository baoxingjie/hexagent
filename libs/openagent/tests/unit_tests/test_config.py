"""Tests for AgentConfig."""

# ruff: noqa: PLR2004

from __future__ import annotations

import pytest

from openagent.config import AgentConfig, CompactionConfig, SkillsConfig
from openagent.runtime.context import DEFAULT_COMPACTION_THRESHOLD


class TestSkillsConfig:
    """Tests for SkillsConfig."""

    def test_default_empty_paths(self) -> None:
        cfg = SkillsConfig()
        assert cfg.search_paths == ()

    def test_custom_paths(self) -> None:
        cfg = SkillsConfig(search_paths=("/mnt/skills", "/home/user/skills"))
        assert cfg.search_paths == ("/mnt/skills", "/home/user/skills")

    def test_frozen(self) -> None:
        cfg = SkillsConfig()
        with pytest.raises(AttributeError):
            cfg.search_paths = ("/new",)  # type: ignore[misc]


class TestCompactionConfig:
    """Tests for CompactionConfig."""

    def test_default_threshold(self) -> None:
        cfg = CompactionConfig()
        assert cfg.threshold == DEFAULT_COMPACTION_THRESHOLD

    def test_custom_threshold(self) -> None:
        cfg = CompactionConfig(threshold=50_000)
        assert cfg.threshold == 50_000


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_all_defaults(self) -> None:
        cfg = AgentConfig()
        assert cfg.skills.search_paths == ()
        assert cfg.compaction.threshold == DEFAULT_COMPACTION_THRESHOLD

    def test_nested_config(self) -> None:
        cfg = AgentConfig(
            skills=SkillsConfig(search_paths=("/mnt/skills",)),
            compaction=CompactionConfig(threshold=80_000),
        )
        assert cfg.skills.search_paths == ("/mnt/skills",)
        assert cfg.compaction.threshold == 80_000
