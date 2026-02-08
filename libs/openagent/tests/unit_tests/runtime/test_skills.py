"""Tests for SkillResolver."""

# ruff: noqa: PLR2004, ASYNC109, ARG001

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from openagent.runtime.skills import SkillResolver, _parse_frontmatter
from openagent.types import CLIResult

# --- Frontmatter parsing tests ---


class TestParseFrontmatter:
    """Tests for _parse_frontmatter()."""

    def test_valid_frontmatter(self) -> None:
        raw = "---\nname: pdf\ndescription: Generate PDFs\n---\n\nBody content here."
        meta, body = _parse_frontmatter(raw)
        assert meta["name"] == "pdf"
        assert meta["description"] == "Generate PDFs"
        assert body == "Body content here."

    def test_missing_opening_delimiter(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            _parse_frontmatter("name: pdf\n---\nBody")

    def test_missing_closing_delimiter(self) -> None:
        with pytest.raises(ValueError, match="missing closing"):
            _parse_frontmatter("---\nname: pdf\n")

    def test_empty_body(self) -> None:
        raw = "---\nname: test\ndescription: test\n---\n"
        meta, body = _parse_frontmatter(raw)
        assert meta["name"] == "test"
        assert body == ""

    def test_multiline_body(self) -> None:
        raw = "---\nname: x\n---\n\nLine 1\nLine 2\nLine 3"
        _meta, body = _parse_frontmatter(raw)
        assert "Line 1" in body
        assert "Line 3" in body


# --- SkillResolver tests ---


def _make_computer(
    run_results: dict[str, CLIResult],
) -> AsyncMock:
    """Create a mock Computer that returns results based on command content."""
    computer = AsyncMock()

    async def mock_run(command: str, *, timeout: float | None = None) -> CLIResult:
        for key, result in run_results.items():
            if key in command:
                return result
        return CLIResult(stdout="", stderr="not found", exit_code=1)

    computer.run = mock_run
    computer.is_running = True
    return computer


VALID_SKILL_MD = """\
---
name: pdf
description: Generate PDF documents
---

When generating PDFs, use the template system."""


class TestSkillResolverDiscover:
    """Tests for SkillResolver.discover()."""

    async def test_discover_finds_skills(self) -> None:
        computer = _make_computer(
            {
                "find /mnt/skills": CLIResult(stdout="/mnt/skills/pdf\n"),
                "cat /mnt/skills/pdf/SKILL.md": CLIResult(stdout=VALID_SKILL_MD),
            }
        )

        resolver = SkillResolver(computer, search_paths=("/mnt/skills",))
        skills = await resolver.discover()

        assert len(skills) == 1
        assert skills[0].name == "pdf"
        assert skills[0].description == "Generate PDF documents"
        assert skills[0].path == "/mnt/skills/pdf"

    async def test_discover_skips_dirs_without_skill_md(self) -> None:
        computer = _make_computer(
            {
                "find /mnt/skills": CLIResult(stdout="/mnt/skills/pdf\n/mnt/skills/empty\n"),
                "cat /mnt/skills/pdf/SKILL.md": CLIResult(stdout=VALID_SKILL_MD),
                "cat /mnt/skills/empty/SKILL.md": CLIResult(exit_code=1, stderr="No such file"),
            }
        )

        resolver = SkillResolver(computer, search_paths=("/mnt/skills",))
        skills = await resolver.discover()

        assert len(skills) == 1
        assert skills[0].name == "pdf"

    async def test_discover_empty_search_path(self) -> None:
        computer = _make_computer(
            {
                "find /mnt/skills": CLIResult(exit_code=1, stderr="No such directory"),
            }
        )

        resolver = SkillResolver(computer, search_paths=("/mnt/skills",))
        skills = await resolver.discover()

        assert skills == []

    async def test_discover_skips_invalid_frontmatter(self) -> None:
        computer = _make_computer(
            {
                "find /mnt/skills": CLIResult(stdout="/mnt/skills/bad\n"),
                "cat /mnt/skills/bad/SKILL.md": CLIResult(stdout="no frontmatter here"),
            }
        )

        resolver = SkillResolver(computer, search_paths=("/mnt/skills",))
        skills = await resolver.discover()

        assert skills == []

    async def test_discover_multiple_search_paths(self) -> None:
        skill_a = "---\nname: a\ndescription: Skill A\n---\nBody A"
        skill_b = "---\nname: b\ndescription: Skill B\n---\nBody B"
        computer = _make_computer(
            {
                "find /path1": CLIResult(stdout="/path1/a\n"),
                "find /path2": CLIResult(stdout="/path2/b\n"),
                "cat /path1/a/SKILL.md": CLIResult(stdout=skill_a),
                "cat /path2/b/SKILL.md": CLIResult(stdout=skill_b),
            }
        )

        resolver = SkillResolver(computer, search_paths=("/path1", "/path2"))
        skills = await resolver.discover()

        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"a", "b"}


class TestSkillResolverLoadContent:
    """Tests for SkillResolver.load_content()."""

    async def test_load_content_returns_wrapped_body(self) -> None:
        computer = _make_computer(
            {
                "find /mnt/skills": CLIResult(stdout="/mnt/skills/pdf\n"),
                "cat /mnt/skills/pdf/SKILL.md": CLIResult(stdout=VALID_SKILL_MD),
            }
        )

        resolver = SkillResolver(computer, search_paths=("/mnt/skills",))
        await resolver.discover()

        content = await resolver.load_content("pdf")
        assert content.startswith("Base directory for this skill: /mnt/skills/pdf")
        assert "When generating PDFs" in content

    async def test_load_content_caches(self) -> None:
        call_count = 0
        original_md = VALID_SKILL_MD

        computer = AsyncMock()

        async def mock_run(command: str, *, timeout: float | None = None) -> CLIResult:
            nonlocal call_count
            if "find" in command:
                return CLIResult(stdout="/mnt/skills/pdf\n")
            if "cat" in command:
                call_count += 1
                return CLIResult(stdout=original_md)
            return CLIResult(exit_code=1)

        computer.run = mock_run
        computer.is_running = True

        resolver = SkillResolver(computer, search_paths=("/mnt/skills",))
        await resolver.discover()

        # discover calls cat once, then load_content calls cat once more
        first_cat_count = call_count  # noqa: F841

        await resolver.load_content("pdf")
        second_cat_count = call_count

        await resolver.load_content("pdf")  # cached — no new cat call
        third_cat_count = call_count

        assert third_cat_count == second_cat_count  # cache hit

    async def test_load_content_unknown_skill_raises(self) -> None:
        computer = _make_computer(
            {
                "find /mnt/skills": CLIResult(stdout=""),
            }
        )

        resolver = SkillResolver(computer, search_paths=("/mnt/skills",))
        await resolver.discover()

        with pytest.raises(KeyError, match="not discovered"):
            await resolver.load_content("nonexistent")
