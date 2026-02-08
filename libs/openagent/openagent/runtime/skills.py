"""Skill discovery and content loading.

SkillResolver scans configured directories on a Computer for skill
folders, parses SKILL.md frontmatter for metadata, and lazily loads
skill content with caching.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from openagent.types import Skill

if TYPE_CHECKING:
    from openagent.computer.base import Computer

logger = logging.getLogger(__name__)

_SKILL_FILENAME = "SKILL.md"


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    """Parse YAML-like frontmatter from a SKILL.md file.

    Supports only simple ``key: value`` lines between ``---`` delimiters.
    Returns (metadata_dict, body_text).

    Args:
        raw: The full SKILL.md content.

    Returns:
        A tuple of (frontmatter dict, markdown body after the closing ---).

    Raises:
        ValueError: If the file does not start with ``---`` or has no closing delimiter.
    """
    stripped = raw.strip()
    if not stripped.startswith("---"):
        msg = "SKILL.md must start with '---' frontmatter delimiter"
        raise ValueError(msg)

    # Find closing ---
    end_idx = stripped.find("---", 3)
    if end_idx == -1:
        msg = "SKILL.md missing closing '---' frontmatter delimiter"
        raise ValueError(msg)

    frontmatter_block = stripped[3:end_idx].strip()
    body = stripped[end_idx + 3 :].strip()

    metadata: dict[str, str] = {}
    for raw_line in frontmatter_block.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            continue
        colon_idx = stripped_line.find(":")
        if colon_idx == -1:
            continue
        key = stripped_line[:colon_idx].strip()
        value = stripped_line[colon_idx + 1 :].strip()
        metadata[key] = value

    return metadata, body


class SkillResolver:
    r"""Discovers skills from the filesystem and lazily loads their content.

    Uses the Computer protocol to run filesystem commands, making it
    work with both LocalNativeComputer and RemoteE2BComputer.

    Examples:
        ```python
        resolver = SkillResolver(computer, search_paths=("/mnt/skills",))
        skills = await resolver.discover()
        # skills == [Skill(name="pdf", description="...", path="/mnt/skills/pdf")]

        content = await resolver.load_content("pdf")
        # content == "Base directory for this skill: /mnt/skills/pdf\n\n..."
        ```
    """

    def __init__(
        self,
        computer: Computer,
        search_paths: tuple[str, ...] | list[str],
    ) -> None:
        """Initialize the resolver.

        Args:
            computer: The Computer instance for filesystem access.
            search_paths: Directories to scan for skill folders.
        """
        self._computer = computer
        self._search_paths = tuple(search_paths)
        self._skills: dict[str, Skill] = {}
        self._content_cache: dict[str, str] = {}

    @property
    def search_paths(self) -> tuple[str, ...]:
        """The configured search paths."""
        return self._search_paths

    async def discover(self) -> list[Skill]:
        """Scan search paths for skill directories and parse metadata.

        Each subdirectory containing a SKILL.md file is treated as a skill.
        The SKILL.md frontmatter must contain ``name`` and ``description``.

        Returns:
            List of discovered Skill objects.
        """
        self._skills.clear()
        discovered: list[Skill] = []

        for base_path in self._search_paths:
            # List subdirectories
            result = await self._computer.run(f"find {base_path} -maxdepth 1 -mindepth 1 -type d 2>/dev/null")
            if result.exit_code != 0 or not result.stdout.strip():
                continue

            for raw_dir in sorted(result.stdout.strip().splitlines()):
                skill_dir = raw_dir.strip()
                if not skill_dir:
                    continue

                skill_file = f"{skill_dir}/{_SKILL_FILENAME}"

                # Read SKILL.md
                cat_result = await self._computer.run(f"cat {skill_file} 2>/dev/null")
                if cat_result.exit_code != 0 or not cat_result.stdout.strip():
                    logger.debug("Skipping %s: no SKILL.md found", skill_dir)
                    continue

                try:
                    metadata, _body = _parse_frontmatter(cat_result.stdout)
                except ValueError:
                    logger.warning(
                        "Skipping %s: invalid SKILL.md frontmatter",
                        skill_dir,
                    )
                    continue

                name = metadata.get("name")
                description = metadata.get("description")
                if not name or not description:
                    logger.warning(
                        "Skipping %s: SKILL.md missing 'name' or 'description'",
                        skill_dir,
                    )
                    continue

                skill = Skill(name=name, description=description, path=skill_dir)
                self._skills[name] = skill
                discovered.append(skill)

        return discovered

    async def load_content(self, name: str) -> str:
        r"""Load the skill's markdown body, with caching.

        Returns the content wrapped with the skill's base directory:
        ``Base directory for this skill: {path}\n\n{body}``

        Args:
            name: The skill name (must have been discovered first).

        Returns:
            The formatted skill content ready for injection as a user message.

        Raises:
            KeyError: If the skill name was not discovered.
            RuntimeError: If the SKILL.md content cannot be read.
        """
        if name in self._content_cache:
            return self._content_cache[name]

        if name not in self._skills:
            msg = f"Skill not discovered: {name}"
            raise KeyError(msg)

        skill = self._skills[name]
        skill_file = f"{skill.path}/{_SKILL_FILENAME}"

        result = await self._computer.run(f"cat {skill_file}")
        if result.exit_code != 0:
            msg = f"Failed to read {skill_file}: {result.stderr}"
            raise RuntimeError(msg)

        try:
            _metadata, body = _parse_frontmatter(result.stdout)
        except ValueError as exc:
            msg = f"Failed to parse {skill_file}: {exc}"
            raise RuntimeError(msg) from exc

        content = f"Base directory for this skill: {skill.path}\n\n{body}"
        self._content_cache[name] = content
        return content
