"""Agent configuration types.

Frozen dataclasses for configuring agent subsystems.
Each subsystem owns its config; AgentConfig composes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_COMPACTION_THRESHOLD = 100_000


@dataclass(frozen=True)
class SkillsConfig:
    """Configuration for the skills subsystem.

    Attributes:
        search_paths: Directories to scan for skill folders.
            Each directory should contain subdirectories with SKILL.md files.
    """

    search_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompactionConfig:
    """Configuration for context compaction.

    Attributes:
        threshold: Token count that triggers compaction.
    """

    threshold: int = DEFAULT_COMPACTION_THRESHOLD


@dataclass(frozen=True)
class AgentConfig:
    """Top-level agent configuration.

    Composes per-subsystem configs. All fields have sensible defaults,
    so ``AgentConfig()`` works with zero arguments.

    Examples:
        ```python
        config = AgentConfig(
            skills=SkillsConfig(search_paths=("/mnt/skills",)),
            compaction=CompactionConfig(threshold=80_000),
        )
        agent = await create_agent(computer, config=config)
        ```
    """

    skills: SkillsConfig = field(default_factory=SkillsConfig)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
