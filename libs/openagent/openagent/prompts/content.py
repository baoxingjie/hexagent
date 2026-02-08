"""Prompt content loading.

Read-only loader for .md fragment files from the ``prompts/`` package
directory via ``importlib.resources`` for wheel compatibility.  Keys are
derived from filenames: ``system_prompt_identity.md`` →
``"system_prompt_identity"``.

This module has no mutable state.  All results are cached for the
lifetime of the process (content files don't change during a session).
"""

from __future__ import annotations

import functools
import importlib.resources


@functools.cache
def _scan_package_keys() -> frozenset[str]:
    """Discover all .md fragment keys in the package."""
    package = importlib.resources.files("openagent.prompts")
    keys: set[str] = set()
    for item in package.iterdir():
        if item.is_file() and item.name.endswith(".md"):
            keys.add(item.name[:-3])
    return frozenset(keys)


@functools.cache
def load(key: str) -> str:
    """Load a prompt fragment by key.

    Results are cached — repeated calls with the same key return the
    same string without hitting the filesystem.

    Args:
        key: Fragment key (e.g. ``"system_prompt_identity"``).

    Returns:
        The fragment text.

    Raises:
        KeyError: If no fragment with the given key exists.
    """
    package = importlib.resources.files("openagent.prompts")
    resource = package.joinpath(f"{key}.md")
    try:
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError) as exc:
        msg = f"Prompt fragment not found: '{key}'"
        raise KeyError(msg) from exc


def find(prefix: str) -> list[str]:
    """Find all fragment keys matching a prefix, sorted.

    Args:
        prefix: Key prefix to match (e.g. ``"tool_instruction_bash_"``).

    Returns:
        Sorted list of matching keys.
    """
    return sorted(k for k in _scan_package_keys() if k.startswith(prefix))
