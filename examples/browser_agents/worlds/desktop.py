"""The desktop-chore world: one Linux machine, a terminal, and a note on the Desktop.

The note is part of the world definition rather than something written into a running
simulator, so an episode's starting state is exactly its definition plus its seed.
"""

from __future__ import annotations

import json
from pathlib import Path

BASE = Path(__file__).parent / "desktop.json"
DESKTOP_BASE = json.loads(BASE.read_text())

MACHINE = "dev"
USER = "agent"
HOME = f"/home/{USER}"


def desktop_world(files: dict[str, str] | None = None, *, theme: str | None = None) -> dict:
    """The base definition with extra files placed in the user's home."""
    definition = json.loads(json.dumps(DESKTOP_BASE))
    computer = definition["computers"][0]
    computer["initial_files"] = {**computer.get("initial_files", {}), **(files or {})}
    if theme:
        definition.setdefault("metadata", {}).setdefault("desktop_themes", {})[MACHINE] = theme
    return definition


def note_world(note: str, seed: int, **kwargs) -> dict:
    """A world whose Desktop holds one task note, named after the episode's seed."""
    return desktop_world({f"Desktop/task-{seed}.txt": note.rstrip("\n") + "\n"}, **kwargs)
