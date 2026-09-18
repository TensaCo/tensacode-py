"""Owning a computerworld world: actor sessions for agents, privileged reads for scoring.

The harness holds the ``World``. An agent only ever gets a ``Surface`` built from an actor
environment with the grants it needs (pointer, keyboard, application; semantic and pixel
observation). Setup and scoring use a separate privileged session — the engine's own
owner/actor split — so an agent cannot read or write through the channel that grades it.

    world = CwWorld(note_world(note, seed), seed)
    ui = CwBody(world.actor(), CwProvider(), episode="desktop-7")
    ...
    world.read(f"{root}/README.md")      # scoring, not visible to the agent
    world.commits(root)                  # [(message, hash), ...] newest first
    world.state_hash()                   # exact reproducibility of the whole episode
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..perception.cw_body import Surface

ACTOR_ACTIONS = ("pointer.v1", "keyboard.v1", "application.v1")
ACTOR_OBSERVATIONS = ("semantic.v1", "pixels.v1")
OWNER_ACTIONS = ("terminal.v1", "filesystem.v1")


@dataclass
class Shell:
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class CwWorld:
    """One episode's world, owned by the harness."""

    def __init__(self, definition: dict, seed: int = 0, *, machine: str = "dev", user: str = "agent", width: int = 1280, height: int = 800) -> None:
        from computerworld import World

        self.definition, self.seed = definition, seed
        self.machine, self.user = machine, user
        self.width, self.height = width, height
        self.world = World(definition, seed)
        self._owner = self.world.environment({
            "actor": user, "machines": [machine],
            "actions": list(OWNER_ACTIONS), "observations": ["terminal.v1"],
        })

    # -- agent side

    def actor(self, *, actions: tuple[str, ...] = ACTOR_ACTIONS, observations: tuple[str, ...] = ACTOR_OBSERVATIONS) -> Surface:
        env = self.world.environment({
            "actor": self.user, "machines": [self.machine],
            "actions": list(actions), "observations": list(observations),
        })
        return Surface(env, self.machine, user=self.user, width=self.width, height=self.height)

    # -- owner side (setup and scoring)

    def shell(self, command: str) -> Shell:
        outcome = self._owner.step([{"family": "terminal.v1", "op": "execute", "machine": self.machine, "payload": {"command": command}}])["outcomes"][0]
        if not outcome.get("success"):
            return Shell(127, "", str(outcome.get("error")))
        value = outcome.get("value") or {}
        return Shell(int(value.get("exit_code", 0)), value.get("stdout", ""), value.get("stderr", ""))

    def _fs(self, op: str, path: str):
        outcome = self._owner.step([{"family": "filesystem.v1", "op": op, "machine": self.machine, "payload": {"path": path}}])["outcomes"][0]
        return outcome.get("value") if outcome.get("success") else None

    def read(self, path: str) -> str | None:
        got = self._fs("read", path)
        return got.get("content") if isinstance(got, dict) else None

    def entries(self, path: str) -> list[str] | None:
        got = self._fs("list", path)
        return list(got) if isinstance(got, list) else None

    def stat(self, path: str) -> dict | None:
        got = self._fs("stat", path)
        return got if isinstance(got, dict) else None

    def exists(self, path: str) -> bool:
        return self.stat(path) is not None

    def commits(self, root: str) -> list[tuple[str, str]]:
        """(message, hash) for a repository's commits, newest first. [] when there is no repo."""
        log = self.shell(f"cd {root} && git log")
        if not log.ok:
            return []
        commits, current = [], None
        for line in log.stdout.splitlines():
            if line.startswith("commit "):
                current = line.split(" ", 1)[1].strip()
            elif line.startswith("    ") and current:
                commits.append((line.strip(), current))
                current = None
        return commits

    def pending(self, root: str) -> list[str]:
        """Paths git reports as added/modified/staged; empty when the tree is clean."""
        status = self.shell(f"cd {root} && git status")
        return [line.strip() for line in status.stdout.splitlines()[1:] if line.strip()] if status.ok else []

    # -- reproducibility

    def state_hash(self) -> str:
        return self.world.state_hash()

    def snapshot(self):
        return self.world.snapshot()

    def restore(self, checkpoint) -> None:
        self.world.restore(checkpoint)

    def fork(self, checkpoint=None):
        branch = self.world.fork(checkpoint if checkpoint is not None else self.world.snapshot())
        clone = object.__new__(CwWorld)
        clone.definition, clone.seed = self.definition, self.seed
        clone.machine, clone.user = self.machine, self.user
        clone.width, clone.height = self.width, self.height
        clone.world = branch
        clone._owner = branch.environment({"actor": self.user, "machines": [self.machine], "actions": list(OWNER_ACTIONS), "observations": ["terminal.v1"]})
        return clone

    def trajectory(self):
        return self.world.trajectory()


HOME_RE = re.compile(r"^~(?=/|$)")


def expand(path: str, user: str = "agent") -> str:
    return HOME_RE.sub(f"/home/{user}", path)
