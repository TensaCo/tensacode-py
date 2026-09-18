"""A provider that returns a scene you hand it. For tests and for replaying recordings."""

from __future__ import annotations

from .protocol import PerceivedScene, Target


class FixtureProvider:
    name = "fixture"
    reliability = 0.5

    def __init__(self, scene: PerceivedScene, name: str = "fixture", reliability: float = 0.5) -> None:
        self.scene, self.name, self.reliability = scene, name, reliability

    def available(self) -> bool:
        return True

    def perceive(self, target: Target) -> PerceivedScene:
        return self.scene
